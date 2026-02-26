# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides the OpenCVCamera class for capturing frames from cameras using OpenCV.
"""

import logging
import math
import os
import platform
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

# Fix MSMF hardware transform compatibility for Windows before importing cv2
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import numpy as np
import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import depthai as dai
from datetime import timedelta

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_backend, get_cv2_rotation
from .configuration_oakd import ColorMode, OAKDCameraConfig

# NOTE(Steven): The maximum opencv device index depends on your operating system. For instance,
# if you have 3 cameras, they should be associated to index 0, 1, and 2. This is the case
# on MacOS. However, on Ubuntu, the indices are different like 6, 16, 23.
# When you change the USB port or reboot the computer, the operating system might
# treat the same cameras as new devices. Thus we select a higher bound to search indices.
MAX_OPENCV_INDEX = 60

logger = logging.getLogger(__name__)


class OAKDCamera(Camera):
    """
    Manages camera interactions using OpenCV for efficient frame recording.

    This class provides a high-level interface to connect to, configure, and read
    frames from cameras compatible with OpenCV's VideoCapture. It supports both
    synchronous and asynchronous frame reading.

    An OpenCVCamera instance requires a camera index (e.g., 0) or a device path
    (e.g., '/dev/video0' on Linux). Camera indices can be unstable across reboots
    or port changes, especially on Linux. Use the provided utility script to find
    available camera indices or paths:
    ```bash
    lerobot-find-cameras opencv
    ```

    The camera's default settings (FPS, resolution, color mode) are used unless
    overridden in the configuration.

    Example:
        ```python
        from lerobot.cameras.opencv import OpenCVCamera
        from lerobot.cameras.configuration_opencv import OpenCVCameraConfig

        # Basic usage with camera index 0
        config = OpenCVCameraConfig(index_or_path=0)
        camera = OpenCVCamera(config)
        camera.connect()

        # Read 1 frame synchronously (blocking)
        color_image = camera.read()

        # Read 1 frame asynchronously (waits for new frame with a timeout)
        async_image = camera.async_read()

        # Get the latest frame immediately (no wait, returns timestamp)
        latest_image, timestamp = camera.read_latest()

        # When done, properly disconnect the camera using
        camera.disconnect()
        ```
    """

    def __init__(self, config: OAKDCameraConfig):
        """
        Initializes the OpenCVCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config
        self.index_or_path = config.index_or_path

        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        
        #self.device: dai.Device | None = None
        # DepthAI v3 API では dai.Device ではなく pipeline そのものでライフサイクルを管理します
        self.pipeline: dai.Pipeline | None = None
        self.q_sync: dai.DataOutputQueue | None = None
        
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.backend: int = get_cv2_backend()

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and opened."""
        #return isinstance(self.videocapture, cv2.VideoCapture) and self.videocapture.isOpened()
        #return self.device is not None and not self.device.isClosed()
        return self.pipeline is not None and self.pipeline.isRunning()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """
        Builds the pipeline, starts it, and launches the read thread.
        """
        self.pipeline = self._build_pipeline()

        try:
            # DepthAI v3: デバイスの起動は pipeline.start() で行う
            self.pipeline.start()
            logger.info(f"{self} connected and pipeline started successfully.")
        except RuntimeError as e:
            raise ConnectionError(f"Failed to connect to OAK-D or start pipeline: {e}")

        self._start_read_thread()

        if warmup and self.warmup_s > 0:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.async_read(timeout_ms=self.warmup_s * 1000)
            logger.info(f"{self} warmup completed.")

    def _build_pipeline(self) -> dai.Pipeline:
        """DepthAI pipeline creation for v3 API"""
        pipeline = dai.Pipeline()

        # 1. RGB Camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setIspScale(self.width, 1920) 
        cam_rgb.setFps(self.fps)
        cam_rgb.setInterleaved(False)

        # 2. Stereo Depth
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setFps(self.fps)

        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setFps(self.fps)

        stereo = pipeline.create(dai.node.StereoDepth)
        #stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        if self.config.align_depth_to_color:
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # 3. Sync node
        sync = pipeline.create(dai.node.Sync)
        sync.setRunOnHost(True) # ホスト側での同期処理を有効化 (v3 API)
        
        cam_rgb.video.link(sync.inputs["rgb"])
        stereo.depth.link(sync.inputs["depth"])

        # 4. Output Queue (XLinkOutは不要になり、直接キューを作成)
        self.q_sync = sync.out.createOutputQueue()

        return pipeline
    
    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        """
        Applies the specified FOURCC, FPS, width, and height settings to the connected camera.

        This method attempts to set the camera properties via OpenCV. It checks if
        the camera successfully applied the settings and raises an error if not.
        FOURCC is set first (if specified) as it can affect the available FPS and resolution options.

        Args:
            fourcc: The desired FOURCC code (e.g., "MJPG", "YUYV"). If None, auto-detect.
            fps: The desired frames per second. If None, the setting is skipped.
            width: The desired capture width. If None, the setting is skipped.
            height: The desired capture height. If None, the setting is skipped.

        Raises:
            RuntimeError: If the camera fails to set any of the specified properties
                          to the requested value.
            DeviceNotConnectedError: If the camera is not connected.
        """

        # Set FOURCC first (if specified) as it can affect available FPS/resolution options
        if self.config.fourcc is not None:
            self._validate_fourcc()
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        default_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if self.width is None or self.height is None:
            self.width, self.height = default_width, default_height
            self.capture_width, self.capture_height = default_width, default_height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = default_height, default_width
                self.capture_width, self.capture_height = default_width, default_height
        else:
            self._validate_width_and_height()

        if self.fps is None:
            self.fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        else:
            self._validate_fps()

    def _validate_fps(self) -> None:
        """Validates and sets the camera's frames per second (FPS)."""

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        if self.fps is None:
            raise ValueError(f"{self} FPS is not set")

        success = self.videocapture.set(cv2.CAP_PROP_FPS, float(self.fps))
        actual_fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        # Use math.isclose for robust float comparison
        if not success or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            raise RuntimeError(f"{self} failed to set fps={self.fps} ({actual_fps=}).")

    def _validate_fourcc(self) -> None:
        """Validates and sets the camera's FOURCC code."""

        fourcc_code = cv2.VideoWriter_fourcc(*self.config.fourcc)

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        success = self.videocapture.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        actual_fourcc_code = self.videocapture.get(cv2.CAP_PROP_FOURCC)

        # Convert actual FOURCC code back to string for comparison
        actual_fourcc_code_int = int(actual_fourcc_code)
        actual_fourcc = "".join([chr((actual_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)])

        if not success or actual_fourcc != self.config.fourcc:
            logger.warning(
                f"{self} failed to set fourcc={self.config.fourcc} (actual={actual_fourcc}, success={success}). "
                f"Continuing with default format."
            )

    def _validate_width_and_height(self) -> None:
        """Validates and sets the camera's frame capture width and height."""

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        if self.capture_width is None or self.capture_height is None:
            raise ValueError(f"{self} capture_width or capture_height is not set")

        width_success = self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        height_success = self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

        actual_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if not width_success or self.capture_width != actual_width:
            raise RuntimeError(
                f"{self} failed to set capture_width={self.capture_width} ({actual_width=}, {width_success=})."
            )

        actual_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not height_success or self.capture_height != actual_height:
            raise RuntimeError(
                f"{self} failed to set capture_height={self.capture_height} ({actual_height=}, {height_success=})."
            )

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available OAK-D cameras using DepthAI API.
        """
        found_cameras_info = []
        # DepthAIのAPIを使って接続可能な全デバイスを検索
        available_devices = dai.Device.getAllAvailableDevices()
        print("available_devices", available_devices)
        for idx, device_info in enumerate(available_devices):
            # OAK-Dの固有ID（MxId）を取得。設定ファイルではこれを index_or_path として扱うと確実です。
            #mxid = device_info.getMxId() 
            mxid = 99
            camera_info = {
                "name": f"OAK-D Camera @ {mxid}",
                "type": "OAKD",
                "id": mxid, 
                "backend_api": "DepthAI",
                "default_stream_profile": {
                    "format": "RGB+Depth",
                    "fourcc": "",
                    "width": 640,
                    "height": 360,
                    "fps": 30.0,
                },
            }
            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _read_from_hardware(self) -> NDArray[Any]:
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        ret, frame = self.videocapture.read()

        if not ret:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        return frame

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Reads a single frame synchronously from the camera.

        This is a blocking call. It waits for the next available frame from the
        camera hardware via OpenCV.

        Returns:
            np.ndarray: The captured frame as a NumPy array in the format
                       (height, width, channels), using the specified or default
                       color mode and applying any configured rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading the frame from the camera fails or if the
                          received frame dimensions don't match expectations before rotation.
            ValueError: If an invalid `color_mode` is requested.
        """

        start_time = time.perf_counter()

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        frame = self.async_read(timeout_ms=10000)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        """
        Applies color conversion, dimension validation, and rotation to a raw frame.

        Args:
            image (np.ndarray): The raw image frame (expected BGR format from OpenCV).

        Returns:
            np.ndarray: The processed image frame.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions do not match the configured
                          `width` and `height`.
        """

        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        if self.color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

# --- バックグラウンドスレッドの処理 ---
    def _read_loop(self) -> None:
        """バックグラウンドでOAK-Dからデータを取得し続ける"""
        if self.stop_event is None:
            raise RuntimeError("stop_event is not initialized.")

        while not self.stop_event.is_set():
            try:
                # パイプラインが停止している場合はループを抜ける
                if not self.pipeline.isRunning():
                    break

                # ブロッキングでデバイスから同期済みのメッセージグループを取得
                sync_msg = self.q_sync.get()
                
                # v3 API では辞書型のようにアクセス可能
                in_rgb = sync_msg["rgb"]
                in_depth = sync_msg["depth"]
                # OpenCV(NumPy)形式に変換
                frame_rgb = in_rgb.getCvFrame() # BGR
                frame_depth = in_depth.getFrame() # uint16 (ミリメートル)

                # RGBへの変換
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                
                capture_time = time.perf_counter()

                # ロックをかけて最新フレームを更新
                with self.frame_lock:
                    self.latest_frames = {
                        "color": frame_rgb,
                        "depth": frame_depth
                    }
                    self.latest_timestamp = capture_time
                
                # メインスレッドに「新しいフレームが来た」ことを通知
                self.new_frame_event.set()

            except RuntimeError as e:
                logger.warning(f"Error reading frame from OAK-D: {e}")
                break

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name="DepthAICamera_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> dict[str, np.ndarray]:
        """Asynchronous wait-for-new-frame read."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from {self}.")

        with self.frame_lock:
            frames = self.latest_frames
            self.new_frame_event.clear()

        if frames is None:
            raise RuntimeError(f"No frames available for {self}.")
        
        #print("frames", frames)
        
        return frames["color"]
    
    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> dict[str, np.ndarray]:
        """待たずに最新のフレームを取得する（Peek）"""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError("Read thread is not running.")

        with self.frame_lock:
            frames = self.latest_frames
            timestamp = self.latest_timestamp

        if frames is None or timestamp is None:
            raise RuntimeError("Has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"Latest frame is too old: {age_ms:.1f} ms.")

        return frames

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        # DepthAI v3の安全な終了処理
        if self.pipeline is not None:
            if hasattr(self.pipeline, "stop"):
                self.pipeline.stop()
            self.pipeline = None

        with self.frame_lock:
            self.latest_frames = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
