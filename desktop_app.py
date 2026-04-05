from __future__ import annotations

import argparse
import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
MPL_CONFIG_DIR = PROJECT_ROOT / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import hand_landmarker as hand_landmarker_module

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None


@dataclass(frozen=True)
class ColorChoice:
    name: str
    bgr: tuple[int, int, int]


@dataclass(frozen=True)
class ToolChoice:
    name: str
    thickness: int
    alpha: float
    glow: bool = False
    pencil_style: bool = False


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    fps: int = 30
    show_preview: bool = True
    fullscreen_preview: bool = True
    enable_virtual_camera: bool = False
    virtual_backend: str | None = None
    virtual_device: str | None = None


def point_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def angle_between(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> float:
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


class AirWriterApp:
    WINDOW_NAME = "Air Writer Studio"
    CONTROL_HAND_LABEL = "Left"
    HELPER_HAND_LABEL = "Right"

    COLOR_CHOICES = [
        ColorChoice("Ink", (112, 54, 10)),
        ColorChoice("Pine", (28, 82, 24)),
        ColorChoice("Wine", (18, 18, 132)),
        ColorChoice("Ochre", (0, 92, 150)),
        ColorChoice("Onyx", (24, 24, 24)),
    ]

    TOOL_CHOICES = [
        ToolChoice("Pen", thickness=4, alpha=1.0),
        ToolChoice("Marker", thickness=9, alpha=1.0),
        ToolChoice("Pencil", thickness=2, alpha=1.0, pencil_style=True),
        ToolChoice("Neon", thickness=5, alpha=1.0, glow=True),
    ]

    FINGER_LANDMARKS = {
        "index": (5, 6, 8),
        "middle": (9, 10, 12),
        "ring": (13, 14, 16),
        "pinky": (17, 18, 20),
    }

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model_path = PROJECT_ROOT / "models" / "hand_landmarker.task"
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Missing model bundle: {self.model_path}. Download the hand_landmarker.task model first."
            )

        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.selected_color = self.COLOR_CHOICES[0]
        self.selected_tool = self.TOOL_CHOICES[0]
        self.canvas: np.ndarray | None = None
        self.previous_point: tuple[int, int] | None = None
        self.last_screenshot_at = 0.0
        self.last_color_select_at = 0.0
        self.last_tool_select_at = 0.0
        self.active_mode_label = "Waiting for left hand"
        self.status_message = "Show your left hand to begin"
        self.status_until = time.time() + 3
        self.capture_dir = PROJECT_ROOT / "captures"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.color_palette_boxes: list[tuple[tuple[int, int, int, int], ColorChoice]] = []
        self.tool_palette_boxes: list[tuple[tuple[int, int, int, int], ToolChoice]] = []
        self.screen_size = self.get_screen_size()
        self.size_scale = 1.0
        self.size_adjust_active = False
        self.virtual_camera: Any | None = None

    def get_screen_size(self) -> tuple[int, int]:
        try:
            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            return 1366, 768

    def start_virtual_camera(self, frame_shape: tuple[int, int, int]) -> None:
        if not self.config.enable_virtual_camera or self.virtual_camera is not None:
            return
        if pyvirtualcam is None:
            raise RuntimeError(
                "Virtual camera mode requires pyvirtualcam. Install dependencies from requirements.txt and try again."
            )

        height, width = frame_shape[:2]
        try:
            self.virtual_camera = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=self.config.fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                backend=self.config.virtual_backend,
                device=self.config.virtual_device,
            )
            self.set_status(f"Virtual camera ready: {self.virtual_camera.device}", duration=3.0)
        except RuntimeError as exc:
            raise RuntimeError(
                "Could not start a virtual camera backend. Install OBS Studio (OBS Virtual Camera) or Unity Capture, then launch again."
            ) from exc

    def send_virtual_camera_frame(self, frame: np.ndarray) -> None:
        if self.virtual_camera is None:
            return
        self.virtual_camera.send(frame)
        self.virtual_camera.sleep_until_next_frame()

    def set_status(self, message: str, duration: float = 1.7) -> None:
        self.status_message = message
        self.status_until = time.time() + duration

    def normalized_to_pixel(
        self,
        landmark: Any,
        frame_shape: tuple[int, int, int],
    ) -> tuple[int, int]:
        height, width = frame_shape[:2]
        x = min(max(int(landmark.x * width), 0), width - 1)
        y = min(max(int(landmark.y * height), 0), height - 1)
        return x, y

    def get_finger_states(
        self,
        landmarks: list[Any],
        frame_shape: tuple[int, int, int],
    ) -> dict[str, bool]:
        wrist = self.normalized_to_pixel(landmarks[0], frame_shape)
        states: dict[str, bool] = {}
        for name, (mcp_idx, pip_idx, tip_idx) in self.FINGER_LANDMARKS.items():
            mcp = self.normalized_to_pixel(landmarks[mcp_idx], frame_shape)
            pip = self.normalized_to_pixel(landmarks[pip_idx], frame_shape)
            tip = self.normalized_to_pixel(landmarks[tip_idx], frame_shape)
            is_straight = angle_between(mcp, pip, tip) > 150
            reaches_out = point_distance(wrist, tip) > point_distance(wrist, pip) * 1.12
            states[name] = is_straight and reaches_out
        return states

    def classify_gesture(self, finger_states: dict[str, bool]) -> str:
        index_up = finger_states["index"]
        middle_up = finger_states["middle"]
        ring_up = finger_states["ring"]
        pinky_up = finger_states["pinky"]

        if index_up and not middle_up and not ring_up and not pinky_up:
            return "draw"
        if middle_up and not index_up and not ring_up and not pinky_up:
            return "color"
        if ring_up and not index_up and not middle_up and not pinky_up:
            return "tool"
        if pinky_up and not index_up and not middle_up and not ring_up:
            return "screenshot"
        if index_up and middle_up and ring_up and pinky_up:
            return "erase"
        return "idle"

    def ensure_canvas(self, frame: np.ndarray) -> None:
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)

    def smooth_point(self, point: tuple[int, int]) -> tuple[int, int]:
        if self.previous_point is None:
            return point
        x = int(self.previous_point[0] * 0.5 + point[0] * 0.5)
        y = int(self.previous_point[1] * 0.5 + point[1] * 0.5)
        return x, y

    def get_tool_thickness(self, tool: ToolChoice) -> int:
        return max(1, int(round(tool.thickness * self.size_scale)))

    def get_eraser_thickness(self) -> int:
        return max(12, int(round(24 * self.size_scale)))

    def get_pointer_radius(self, base_size: int) -> int:
        return max(4, min(18, base_size // 2 + 2))

    def draw_visible_line(
        self,
        canvas: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        outline_thickness = max(thickness + 2, thickness)
        cv2.line(canvas, start, end, (225, 225, 225), outline_thickness, cv2.LINE_AA)
        cv2.line(canvas, start, end, color, thickness, cv2.LINE_AA)

    def draw_segment(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        tool: ToolChoice,
        thickness: int,
    ) -> None:
        if self.canvas is None:
            return

        if tool.pencil_style:
            self.draw_visible_line(self.canvas, start, end, color, thickness)
            return

        if tool.alpha < 1.0:
            overlay = np.zeros_like(self.canvas)
            self.draw_visible_line(overlay, start, end, color, thickness)
            cv2.addWeighted(overlay, tool.alpha, self.canvas, 1.0, 0.0, dst=self.canvas)
        else:
            self.draw_visible_line(self.canvas, start, end, color, thickness)

        if tool.glow:
            glow = np.zeros_like(self.canvas)
            cv2.line(glow, start, end, color, max(4, thickness * 2), cv2.LINE_AA)
            glow = cv2.GaussianBlur(glow, (0, 0), 8)
            cv2.addWeighted(glow, 0.28, self.canvas, 1.0, 0.0, dst=self.canvas)
            cv2.line(self.canvas, start, end, color, max(2, thickness - 2), cv2.LINE_AA)

    def erase_segment(self, start: tuple[int, int], end: tuple[int, int], thickness: int) -> None:
        if self.canvas is None:
            return
        cv2.line(self.canvas, start, end, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.circle(self.canvas, end, max(8, thickness // 2), (0, 0, 0), -1)

    def blend_canvas(self, frame: np.ndarray) -> np.ndarray:
        if self.canvas is None:
            return frame.copy()
        return cv2.addWeighted(frame, 1.0, self.canvas, 1.0, 0.0)

    def save_screenshot(self, composed_frame: np.ndarray) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = self.capture_dir / f"airwriter-{timestamp}.png"
        cv2.imwrite(str(output_path), composed_frame)
        self.set_status(f"Screenshot saved: {output_path.name}", duration=2.4)

    def pointer_inside(
        self, point: tuple[int, int], rect: tuple[int, int, int, int]
    ) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= point[0] <= x2 and y1 <= point[1] <= y2

    def control_selector_points(
        self,
        control_hand: list[Any],
        gesture: str,
        frame_shape: tuple[int, int, int],
    ) -> list[tuple[int, int]]:
        if gesture == "color":
            return [self.normalized_to_pixel(control_hand[12], frame_shape)]
        if gesture == "tool":
            return [self.normalized_to_pixel(control_hand[16], frame_shape)]
        return []

    def draw_color_palette(self, frame: np.ndarray, active: bool) -> None:
        self.color_palette_boxes = []
        if not active:
            return

        box_width = 108
        box_height = 52
        margin_left = 28
        top = 98
        gap = 10

        for idx, choice in enumerate(self.COLOR_CHOICES):
            left = margin_left + idx * (box_width + gap)
            rect = (left, top, left + box_width, top + box_height)
            self.color_palette_boxes.append((rect, choice))

            fill_color = tuple(min(channel + 6, 255) for channel in choice.bgr)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), fill_color, -1)
            border = (255, 255, 255) if choice == self.selected_color else (50, 50, 50)
            thickness = 3 if choice == self.selected_color else 2
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), border, thickness)
            cv2.putText(
                frame,
                choice.name,
                (rect[0] + 12, rect[1] + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "Color mode: use the left middle fingertip to choose.",
            (28, top + box_height + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def draw_tool_palette(self, frame: np.ndarray, active: bool) -> None:
        self.tool_palette_boxes = []
        if not active:
            return

        box_width = 126
        box_height = 50
        gap = 10
        total_width = len(self.TOOL_CHOICES) * box_width + (len(self.TOOL_CHOICES) - 1) * gap
        start_x = max((frame.shape[1] - total_width) // 2, 16)
        top = frame.shape[0] - 86

        for idx, tool in enumerate(self.TOOL_CHOICES):
            left = start_x + idx * (box_width + gap)
            rect = (left, top, left + box_width, top + box_height)
            self.tool_palette_boxes.append((rect, tool))

            background = (35, 42, 54) if tool != self.selected_tool else (70, 88, 112)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), background, -1)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (235, 235, 235), 2)
            cv2.putText(
                frame,
                tool.name,
                (rect[0] + 18, rect[1] + 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "Tool mode: point the left ring fingertip at a tool.",
            (28, top - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def apply_palette_selection(
        self,
        selector_points: Iterable[tuple[int, int]],
        items: list[tuple[tuple[int, int, int, int], object]],
        kind: str,
    ) -> None:
        current_time = time.time()
        for point in selector_points:
            for rect, item in items:
                if not self.pointer_inside(point, rect):
                    continue
                if kind == "color" and current_time - self.last_color_select_at >= 0.35:
                    self.selected_color = item  # type: ignore[assignment]
                    self.last_color_select_at = current_time
                    self.set_status(f"Color changed to {self.selected_color.name}")
                    return
                if kind == "tool" and current_time - self.last_tool_select_at >= 0.35:
                    self.selected_tool = item  # type: ignore[assignment]
                    self.last_tool_select_at = current_time
                    self.set_status(f"Tool changed to {self.selected_tool.name}")
                    return

    def draw_pointer(
        self,
        frame: np.ndarray,
        point: tuple[int, int],
        color: tuple[int, int, int],
        radius: int = 6,
    ) -> None:
        inner_radius = max(2, radius // 3)
        cv2.circle(frame, point, radius, color, 1, cv2.LINE_AA)
        cv2.circle(frame, point, inner_radius, color, -1, cv2.LINE_AA)

    def update_size_from_helper_hand(
        self,
        frame: np.ndarray,
        helper_hand: list[Any] | None,
    ) -> None:
        self.size_adjust_active = False
        if helper_hand is None:
            return

        helper_states = self.get_finger_states(helper_hand, frame.shape)
        is_size_mode = (
            helper_states["index"]
            and not helper_states["middle"]
            and not helper_states["ring"]
            and not helper_states["pinky"]
        )
        if not is_size_mode:
            return

        self.size_adjust_active = True
        thumb_tip = self.normalized_to_pixel(helper_hand[4], frame.shape)
        index_tip = self.normalized_to_pixel(helper_hand[8], frame.shape)
        distance = point_distance(index_tip, thumb_tip)
        mapped_scale = float(np.interp(distance, [10.0, 180.0], [0.45, 3.2]))
        self.size_scale = self.size_scale * 0.7 + mapped_scale * 0.3

        preview_radius = self.get_pointer_radius(self.get_tool_thickness(self.selected_tool))
        cv2.line(frame, index_tip, thumb_tip, (255, 215, 140), 2, cv2.LINE_AA)
        self.draw_pointer(frame, index_tip, (255, 255, 255), radius=preview_radius)
        self.draw_pointer(frame, thumb_tip, (255, 255, 255), radius=preview_radius)
        cv2.putText(
            frame,
            "Size",
            (min(index_tip[0], thumb_tip[0]), max(28, min(index_tip[1], thumb_tip[1]) - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 245, 220),
            2,
            cv2.LINE_AA,
        )

    def render_hud(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (18, 18), (520, 138), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.42, frame, 0.58, 0.0, frame)

        cv2.putText(
            frame,
            "Air Writer Studio",
            (34, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Mode: {self.active_mode_label}",
            (34, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (238, 238, 238),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Tool: {self.selected_tool.name}",
            (34, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Left hand draws. Right hand thumb+index pinch/spread changes size.",
            (560, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "C clear  Q quit",
            (560, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (560, 92), (598, 130), self.selected_color.bgr, -1)
        cv2.rectangle(frame, (560, 92), (598, 130), (255, 255, 255), 2)
        cv2.putText(
            frame,
            self.selected_color.name,
            (612, 119),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Brush {self.get_tool_thickness(self.selected_tool)} px",
            (780, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Eraser {self.get_eraser_thickness()} px",
            (780, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        if self.size_adjust_active:
            cv2.putText(
                frame,
                "Size adjust active",
                (980, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 235, 170),
                1,
                cv2.LINE_AA,
            )
        if self.config.enable_virtual_camera:
            device_label = (
                self.virtual_camera.device if self.virtual_camera is not None else "starting..."
            )
            cv2.putText(
                frame,
                f"Virtual cam: {device_label}",
                (1100, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (235, 245, 235),
                1,
                cv2.LINE_AA,
            )

        if time.time() <= self.status_until:
            cv2.putText(
                frame,
                self.status_message,
                (30, frame.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (250, 250, 250),
                2,
                cv2.LINE_AA,
            )

    def draw_hand_mesh(
        self,
        frame: np.ndarray,
        landmarks: list[Any],
        label: str,
    ) -> None:
        connection_color = (125, 210, 255) if label == "Right" else (178, 255, 196)
        point_color = (255, 255, 255)

        for connection in hand_landmarker_module.HandLandmarksConnections.HAND_CONNECTIONS:
            start = self.normalized_to_pixel(landmarks[connection.start], frame.shape)
            end = self.normalized_to_pixel(landmarks[connection.end], frame.shape)
            cv2.line(frame, start, end, connection_color, 2, cv2.LINE_AA)

        for idx, landmark in enumerate(landmarks):
            point = self.normalized_to_pixel(landmark, frame.shape)
            radius = 4 if idx in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, point, radius, point_color, -1, cv2.LINE_AA)
            cv2.circle(frame, point, radius + 2, connection_color, 1, cv2.LINE_AA)

        wrist = self.normalized_to_pixel(landmarks[0], frame.shape)
        cv2.putText(
            frame,
            label,
            (wrist[0] - 18, max(18, wrist[1] - 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            connection_color,
            2,
            cv2.LINE_AA,
        )

    def handle_control_hand(
        self,
        display_frame: np.ndarray,
        raw_frame: np.ndarray,
        control_hand: list[Any] | None,
        gesture: str,
    ) -> np.ndarray:
        if control_hand is None:
            self.previous_point = None
            self.active_mode_label = "Waiting for left hand"
            return display_frame

        selector_points = self.control_selector_points(
            control_hand, gesture, display_frame.shape
        )

        index_tip = self.normalized_to_pixel(control_hand[8], display_frame.shape)
        draw_point = self.smooth_point(index_tip)
        brush_thickness = self.get_tool_thickness(self.selected_tool)
        eraser_thickness = self.get_eraser_thickness()

        if gesture == "draw":
            self.active_mode_label = "Draw"
            self.draw_pointer(
                display_frame,
                draw_point,
                self.selected_color.bgr,
                radius=self.get_pointer_radius(brush_thickness),
            )
            if self.previous_point is not None:
                self.draw_segment(
                    self.previous_point,
                    draw_point,
                    self.selected_color.bgr,
                    self.selected_tool,
                    brush_thickness,
                )
            self.previous_point = draw_point
        elif gesture == "erase":
            self.active_mode_label = "Erase"
            self.draw_pointer(
                display_frame,
                draw_point,
                (255, 255, 255),
                radius=self.get_pointer_radius(eraser_thickness),
            )
            if self.previous_point is not None:
                self.erase_segment(self.previous_point, draw_point, eraser_thickness)
            self.previous_point = draw_point
        else:
            self.previous_point = None
            if gesture == "color":
                self.active_mode_label = "Color select"
                self.apply_palette_selection(selector_points, self.color_palette_boxes, "color")
                for point in selector_points:
                    self.draw_pointer(display_frame, point, (255, 255, 255))
            elif gesture == "tool":
                self.active_mode_label = "Tool select"
                self.apply_palette_selection(selector_points, self.tool_palette_boxes, "tool")
                for point in selector_points:
                    self.draw_pointer(display_frame, point, (255, 255, 255))
            elif gesture == "screenshot":
                self.active_mode_label = "Screenshot"
                now = time.time()
                if now - self.last_screenshot_at >= 1.6:
                    self.last_screenshot_at = now
                    shot_frame = self.blend_canvas(raw_frame.copy())
                    self.save_screenshot(shot_frame)
            else:
                self.active_mode_label = "Idle"

        return display_frame

    def run(self) -> None:
        camera = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(self.config.camera_index)
        if not camera.isOpened():
            raise RuntimeError("Could not open the default camera.")

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, self.config.fps)

        if self.config.show_preview:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            if self.config.fullscreen_preview:
                cv2.setWindowProperty(
                    self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )

        try:
            while True:
                success, frame = camera.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                self.ensure_canvas(frame)
                self.start_virtual_camera(frame.shape)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.hands.detect_for_video(mp_image, int(time.monotonic() * 1000))

                display = frame.copy()
                control_hand: list[Any] | None = None
                helper_hand: list[Any] | None = None
                if results.hand_landmarks and results.handedness:
                    for hand_landmarks, handedness in zip(
                        results.hand_landmarks, results.handedness
                    ):
                        label = handedness[0].category_name
                        if label == self.CONTROL_HAND_LABEL:
                            control_hand = hand_landmarks
                        elif label == self.HELPER_HAND_LABEL:
                            helper_hand = hand_landmarks
                        self.draw_hand_mesh(display, hand_landmarks, label)

                gesture = "idle"
                if control_hand is not None:
                    states = self.get_finger_states(control_hand, display.shape)
                    gesture = self.classify_gesture(states)

                self.update_size_from_helper_hand(display, helper_hand)

                self.draw_color_palette(display, active=gesture == "color")
                self.draw_tool_palette(display, active=gesture == "tool")

                display = self.handle_control_hand(display, frame, control_hand, gesture)
                display = self.blend_canvas(display)
                self.render_hud(display)
                self.send_virtual_camera_frame(display)

                if self.config.show_preview:
                    screen_width, screen_height = self.screen_size
                    preview_frame = cv2.resize(
                        display,
                        (screen_width, screen_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    cv2.imshow(self.WINDOW_NAME, preview_frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = -1

                if key == ord("q"):
                    break
                if key == ord("c") and self.canvas is not None:
                    self.canvas[:] = 0
                    self.set_status("Canvas cleared")
        finally:
            camera.release()
            if self.virtual_camera is not None:
                self.virtual_camera.close()
            if self.config.show_preview:
                cv2.destroyAllWindows()
            self.hands.close()


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Air Writer Studio")
    parser.add_argument("--camera-index", type=int, default=0, help="Physical camera index.")
    parser.add_argument("--fps", type=int, default=30, help="Target capture and virtual camera fps.")
    parser.add_argument(
        "--virtual-camera",
        action="store_true",
        help="Publish the Air Writer output to a virtual camera for Zoom, Teams, Meet, and similar apps.",
    )
    parser.add_argument(
        "--virtual-backend",
        default=None,
        help="Optional pyvirtualcam backend name, such as obs or unitycapture.",
    )
    parser.add_argument(
        "--virtual-device",
        default=None,
        help="Optional virtual camera device name.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without the local preview window.",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Show the preview in a normal resizable window instead of fullscreen.",
    )
    args = parser.parse_args()
    return AppConfig(
        camera_index=args.camera_index,
        fps=args.fps,
        show_preview=not args.no_preview,
        fullscreen_preview=not args.windowed,
        enable_virtual_camera=args.virtual_camera,
        virtual_backend=args.virtual_backend,
        virtual_device=args.virtual_device,
    )


if __name__ == "__main__":
    AirWriterApp(parse_args()).run()
