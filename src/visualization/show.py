import cv2
import numpy as np

from data_classes.frame import FrameData

from utils.utils import FPSCounter


class Show:
    def __init__(self, conf, traffic_rois: list[np.ndarray]):
        self.conf = conf

        self.draw_person_way = conf["show"]["draw_person_way"]
        self.fps_buffer = conf["show"]["fps_buffer"]
        self.draw_roi = conf["show"]["draw_roi"]
        self.show = conf["show"]["show"]

        self.traffic_rois = traffic_rois

        if self.fps_buffer > 1:
            self.fps_counter = FPSCounter(self.fps_buffer)
        else:
            self.fps_counter = None

    def process(self, frame_data: FrameData) -> FrameData:
        frame_data.frame_out = frame_data.frame.copy()
        self.draw_frame_info(frame_data)

        track_info = zip(frame_data.track_xyxy, frame_data.track_id, frame_data.track_cls, frame_data.track_conf)
        for bbox, id_, cls, conf in track_info:
            if cls == "person":
                self.draw_person(frame_data, bbox, id_, cls, conf)

            elif cls == "car":
                self.draw_car(frame_data, bbox, id_, cls, conf)

            else:
                self.draw_other(frame_data, bbox, id_, cls, conf)


        if self.draw_roi:
            for roi in self.traffic_rois:
                cv2.polylines(frame_data.frame_out, [roi], True, (0, 255, 0), 2)

        if self.show:
            cv2.imshow('frame', frame_data.frame_out)
            cv2.waitKey(1)

        return frame_data

    @staticmethod
    def draw_box(frame: np.ndarray, xyxy: list[int], text: str, color: tuple[int, int, int]) -> np.ndarray:
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        cv2.putText(
            frame,
            text,
            (xyxy[0] + 5, xyxy[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        return frame

    def draw_person(self, frame_data: FrameData, bbox, id_, cls, conf):
        people = frame_data.people[id_]
        if people.num_dangers_frames > 2:
            color = (0, 255, 255)

        elif people.crash:
            color = (0, 0, 255)
            cv2.putText(frame_data.frame_out, "Detected accident", (40, 160), 1, 5, (0, 0, 255),5)
        else:
            color = (0, 255, 0)

        if self.draw_person_way:
            for line_id in range(len(people.points) - 1):
                cv2.line(frame_data.frame_out, people.points[line_id], people.points[line_id + 1], color, 2)

        self.draw_box(frame_data.frame_out, bbox, f"{cls} {round(conf, 2)}", color)


    def draw_car(self, frame_data: FrameData, bbox, id_, cls, conf):
        self.draw_box(frame_data.frame_out, bbox, f"{cls} {round(conf, 2)}", (255, 0, 0))

    def draw_other(self, frame_data: FrameData, bbox, id_, cls, conf):
        self.draw_box(frame_data.frame_out, bbox, f"{id_} {cls} {round(conf, 2)}", self.get_color(id_))

    def draw_frame_info(self, frame_data: FrameData):
        height, width = frame_data.frame_out.shape[:2]
        scale_w = (width // 100) or 1
        scale_h = (height // 100) or 1

        if self.fps_counter is not None:
            cv2.putText(
                frame_data.frame_out,
                f"FPS {self.fps_counter.get_fps():.1f}",
                (scale_w * 2, scale_h * 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.15 * scale_h,
                (0, 255, 0),
                scale_h // 2
            )

    @staticmethod
    def get_color(id_: int) -> tuple[int, int, int]:
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Purple
            (0, 255, 255),  # Blue
            (255, 20, 147),  # Deep Pink
            (255, 165, 0),  # Orange
            (32, 178, 170),  # Light Sea
            (148, 0, 211)  # Dark Violet
        ]

        return colors[id_ % 10]