from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


class VideoWriter:
    def __init__(self, config, project_dir: Path, filepath: str = None):
        self.config = config["video_writer"]
        self.fps = self.config["fps"]
        self.fourcc = self.config["fourcc"]
        self.skip_frames = self.config["skip_frames"]
        self.segment_size = self.config["segment_size"]

        output_path = Path(self.config["output_path"]).expanduser()
        self.output_path = output_path if output_path.is_absolute() else project_dir / output_path
        Path.mkdir(self.output_path, parents=True, exist_ok=True)

        self.frames_in_segment = 0
        self.total_frames_processed = 0

        self.segment_frames = int(self.fps * self.segment_size)

        self.video_path = filepath
        self.writer: cv2.VideoWriter | None = None


    def create_new_writer(self, width: int, height: int) -> cv2.VideoWriter:
        if self.video_path:
            filepath = self.video_path

        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_path / f"{timestamp}.mkv"

        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)  # noqa
        writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (width, height))
        return writer

    def process(self, frame: np.ndarray) -> None:
        if frame is None:
            return

        self.total_frames_processed += 1
        if (self.total_frames_processed % self.skip_frames) != 0:
            return

        if not self.writer:
            self.writer = self.create_new_writer(frame.shape[1], frame.shape[0])

        self.writer.write(frame)
        self.frames_in_segment += 1

        if self.frames_in_segment >= self.segment_frames:
            self.close_current_writer()
            self.frames_in_segment = 0

    def close_current_writer(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __del__(self):
        self.close_current_writer()
