from pathlib import Path

import hydra

import numpy as np

from models.notify import Notify
from models.video_reader import VideoReader
from models.track_observer import TrackObserver
from models.detection_tracking import DetectionTracking

from visualization.show import Show
from visualization.web import VideoServer
from visualization.video_writer import VideoWriter

project_dir = Path(__file__).parent.parent
conf_dir = project_dir / "configs"


def should_render_output(config: dict) -> bool:
    return (
        config["show"]["show"] or
        config["web_stream"]["show"] or
        config["video_writer"]["write"]
    )


@hydra.main(version_base=None, config_path=conf_dir.__str__(), config_name="config")
def main(config) -> None:
    traffic_rois: list[np.ndarray] = []
    for roi in config["source_info"]["traffic_roi"]:
        traffic_rois.append(np.array(roi, dtype=np.int32))

    notify = Notify(config, project_dir)
    video_reader = VideoReader(str(project_dir / config["source_info"]["src"]))
    detection_tracking = DetectionTracking(config, project_dir)
    track_observer = TrackObserver(config, traffic_rois)

    # Init visualization
    render_output = should_render_output(config)
    video_writer = VideoWriter(config, project_dir)
    show = Show(config, traffic_rois)
    web = VideoServer(config)

    if config["web_mov"]["show"]:
        web.run()

    for frame_data in video_reader.process():
        frame_data = detection_tracking.process(frame_data)
        frame_data = track_observer.process(frame_data)

        if render_output:
            frame_data = show.process(frame_data)
            if config["web_mov"]["show"]:
                web.update_image(frame_data.frame_out)

            if config["video_writer"]["write"]:
                video_writer.process(frame_data.frame_out)

        notify.process(frame_data)


if __name__ == "__main__":
    main()
