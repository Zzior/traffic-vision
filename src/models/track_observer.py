import cv2
import numpy as np

from data_classes.frame import FrameData
from data_classes.track import Person, Car

from utils.utils import detect_motion_anomalies


class TrackObserver:
    def __init__(self, config, traffic_rois: list[np.ndarray]):
        self.track_buffer = config["detection"]["track_buffer"]

        self.traffic_rois = traffic_rois

        self.people: dict[int, Person] = {}
        self.cars: dict[int, Car] = {}

    def process(self, frame_data: FrameData) -> FrameData:
        updated = set()

        for id_, track_id in enumerate(frame_data.track_id):
            updated.add(id_)

            object_class = frame_data.track_cls[id_]
            if object_class == "person":
                self.update_person(track_id, frame_data.track_xyxy[id_])

            elif object_class in ("bus", "car", "truck", "train"):
                self.update_cars(track_id, frame_data.track_xyxy[id_])

        self.delete_objects(updated)

        frame_data.people = self.people
        frame_data.cars = self.cars
        return frame_data



    def update_person(self, track_id, xyxy):
        person = self.people.setdefault(track_id, Person())
        person.num_disappearances = 0

        point = self.calc_bottom_point(xyxy)
        person.points.append(point)

        person.l_points.append((xyxy[0], xyxy[3]))
        person.r_points.append((xyxy[1], xyxy[3]))

        in_danger_zone = False
        for roi in self.traffic_rois:
            if self.check_intersection_polygon(point, roi):
                in_danger_zone = True

        if in_danger_zone:
            person.num_dangers_frames += 1
        else:
            person.num_dangers_frames = 0

        if not person.crash and self.detect_crash(person):
            person.crash = True

    def detect_crash(self, person: Person) -> bool:
        car_intersected = None
        for car in self.cars.values():
            if (
                    self.check_intersection_box(person.l_points[-1], car.box)
                    or self.check_intersection_box(person.points[-1], car.box)
            ):
                car_intersected = car
                break

        if not car_intersected:
            return False

        mov = self.check_mov(car_intersected.points)

        if not mov:
            return False

        anomalies = detect_motion_anomalies(person.points[-5:])
        if len(anomalies) < 3:
            return False

        return True


    def update_cars(self, track_id, xyxy):
        car = self.cars.setdefault(track_id, Car(xyxy))
        car.num_disappearances = 0

        point = self.calc_central_point(xyxy)
        car.points.append(point)
        car.box = xyxy


    def delete_objects(self, updated: set[int]):
        for id_, person in list(self.people.items()):
            if id_ not in updated:
                person.num_disappearances += 1

            if person.num_disappearances >= self.track_buffer:
                del self.people[id_]


        for id_, car in list(self.cars.items()):
            if id_ not in updated:
                car.num_disappearances += 1

            if car.num_disappearances >= self.track_buffer:
                del self.cars[id_]


    @staticmethod
    def calc_central_point(bbox: list[int]) -> tuple[int, int]:
        return (
            (bbox[0] + bbox[2]) // 2,
            (bbox[1] + bbox[3]) // 2,
        )

    @staticmethod
    def calc_bottom_point(bbox: list[int]) -> tuple[int, int]:
        return (
            (bbox[0] + bbox[2]) // 2,
            bbox[3],
        )

    @staticmethod
    def check_intersection_box(obj_c_point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
        x, y = obj_c_point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def check_intersection_polygon(obj_c_point: tuple[int, int], roi: np.ndarray) -> bool:
        intersection = cv2.pointPolygonTest(roi, obj_c_point, False)

        if intersection >= 0:
            return True

        else:
            return False

    @staticmethod
    def check_mov(c_points: list[tuple[int, int]], interval: int = 3, max_iter: int = 3, min_movement: int = 20) -> bool:
        point_size = len(c_points)
        if point_size < 2:
            return False

        if point_size > max_iter * interval + 1:
            c_points = c_points[-(max_iter * interval + 1):]

        for i in range(max_iter):
            idx1 = -(i * interval + 1)
            idx2 = -(i * interval + interval + 1)
            if abs(idx2) > len(c_points):
                break

            x1, y1 = c_points[idx1]
            x2, y2 = c_points[idx2]

            dx = abs(x1 - x2)
            dy = abs(y1 - y2)

            if dx >= min_movement or dy >= min_movement:
                return True

        return False