from typing import Sequence, cast
import numpy as np
import cv2
import cv2.aruco as aruco
import math
from cv2.typing import MatLike
from enum import Enum


class ArucoDictionary(Enum):
    Dict_4X4_50 = aruco.DICT_4X4_50
    Dict_4X4_100 = aruco.DICT_4X4_100
    Dict_4X4_250 = aruco.DICT_4X4_250
    Dict_4X4_1000 = aruco.DICT_4X4_1000
    Dict_5X5_50 = aruco.DICT_5X5_50
    Dict_5X5_100 = aruco.DICT_5X5_100
    Dict_5X5_250 = aruco.DICT_5X5_250
    Dict_5X5_1000 = aruco.DICT_5X5_1000
    Dict_6X6_50 = aruco.DICT_6X6_50
    Dict_6X6_100 = aruco.DICT_6X6_100
    Dict_6X6_250 = aruco.DICT_6X6_250
    Dict_6X6_1000 = aruco.DICT_6X6_1000
    Dict_7X7_50 = aruco.DICT_7X7_50
    Dict_7X7_100 = aruco.DICT_7X7_100
    Dict_7X7_250 = aruco.DICT_7X7_250
    Dict_7X7_1000 = aruco.DICT_7X7_1000
    Dict_APRILTAG_16h5 = aruco.DICT_APRILTAG_16h5
    Dict_APRILTAG_25h9 = aruco.DICT_APRILTAG_25h9
    Dict_APRILTAG_36h10 = aruco.DICT_APRILTAG_36h10
    Dict_APRILTAG_36h11 = aruco.DICT_APRILTAG_36h11
    Dict_ArUco_ORIGINAL = aruco.DICT_ARUCO_ORIGINAL


FONT = cv2.FONT_HERSHEY_SIMPLEX
DICTIONARY = ArucoDictionary.Dict_4X4_50
MARKER_LENGTH = 0.1
AXIS_LENGTH = 0.05
dictionary = aruco.getPredefinedDictionary(DICTIONARY.value)


def estimate_pose(frame: MatLike, mtx: MatLike, dist: MatLike):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters()

    # https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
    # https://docs.opencv.org/4.x/d2/d1a/classcv_1_1aruco_1_1ArucoDetector.html
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # pylint: disable-next=unpacking-non-sequence
    marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(gray)

    # 如果检测到ArUco标记
    if marker_ids is not None and len(marker_ids) > 0:
        for i in range(len(marker_ids)):
            # aruco.estimatePoseSingleMarkers is deprecated
            # use cv::solvePnP instead
            # pylint: disable-next=unpacking-non-sequence
            rvec, tvec, op = aruco.estimatePoseSingleMarkers(
                cast(Sequence[MatLike], marker_corners[i]), MARKER_LENGTH, mtx, dist
            )
            # 将旋转向量（rvec）转换为旋转矩阵（R）3*3
            R, _ = cv2.Rodrigues(rvec)
            # 旋转矩阵是正交矩阵，所以它的逆矩阵等于它的转置 R_inv 表示从相机坐标系到标记坐标系的旋转。
            R_inv = np.transpose(R)
            if tvec.shape != (3, 1):
                tvec = tvec.reshape(3, 1)

            tvec_inv = -R_inv @ tvec
            tvec_inv_t = np.transpose(tvec_inv)
            pos = tvec_inv_t[0]
            distance = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
            rad = pos[0] / pos[2]
            angle_in_radians = math.atan(rad)
            angle_in_degrees = math.degrees(angle_in_radians)

            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, length=AXIS_LENGTH)

            tvec_inv_str = " ".join([f"{num:.2f}m" for num in tvec_inv_t.flatten()])
            cv2.putText(
                frame,
                "tvec_inv: " + tvec_inv_str,
                (0, 80),
                FONT,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "distance:" + str(round(distance, 4)) + str("m"),
                (0, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "degree: " + str(angle_in_degrees),
                (0, 150),
                FONT,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        aruco.drawDetectedMarkers(frame, marker_corners)
    else:
        # 如果没有检测到ArUco标记，则在图像上显示"No Ids"
        cv2.putText(frame, "No Ids", (0, 64), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mtx = np.array(
            [
                [711.77689507, 0, 672.53236606],
                [0, 711.78573804, 313.37884074],
                [0, 0, 1],
            ]
        )
        dist = np.array(
            [
                1.26638295e-01,
                -1.16132908e-01,
                -2.24690373e-05,
                -2.25867957e-03,
                -6.76164003e-02,
            ]
        )
        frame = estimate_pose(frame, mtx, dist)
        new_width = 1280
        new_height = 720
        frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
