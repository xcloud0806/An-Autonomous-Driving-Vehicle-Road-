import os
import numpy as np
import json
from copy import deepcopy


WGS84_F = 1.0 / 298.257223565
WGS84_A = 6378137.0


def roll_matrix(theta):
    mat = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )
    return mat


def pitch_matrix(theta):
    mat = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=np.float32,
    )
    return mat


def yaw_matrix(theta):
    mat = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return mat


def gnss_to_ecef(gnss_frame):
    roll = gnss_frame.roll
    pitch = gnss_frame.pitch
    yaw = gnss_frame.yaw

    rmat_enu_gnss = roll_matrix(roll) @ pitch_matrix(pitch) @ yaw_matrix(yaw)
    rmat_gnss_enu = rmat_enu_gnss.T
    tvec_gnss_enu = np.array([0, 0, 0])

    rmat_enu_ecef = np.array(
        [
            [-np.sin(gnss_frame.longitude), np.cos(gnss_frame.longitude), 0],
            [
                -np.sin(gnss_frame.latitude) * np.cos(gnss_frame.longitude),
                -np.sin(gnss_frame.latitude) * np.sin(gnss_frame.longitude),
                np.cos(gnss_frame.latitude),
            ],
            [
                np.cos(gnss_frame.latitude) * np.cos(gnss_frame.longitude),
                np.cos(gnss_frame.latitude) * np.sin(gnss_frame.longitude),
                np.sin(gnss_frame.latitude),
            ],
        ]
    ).T

    square_e = WGS84_F * (2 - WGS84_F)
    n = WGS84_A / np.sqrt(1 - square_e * np.power(np.sin(gnss_frame.latitude), 2))
    tvec_enu_ecef = np.array(
        [
            (n + gnss_frame.altitude)
            * np.cos(gnss_frame.latitude)
            * np.cos(gnss_frame.longitude),
            (n + gnss_frame.altitude)
            * np.cos(gnss_frame.latitude)
            * np.sin(gnss_frame.longitude),
            (n * (1 - square_e) + gnss_frame.altitude) * np.sin(gnss_frame.latitude),
        ]
    )

    ecef_pose = np.eye(4)
    ecef_pose[:3, :3] = rmat_enu_ecef @ rmat_gnss_enu
    ecef_pose[:3, 3] = rmat_enu_ecef @ tvec_gnss_enu + tvec_enu_ecef

    return ecef_pose


def calculate_euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def detect_loops(trajectory, pose_list, dis_threshold, height_threshold):
    loops = []
    trajectory_distance = np.zeros(len(trajectory), dtype=np.float32)
    for i in range(1, len(trajectory)):
        trajectory_distance[i] = trajectory_distance[
            i - 1
        ] + calculate_euclidean_distance(trajectory[i][0][:2], trajectory[i - 1][0][:2])

    for i in range(len(trajectory)):
        for j in range(i + 1, len(trajectory)):
            distance = calculate_euclidean_distance(
                trajectory[i][0][:2], trajectory[j][0][:2]
            )
            if (
                (distance < dis_threshold)
                and (
                    trajectory_distance[j] - trajectory_distance[i] > 3 * dis_threshold
                )
                and (
                    np.abs(trajectory[i][0][2] - trajectory[j][0][2]) > height_threshold
                )
            ):
                # loop start idx, loop end idx, trajectory distance
                loops.append(
                    [trajectory[i][1], trajectory[j][1] - 1, trajectory_distance[j]]
                )
                break
    if len(loops) == 0:
        loops = [[0, len(pose_list) - 1, trajectory_distance[-1]]]
    return loops


def get_segment_clip_cut_idx(pose_list, dis_threshold, height_threshold):
    trajectory_list = []
    last_xy_location = np.array([0, 0], dtype=np.float32)
    for idx, pose in enumerate(pose_list):
        xy_location = pose[:2, 3]
        if calculate_euclidean_distance(last_xy_location, xy_location) < 0.2:
            continue
        last_xy_location = xy_location
        trajectory_list.append((pose[:3, 3], idx))  # x and y in world space

    loops = detect_loops(trajectory_list, pose_list, dis_threshold, height_threshold)
    loops = np.asarray(loops)

    select_idx = np.argmin(loops[:, 1])
    cut_idx = int(loops[select_idx, 1])
    trajectory_dis = loops[select_idx, 2]
    return cut_idx, trajectory_dis


def draw_2D_trajectory(points_list, save_path="", mode="scatter"):
    import matplotlib.pyplot as plt

    figure = plt.figure()
    ax = plt.subplot()
    for points in points_list:
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        if mode == "scatter":
            ax.scatter(x, y)
            # ax.set_aspect('equal', adjustable='box')
        elif mode == "line":
            ax.plot(x, y)
    if save_path != "":
        plt.savefig(save_path)
    plt.close()


class GnssFrame:
    def __init__(self, roll, pitch, yaw, longitude, latitude, altitude):
        self.roll = self.deg_to_rad(roll)
        self.pitch = self.deg_to_rad(pitch)
        self.yaw = self.deg_to_rad(yaw)
        self.longitude = self.deg_to_rad(longitude)
        self.latitude = self.deg_to_rad(latitude)
        self.altitude = altitude

    def deg_to_rad(self, theta):
        return theta * np.pi / 180


def re_segment_clip(
    gnss_list, dis_threshold=50, height_threshold=3
):
    # gnss = json.load(open(gnss_path, "r"))
    pose_list = []
    ts_list = []  # time stamp
    cnt = 0
    for k, frame in gnss_list.items():
        if cnt % 10 != 0:
            cnt += 1
            continue
        ts_list.append(k)
        for k, v in frame.items():
            frame[k] = float(v)
        gnss_frame = GnssFrame(
            frame["roll"],
            frame["pitch"],
            frame["yaw"],
            frame["longitude"],
            frame["latitude"],
            frame["altitude"],
        )
        pose = gnss_to_ecef(gnss_frame)
        pose_list.append(pose)
        cnt += 1

    normal_pose_list = [
        np.eye(4),
    ]
    pose_0 = pose_list[0]
    pose_0_inv = np.linalg.pinv(pose_0)
    for pose in pose_list[1:]:
        normal_pose = pose_0_inv @ pose
        normal_pose_list.append(normal_pose)

    # 正向切段
    sub_clip_list = []
    sub_clip_dis_list = []

    normal_pose_list_runnig = deepcopy(normal_pose_list)
    ts_list_running = deepcopy(ts_list)

    while len(normal_pose_list_runnig) > 0:
        cut_idx, trajectory_dis = get_segment_clip_cut_idx(
            normal_pose_list_runnig, dis_threshold, height_threshold
        )
        sub_clip_list.append([ts_list_running[0], ts_list_running[cut_idx]])
        sub_clip_dis_list.append(trajectory_dis)
        normal_pose_list_runnig = normal_pose_list_runnig[cut_idx + 1 :]
        ts_list_running = ts_list_running[cut_idx + 1 :]

    if len(sub_clip_list) > 1:
        # 逆向切段
        sub_clip_list_reverse = []
        sub_clip_dis_list_reverse = []

        normal_pose_list_runnig = deepcopy(normal_pose_list[::-1])
        ts_list_running = deepcopy(ts_list[::-1])

        while len(normal_pose_list_runnig) > 0:
            cut_idx, trajectory_dis = get_segment_clip_cut_idx(
                normal_pose_list_runnig, dis_threshold, height_threshold
            )
            sub_clip_list_reverse.append([ts_list_running[0], ts_list_running[cut_idx]])
            sub_clip_dis_list_reverse.append(trajectory_dis)
            normal_pose_list_runnig = normal_pose_list_runnig[cut_idx + 1 :]
            ts_list_running = ts_list_running[cut_idx + 1 :]

        max_dis = np.array(sub_clip_dis_list).max()
        max_dis_reverse = np.array(sub_clip_dis_list_reverse).max()

        if max_dis < max_dis_reverse:
            sub_clip_dis_list = sub_clip_dis_list_reverse[::-1]
            sub_clip_list = [
                trajectory_range[::-1]
                for trajectory_range in sub_clip_list_reverse[::-1]
            ]

    if False:
        points_list = []
        for range in sub_clip_list:
            points = []
            for i, ts in enumerate(ts_list):
                if ts >= range[0] and ts <= range[1]:
                    points.append(normal_pose_list[i][:2, 3])
            points = np.asarray(points)
            points_list.append(points)

        draw_2D_trajectory(points_list, save_path="/data_autodrive/users/brli/dev_raw_data/re_segment_clips/traj.jpg")

    return sub_clip_list, sub_clip_dis_list


if __name__ == "__main__":
    # path = '/data_autodrive/users/jhyan2/MetaData/problem/single_clip/V2-14/sihao_y7862_20240408-08-49-28_seg0/gnss.json'
    path = "/data_cold2/origin_data/sihao_y7862/custom_seg/city_scene_test/20240408/sihao_y7862_20240408-09-37-22_seg0/gnss.json"
    print(re_segment_clip(path, vis=True, save_path="./test/trajectory.jpg"))
