import os
import csv
from os.path import join
import json
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from copy import deepcopy
from loguru import logger

WGS84_F = 1.0 / 298.257223565
WGS84_A = 6378137.0

min_trajectory_distance = 150.
max_match_distance = 20.
max_match_degree = 60.

def roll_matrix(theta):
    mat = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ], dtype=np.float32)
    return mat

def pitch_matrix(theta):
    mat = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ], dtype=np.float32)
    return mat

def yaw_matrix(theta):
    mat = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    return mat


def gnss_to_ecef(gnss_frame):
    roll = gnss_frame.roll
    pitch = gnss_frame.pitch
    yaw = gnss_frame.yaw

    rmat_enu_gnss = roll_matrix(roll) @ pitch_matrix(pitch) @ yaw_matrix(yaw)
    rmat_gnss_enu = rmat_enu_gnss.T
    tvec_gnss_enu = np.array([0, 0, 0])

    rmat_enu_ecef = np.array([
        [-np.sin(gnss_frame.longitude), np.cos(gnss_frame.longitude), 0],
        [-np.sin(gnss_frame.latitude) * np.cos(gnss_frame.longitude), -np.sin(gnss_frame.latitude) * np.sin(gnss_frame.longitude), np.cos(gnss_frame.latitude)],
        [np.cos(gnss_frame.latitude) * np.cos(gnss_frame.longitude), np.cos(gnss_frame.latitude) * np.sin(gnss_frame.longitude), np.sin(gnss_frame.latitude)]
    ]).T

    square_e = WGS84_F * (2 - WGS84_F)
    n = WGS84_A / np.sqrt(1 - square_e * np.power(np.sin(gnss_frame.latitude), 2))
    tvec_enu_ecef = np.array([
        (n + gnss_frame.altitude) * np.cos(gnss_frame.latitude) * np.cos(gnss_frame.longitude),
        (n + gnss_frame.altitude) * np.cos(gnss_frame.latitude) * np.sin(gnss_frame.longitude),
        (n * (1 - square_e) + gnss_frame.altitude) * np.sin(gnss_frame.latitude)
    ])

    ecef_pose = np.eye(4)
    ecef_pose[:3, :3] = rmat_enu_ecef @ rmat_gnss_enu
    ecef_pose[:3, 3] = rmat_enu_ecef @ tvec_gnss_enu + tvec_enu_ecef

    return ecef_pose


def calculate_euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def draw_2D_trajectory(points_list, save_path='', mode='scatter'):
    import matplotlib.pyplot as plt
    figure = plt.figure()
    ax = plt.subplot()
    for points in points_list:
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        if mode == 'scatter':
            ax.scatter(x, y)
            # ax.set_aspect('equal', adjustable='box')
        elif mode == 'line':
            ax.plot(x, y)
    if save_path != '':
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

def parse_gnss_csv(gnss_csv):
    gnss_info = dict()
    if not os.path.exists(gnss_csv):
        return FileNotFoundError
    with open(gnss_csv, encoding='utf-8') as fp:
        reader = csv.reader(fp)
        header = next(reader)
        key_idx = {}
        for idx, key in enumerate(header):
            key_idx[key] = idx
        time_key = "utc_time"
        time_id = key_idx[time_key]
        for row in reader:
            item = {}                
            k = row[time_id]
            if k == 'na' or k == time_key:
                continue
            for _i, v in enumerate(row):
                if _i > (len(header)-1):
                    continue
                _k = header[_i]
                item[_k] = v
            gnss_info[k] = item
    return gnss_info


def process_gnss_json(gnss):
    pose_list = []
    ts_list = [] # time stamp
    cnt = 0
    for ts, frame in gnss.items():
        if cnt % 10 != 0:
            cnt += 1
            continue
        for k, v in frame.items():
            frame[k] = float(v)
        gnss_frame = GnssFrame(frame['roll'], frame['pitch'], frame['yaw'], frame['longitude'], frame['latitude'], frame['altitude'])
        pose = gnss_to_ecef(gnss_frame)
        if (cnt != 0) and (calculate_euclidean_distance(pose[:3, 3], pose_list[-1][:3, 3]) < 0.5):
            continue
        ts_list.append(ts)
        pose_list.append(pose)
        cnt += 1
    return np.array(pose_list, dtype=np.float32), ts_list

def longest_consecutive_sequence(nums, poses):
    start_id = nums[0]
    end_id = nums[0]
    last_id = nums[0]
    sub_seq_list = []
    for i in nums[1:]:
        if i == last_id + 1:
            end_id = i
            last_id = i
        else:
            sub_seq_list.append((start_id, end_id))
            start_id = i
            end_id = i
            last_id = i
    sub_seq_list.append((start_id, end_id))
    
    max_length_seq = 0
    start_id = -1
    end_id = -1
    for sub_seq in sub_seq_list:
        length = 0.
        for i in range(poses.shape[0])[sub_seq[0] + 1:sub_seq[1] + 1]:
            length += calculate_euclidean_distance(poses[i,:3,3], poses[i-1,:3,3])        
        if length > max_length_seq:
            max_length_seq = length
            start_id = sub_seq[0]
            end_id = sub_seq[1]

    return start_id, end_id, max_length_seq

def match_night_to_day(poses_A, poses_B):
    dist = cdist(poses_A[:, :3, 3], poses_B[:, :3, 3]) # Mx3 and NX3 -> M x N
    oris_A = poses_A[:, :2, 1] # 取GNSS y轴计算帧之间的角度，用于判断是否为对向车道轨迹
    oris_B = poses_B[:, :2, 1]
    dot_product = oris_A @ oris_B.T # Mx2 and 2xN -> M X N
    norm = np.matmul(np.linalg.norm(oris_A, axis=1, keepdims=True),
                        np.linalg.norm(oris_B, axis=1, keepdims=True).T) # Mx1 and 1xN -> MxN
    cos_angle = dot_product / norm
    cos_angle = np.clip(cos_angle, -1, 1)
    theta = np.arccos(cos_angle)
    theta = theta / np.pi * 180

    match_id = np.zeros(poses_A.shape[0], dtype=np.int32) - 1
    flag = (dist <= max_match_distance) & (theta <= max_match_degree)
    for i in range(dist.shape[0]): # M
        idx = np.where(flag[i])[0]
        if idx.shape[0] == 0:
            continue
        match_id[i] = idx[0]

    match_idxs = np.where(match_id != -1)[0]
    if match_idxs.shape[0] < 2:
        return None, None, None

    start_id, end_id, trajectory_dis = longest_consecutive_sequence(match_idxs, poses_A)
    if start_id == -1 or end_id == -1:
        tqdm.write('no match sub seq')
        return None, None, None
    if trajectory_dis < min_trajectory_distance:
        tqdm.write('sub seq too short')
        return None, None, None
    
    return start_id, end_id, trajectory_dis


def get_day_seg_gnss_info(day_dir, night_center_3d):
    day_seg_gnss_info = dict()
    for seg_name in tqdm(os.listdir(day_dir), ncols=150):
        gnss_path = join(day_dir, seg_name, 'gnss.json')
        gnss = json.load(open(gnss_path, 'r'))
        poses, ts_list = process_gnss_json(gnss)
        day_center_3d = np.mean(poses[:, :3, 3], axis=0)
        # if calculate_euclidean_distance(night_center_3d, day_center_3d) > 10000:
            # continue
        day_seg_gnss_info[seg_name] = {
            'poses': poses,
            'ts_list': ts_list,
        }
    return day_seg_gnss_info


def normalize_pose(poses, normal_pose):
    for i in range(poses.shape[0]):
        poses[i] = normal_pose @ poses[i]
    return poses

def segment_night_data(day_seg_dir, night_gnss_path, work_temp_dir):
    print('Load night gnss data')
    night_gnss = parse_gnss_csv(night_gnss_path)
    night_poses, night_ts_list = process_gnss_json(night_gnss)
    night_center_3d = np.mean(night_poses[:, :3, 3], axis=0)
    print('Load day seg gnss data')
    day_seg_gnss_info = get_day_seg_gnss_info(day_seg_dir, night_center_3d)

    segment_results_list = []

    print('Match night to day seg data')
    for seg_name, day_gnss_info in tqdm(day_seg_gnss_info.items(), ncols=150):
        tqdm.write(f'Match night to seg: {seg_name}')
        day_seg_path = os.path.join(day_seg_dir, seg_name)
        day_seg_poses = day_gnss_info['poses']
        start_id, end_id, trajectory_dis = match_night_to_day(night_poses, day_seg_poses)
        if start_id is None:
            continue
        
        # 结果
        segment_results_list.append([night_ts_list[start_id], night_ts_list[end_id], trajectory_dis, day_seg_path])

        # 可视化代码
        night_seg_poses = deepcopy(night_poses[start_id:end_id + 1])
        normal_pose = deepcopy(np.linalg.inv(day_seg_poses[0]))
        day_seg_poses = normalize_pose(day_seg_poses, normal_pose)
        night_seg_poses = normalize_pose(night_seg_poses, normal_pose)
        save_path = f'{work_temp_dir}/segment_night_day/trajectory_{seg_name}.jpg'
        os.makedirs(os.path.dirname(save_path), mode=0o777, exist_ok=True)
        points_list = [day_seg_poses[:, :2, 3], night_seg_poses[:, :2, 3]]
        draw_2D_trajectory(points_list, save_path=save_path)

    print(night_ts_list[0], night_ts_list[-1])

    return segment_results_list

def call_gen_night_clips(config_file, related_day):
    work_tmp = os.path.dirname(config_file)
    with open(config_file, 'r') as fp:
        config = json.load(fp)
    seg_config = config['preprocess']
    frames_path = seg_config["frames_path"]
    segment_path = seg_config["segment_path"]
    
    clips = os.listdir(frames_path)
    clips.sort()
    logger.info(f"Night data in {frames_path} total have {len(clips)} clips")
    clip_cut_segs = {}
    for clip in clips:
        if not clip.startswith("202"):
            logger.warning(f"{clip} is not a valid night clip")
            continue
        clip_frame = os.path.join(frames_path, clip)
        gnss_csv_file = os.path.join(clip_frame, 'gnss.csv')
        if not os.path.exists(gnss_csv_file):
            logger.warning(f"{clip} is not a valid night clip as gnss.csv not exists")
            continue
        if isinstance(related_day, dict):
            if clip not in related_day:
                logger.warning(f"{clip} is not a valid night clip as related day not exists")
                continue
            car, subfix, related_day_subfix = related_day[clip]
            related_day_seg_root = segment_path.replace(subfix, related_day_subfix)
            if 'night' in segment_path:
                related_day_seg_root = related_day_seg_root.replace('night', 'day')
        else:
            related_day_seg_root = related_day
        if not os.path.exists(related_day_seg_root):
            logger.error(f"{related_day_seg_root} not exists")
            continue
        cut_segs = segment_night_data(related_day_seg_root, gnss_csv_file, work_tmp)
        clip_cut_segs[clip] = cut_segs
    return clip_cut_segs
