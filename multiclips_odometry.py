import argparse
import os
import sys
import json
import numpy as np
import datetime
import random
import string
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "lidar_odometry/lib"))
import odometry

import pymongo
from pymongo.errors import PyMongoError
from utils import (
    PymongoHelper,
    match_database,
    construct_info,
    get_database_sample,
    get_database_distance,
    get_increment_annotation_data,
)
from utils import cal_trajectory_angle

from multiprocessing.pool import Pool
import multiprocessing as mp

def insert_clip(db_handle:PymongoHelper, clip_path):
    odometry.TrajectoryGrid(clip_path) #计算网格
    info = construct_info(clip_path)
    if not db_handle.insert(info, "maps"):
        db_handle.update(info, 'maps')
    return


def insert_folder_clip(db_handle:PymongoHelper, root_path):
    if not os.path.exists(root_path):
        return
    folders = os.listdir(root_path)  # 获取子文件夹列表
    for sub in folders:  # 遍历子文件夹列表
        path = os.path.join(root_path, sub)  # 子文件夹路径
        insert_clip(db_handle, path)
    return

def cal_lidar_odometry(seg_folder):
    meta_json = os.path.join(seg_folder, "meta.json")
    with open(meta_json, "r", encoding='utf-8') as fp:
        meta = json.load(fp)
        meta_json_str = json.dumps(meta)

    gnss_json = os.path.join(seg_folder, "gnss.json")
    with open(gnss_json, "r", encoding='utf-8') as fp:
        gnss = json.load(fp)
        gnss_json_str = json.dumps(gnss)

    vehicle_json = os.path.join(seg_folder, "vehicle.json")
    with open(vehicle_json, "r", encoding='utf-8') as fp:
        vehicle = json.load(fp)
        vehicle_json_str = json.dumps(vehicle)

    meta_ss = odometry.LidarWheelConstruct(seg_folder, meta_json_str, gnss_json_str, vehicle_json_str)
    
    dst_json_path = os.path.join(seg_folder, "meta.json")
    if len(meta_ss) == 0:
        logger.error("Odeometry Failed...")
    else:
        with open(dst_json_path, "w") as fp:
            fp.write(meta_ss)

def computer_clip_odometry(seg_folder):
    err0 = odometry.ComputerAccuracy(seg_folder, "meta.json", True, False)
    cal_lidar_odometry(seg_folder)
    err1 = odometry.ComputerAccuracy(seg_folder, "meta.json", True, False)
    logger.info(f"{seg_folder}, {err0}, {err1}")

def computer_multi_odometry(root_path):
    if not os.path.exists(root_path):
        return
    folders = os.listdir(root_path) 
    for sub in folders:  
        seg_folder = os.path.join(root_path, sub)
        err = odometry.ComputerAccuracy(seg_folder, "meta.json", True, False)
        if (err > 5.0):
            # print(seg_folder)
            computer_clip_odometry(seg_folder)

def detect_multi_meta(folder):
    subs = os.listdir(folder) 
    for sub in subs:  
        seg_folder = os.path.join(folder, sub)  
        multi_meta_path = os.path.join(seg_folder, "multi_meta.json")
        if not os.path.exists(multi_meta_path):
            print(seg_folder)

class DataAssociation():
    def __init__(self):
        self.mode = 0
        self.history_folders = []
    
    def generate_password(self):
        password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        return password
    
    def get_current_time(self):
        current_time = datetime.datetime.now()
        format_date = current_time.strftime("%Y%m%d-%H-%M-%S-")
        return format_date
    
    def get_crossing_key(self, subs):
        crossing_keys = []
        for sub in subs:
            sub_split = sub.split("_")
            sub_split.pop()
            sub_merge = '_'.join(sub_split)
            if sub_merge not in crossing_keys:
                crossing_keys.append(sub_merge)
        return crossing_keys
    
    def get_crossing_key_list(self, key, subs, seg_folders):
        crossing_lists = []
        for sub in subs:
            if key in sub:
                crossing_lists.append(os.path.join(seg_folders, sub))
        return crossing_lists
    
    def remove_empty_files(self, path_lists):
        exist_path_lists = []
        for file in path_lists:
            if os.path.exists(file):
                exist_path_lists.append(file)
        return exist_path_lists
    
    def generate_multi_info(self, main_clip_lists, clips_lists, incre_mark):
        match_info = {}
        info_key = self.get_current_time() + self.generate_password()
        match_info[info_key] = {}
        match_info[info_key]["main_clip_path"] = main_clip_lists
        match_info[info_key]["clips_path"] = clips_lists

        match_info[info_key]["increment_enable"] = incre_mark
        match_info[info_key]["increment_status"] = False
        match_info[info_key]["multi_odometry_status"] = False

        match_info[info_key]["mode"] = self.mode
        match_info[info_key]["version"] = "2.0"
        match_info[info_key]["reconstruct_path"] = ""
        return match_info
    
    def generate_night_dict(self, args):
        match_infos = {}
        root_folder = args.frames
        subs = os.listdir(root_folder)
        for sub in subs:
            seg_folder = os.path.join(root_folder, sub)
            db_helper = PymongoHelper("172.30.35.11", 27017, "ar_maps", ["maps"])
            odometry.TrajectoryGrid(seg_folder) 
            sample_lists = get_database_distance(db_helper, seg_folder, 500, 0.5, 5)
            del(db_helper)
            sample_lists = list(set(sample_lists))
            
            main_lists = []
            main_lists.append(seg_folder)
            minor_lists = []
            for sample in sample_lists:
                if sub not in sample:
                    minor_lists.append(sample)
            
            minor_lists = self.remove_empty_files(minor_lists)
            if (len(minor_lists) == 0):
                continue

            match_info = self.generate_multi_info(main_lists, minor_lists, False)
            match_infos[sub] = match_info
        return match_infos
    
    def generate_crossing_dict(self, seg_folders):
        match_infos = {}
        subs = os.listdir(seg_folders)
        crossing_keys = self.get_crossing_key(subs)
        for key in crossing_keys:
            crossing_lists = self.get_crossing_key_list(key, subs, seg_folders)
            crossing_match = []
            db_helper = PymongoHelper("172.30.35.11", 27017, "ar_maps", ["maps"])
            for crossing_seg in crossing_lists:
                odometry.TrajectoryGrid(crossing_seg) 
                sample_lists = get_database_distance(db_helper, crossing_seg, 500, 0.5, 2)
                if len(sample_lists) > 0:
                    crossing_match.extend(sample_lists)
            del(db_helper)
            crossing_match = list(set(crossing_match))

            clips_lists = []
            for match in crossing_match:
                if match not in crossing_lists:
                    clips_lists.append(match)
            clips_lists = self.remove_empty_files(clips_lists)

            match_info = self.generate_multi_info(crossing_lists, clips_lists, False)
            match_infos[key] = match_info
        return match_infos
    
    def calculate_angle(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_angle = dot_product / (norm_v1 * norm_v2)
        angle = np.arccos(cos_angle) * (180 / np.pi)
        return angle
    
    def select_direct_traject(self, samples):
        samples_direct = []
        for sample in samples:
            unit_direct = cal_trajectory_angle(sample)
            samples_direct.append(unit_direct)
        
        cluster_samples = []
        for i, source in enumerate(samples_direct):
            cluster = []
            cluster.append(i)
            for j, target in enumerate(samples_direct):
                if i == j:
                    continue
                angle = self.calculate_angle(samples_direct[i], samples_direct[j])
                if (angle < 90):
                    cluster.append(j)
            cluster_samples.append(cluster)
        
        best_samples = []
        best_num = 0
        for clusters in cluster_samples:
            if len(clusters) > best_num:
                best_samples = []
                for index in clusters:
                    best_samples.append(samples[index])
                best_num = len(clusters)
        return best_samples
    
    def select_direct_match_clips(self, samples, match_clips):
        unit_sample = cal_trajectory_angle(samples[0])
        match_direct_clips = []
        for match_clip in match_clips:
            unit_match_clip = cal_trajectory_angle(match_clip)
            angle = self.calculate_angle(unit_sample, unit_match_clip)
            if (angle < 90):
                match_direct_clips.append(match_clip)
        return match_direct_clips

    # 基于单段匹配数据库的段; 主段:同属一个根目录,其他属于辅段
    def generate_updown_dict(self, args):
        match_infos = {}
        seg_folders = args.frames
        subs = os.listdir(seg_folders)
        for sub in subs:
            seg_folder = os.path.join(seg_folders, sub)  
            db_helper = PymongoHelper("172.30.35.11", 27017, "ar_maps", ["maps"])
            odometry.TrajectoryGrid(seg_folder) 
            sample_lists = get_database_distance(db_helper, seg_folder, args.match_distance, args.match_coincide) 
            del(db_helper)
            
            sample_filters = []
            for sample in sample_lists:
                if seg_folders in sample:
                    if sample not in self.history_folders:
                        sample_filters.append(sample)
            
            sample_filters = self.select_direct_traject(sample_filters)
            sample_filters = list(set(sample_filters))
            if len(sample_filters) < 2:
                continue
            for filter in sample_filters:
                self.history_folders.append(filter)

            crossing_match = []
            db_helper = PymongoHelper("172.30.35.11", 27017, "ar_maps", ["maps"])
            for crossing_seg in sample_filters:
                odometry.TrajectoryGrid(crossing_seg) 
                sample_lists = get_database_distance(db_helper, crossing_seg, 500, 0.5, 3)
                if len(sample_lists) > 0:
                    crossing_match.extend(sample_lists)
            del(db_helper)
            crossing_match = list(set(crossing_match))

            clips_lists = []
            for match in crossing_match:
                if match not in sample_filters:
                    clips_lists.append(match)
            clips_lists = self.remove_empty_files(clips_lists)
            clips_lists = self.select_direct_match_clips(sample_filters, clips_lists)

            match_info = self.generate_multi_info(sample_filters, clips_lists, False)
            match_infos[sub] = match_info
        return match_infos

    def generate_increment_dict(self, args):
        match_infos = {}
        root_folder = args.frames
        subs = os.listdir(root_folder)
        for sub in subs:
            seg_folder = os.path.join(root_folder, sub)
            db_helper = PymongoHelper("172.30.35.11", 27017, "ar_maps", ["maps"])
            odometry.TrajectoryGrid(seg_folder) 
            sample_lists = get_increment_annotation_data(db_helper, seg_folder, 500, 0.5, 5)
            del(db_helper)

            if (len(sample_lists) == 0):
                continue
            sample_lists = list(set(sample_lists))
            
            main_lists = []
            main_lists.append(seg_folder)
            minor_lists = []
            for sample in sample_lists:
                if sub not in sample:
                    minor_lists.append(sample)

            minor_lists = self.remove_empty_files(minor_lists)
            if (len(minor_lists) == 0):
                continue

            match_info = self.generate_multi_info(main_lists, minor_lists, True)
            match_infos[sub] = match_info
        return match_infos

def create_multi_info_path(args, v):
    keys = [key for key in v]
    root_path = os.path.join(args.save_folder, keys[0])
    if not os.path.exists(root_path):
        os.makedirs(root_path, mode=0o777, exist_ok=True)
    return root_path

def multi_night_odometry(args):
    logger.info("Night is calculating")
    computer_multi_odometry(args.frames)
    computer_multi_odometry(args.database)
    db_handle = PymongoHelper("172.30.35.11", 27017, "ar_maps", ["maps"])
    insert_folder_clip(db_handle, args.database)
    del(db_handle)

    data_manager = DataAssociation()
    data_manager.mode = 0
    night_infos = data_manager.generate_night_dict(args)
    for k, v in night_infos.items():
        root_path = create_multi_info_path(args, v)
        json.dump(v, open(os.path.join(root_path, 'multi_info.json'), 'w'))

    if not os.path.exists(args.save_folder):
        return
    for folder in os.listdir(args.save_folder):
        sub_folder = os.path.join(args.save_folder, folder)
        if not os.path.isdir(sub_folder):
            continue
        status = odometry.MultiLidarConstruct(sub_folder)


def multi_incre_odometry(args):
    logger.info("Night is calculating")
    computer_multi_odometry(args.frames)
    computer_multi_odometry(args.database)
    db_handle = PymongoHelper("172.30.35.11", 27017, "ar_maps", ["maps"])
    insert_folder_clip(db_handle, args.database)
    del(db_handle)

    data_manager = DataAssociation()
    data_manager.mode = 3
    incre_infos = data_manager.generate_increment_dict(args)
    for k, v in incre_infos.items():
        root_path = create_multi_info_path(args, v)
        json.dump(v, open(os.path.join(root_path, 'multi_info.json'), 'w'))

    if not os.path.exists(args.save_folder):
        return
    for folder in os.listdir(args.save_folder):
        sub_folder = os.path.join(args.save_folder, folder)
        if not os.path.isdir(sub_folder):
            continue
        status = odometry.MultiIncrementConstruct(sub_folder)

def cal_Multi_UpDown_Construct(sub_folder):
    status = odometry.MultiUpDownConstruct(sub_folder)         

def multi_updown_odometry(args):
    mp.set_start_method('spawn')
    logger.info("Night Up and down ramps is calculating")
    computer_multi_odometry(args.frames)
    computer_multi_odometry(args.database)
    db_handle = PymongoHelper("172.30.35.11", 27017, "ar_maps", ["maps"])
    insert_folder_clip(db_handle, args.frames)
    insert_folder_clip(db_handle, args.database)
    del(db_handle)

    data_manager = DataAssociation()
    data_manager.mode = 1
    match_infos = data_manager.generate_updown_dict(args)
    for k, v in match_infos.items():
        root_path = create_multi_info_path(args, v)
        json.dump(v, open(os.path.join(root_path, 'multi_info.json'), 'w'))

    if not os.path.exists(args.save_folder):
        return

    for folder in os.listdir(args.save_folder):
        sub_folder = os.path.join(args.save_folder, folder)
        if not os.path.isdir(sub_folder):
            continue
        cal_Multi_UpDown_Construct(sub_folder)


def multi_cross_odometry(args):
    logger.info("Cross roads is calculating")
    computer_multi_odometry(args.frames)
    computer_multi_odometry(args.database)
    db_handle = PymongoHelper("172.30.35.11", 27017, "ar_maps", ["maps"])
    insert_folder_clip(db_handle, args.database)
    del(db_handle)

    data_manager = DataAssociation()
    data_manager.mode = 2
    match_infos = data_manager.generate_crossing_dict(args.frames)
    for k, v in match_infos.items():
        root_path = create_multi_info_path(args, v)
        json.dump(v, open(os.path.join(root_path, 'multi_info.json'), 'w'))

    if not os.path.exists(args.save_folder):
        return
    for folder in os.listdir(args.save_folder):
        sub_folder = os.path.join(args.save_folder, folder)
        if not os.path.isdir(sub_folder):
            continue
        status = odometry.MultiCrossConstruct(sub_folder)

class dft_args:
    mode = 0
    frames = ""
    save_folder = ""
    database = ""
    match_distance = 1000
    match_coincide = 0.1

if __name__ == '__main__':
    config_file = "./utils/utils_template.cfg"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        logger.error(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "multiclips_odometry.log"))

    multi_enable = run_config["multi_seg"]["enable"]
    if not multi_enable:
        logger.error("multi_enable is false")
        sys.exit(1)

    segment_path = run_config["multi_seg"]["segment_path"]
    if not os.path.exists(segment_path):
        logger.error("No Segment Path")
        sys.exit(1)

    folders = os.listdir(segment_path)
    if len(folders) == 0:
        logger.error("No Segment Folder")
        sys.exit(1)

    args = dft_args()
    args.mode = run_config["multi_seg"]["mode"]
    args.frames = run_config["multi_seg"]["segment_path"]
    args.save_folder = run_config["multi_seg"]["multi_info_path"]
    args.database = run_config["multi_seg"]["database_folder"]
    args.match_distance = run_config["multi_seg"]["match_distance"]
    args.match_coincide = run_config["multi_seg"]["match_coincide"]
    
    if (args.mode == 1):
        multi_updown_odometry(args)    
    elif (args.mode == 2):
        multi_cross_odometry(args)
    else:
        logger.error(f"Wrong Mode in {args.frames}")
        sys.exit(1)
    
    logger.info("Odometry completed")
    multiinfo_path = args.save_folder
    colls = os.listdir(multiinfo_path)
    valid_coll = []
    for coll in colls:
        coll_path = os.path.join(multiinfo_path, coll)
        if not os.path.isdir(coll_path):
            continue
        info_file = os.path.join(coll_path, "multi_info.json")
        if not os.path.exists(info_file):
            continue
        with open(info_file, "r") as fp:
            info = json.load(fp)
            if info[coll]["multi_odometry_status"]:
                valid_coll.append(coll)
    logger.info(f"Valid Colls'num ({len(valid_coll)}) -> [{valid_coll}]")
