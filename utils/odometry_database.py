import os
import numpy as np
import json

import math
import re
from xml.dom.minidom import Element
import pymongo
from pymongo.errors import PyMongoError
import traceback as tb
from datetime import datetime

import sys
# import odometry

MONGO_USER = "brli"
MONGO_PASS = "lerinq1w2E#R$"

RECORD_KEY="record"
MAIN_KEY="seg_uid"
INTERVAL_KEY="time_interval"
DISTANCE_KEY="distance"
CAMERAS_KEY="cameras"
DATE_KEY="date"

sunrise = {
    "1_3":{ "start": "0700", "finish": "1822"},
    "4_6":{ "start": "0546", "finish": "1901"},
    "7_9":{ "start": "0604", "finish": "1858"}, 
    "10_12":{ "start": "0642", "finish": "1742"}
}

WGS84_F=1.0 / 298.257223565
WGS84_A=6378137.0

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


def cal_trajectory_angle(seg_folder):
    gnss_path = os.path.join(seg_folder, "gnss.json")
    gnss = json.load(open(gnss_path, 'r'))
    initial = False
    direct = np.array([0 , 0, 0])
    for i, element in enumerate(sorted(gnss.keys())):
        if i % 20 != 0:
            continue
        frame = gnss[element]
        
        for k, v in frame.items():
            frame[k] = float(v)
        if (frame['gps_status']==0 or frame['longitude']==0 or frame['latitude']==0):
            continue
        gnss_frame = GnssFrame(frame['roll'], frame['pitch'], frame['yaw'], frame['longitude'], frame['latitude'], frame['altitude'])
        pose = gnss_to_ecef(gnss_frame)

        if (not initial):
            initial = True
            start_point = np.array([pose[0,3], pose[1,3], 0])
        else:
            finish_point = np.array([pose[0,3], pose[1,3], 0])
            finish_start = finish_point - start_point
            direct = direct + finish_start
            start_point = finish_point

    direct_norm = np.linalg.norm(direct)
    unit_direct = direct / direct_norm
    return unit_direct
        

def night_time(name):
    night_state = True
    time_list = name.split("_")
    for time in time_list:
        time_split = time.split("-")
        if len(time_split) == 4:
            year_month = time_split[0]
            day_time = time_split[1] + time_split[2]
   
    month = int(year_month[4:6])
    sun_key = "1_3"
    if month in [4,5,6]:
        sun_key = "4_6"
    if month in [7,8,9]:
        sun_key = "7_9"
    if month in [10,11,12]:
        sun_key = "10_12"

    hour_minute = int(day_time)
    if (hour_minute > int(sunrise[sun_key]["start"])) and (hour_minute < int(sunrise[sun_key]["finish"])):
        night_state = False
    return night_state

class PymongoHelper:
    def __init__(self, ip:str, port:int, db, collections) -> None:
        self.status = False
        #self.translator = Translator(to_lang="en", from_lang="zh")
        try:
            self.client = pymongo.MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASS}@{ip}:{port}/")        
            self.db = self.client[db]
            #self.db.authenticate(MONGO_USER, MONGO_PASS)
            self.collects = {}
            for collection in collections:
                self.collects[collection] = self.db[collection]
            self.status=True
        except PyMongoError as exc:
            tb.print_exc()
            if exc.timeout:
                print(f"[MongoDB] block timed out: {exc!r}")
            else:
                print(f"[MongoDB] failed with non-timeout error: {exc!r}")
            pass
    
    def insert(self, info, coll):
        collect = self.collects[coll]
        segid = info[MAIN_KEY]
        try:
            condition = {MAIN_KEY: segid}
            ret = collect.find_one(condition)
            if ret is None:
                ret = collect.insert_one(info)
                print(f"[MongoDB] Successfully inserted clip data from {segid}")
                return True
            else:
                print(f"[MongoDB] Some segs already exist in the collection {coll}") 
                return False
        except Exception as e:
            print(f"[MongoDB] Failed to insert clip data from {segid} due to {e}")
            tb.print_exc()
            return False
    
    def insert_many(self, infos, coll):
        collect = self.collects[coll]
        try:
            find_flag = False
            for info in infos:
                segid = info[MAIN_KEY]
                condition = {MAIN_KEY: segid}
                ret = collect.find_one(condition)
                if ret is not None:
                    find_flag = True
            if find_flag:                
                print(f"[MongoDB] Some segs already exist in the collection {coll}")                
                return False        
            ret = collect.insert_many(infos)
            print(f"[MongoDB] Successfully inserted clip data from {len(infos)} segs")
            return True
        except Exception as e:
            print(f"[MongoDB] Failed to insert clip data due to {e}")
            tb.print_exc()
            return False
    
    def update(self, info, coll):
        collect =  self.collects[coll]
        segid = info[MAIN_KEY]
        try:
            condition = {MAIN_KEY: segid}
            new_values = {"$set": info}
            ret = collect.find_one(condition)
            if ret is None: 
                print(f"[MongoDB] {segid} not exists.")      
                return False
            collect.update_one(condition, new_values)
            print(f"[MongoDB] Successfully updated clip data from {segid}")
            return True
        except Exception as e:
            print(f"[MongoDB] Failed to update clip data from {segid} due to {e}")
            tb.print_exc()
            return False
        
    def query(self, segid, coll):
        collect = self.collects[coll]
        try:
            condition = {MAIN_KEY: segid}
            ret = collect.find_one(condition)
            if ret is None: 
                print(f"[MongoDB] {segid} not exists in {coll}.")      
                return {}
            return ret
        except Exception as e:
            print(f"[MongoDB] Failed to query clip data from {segid} due to {e}")
            tb.print_exc()
            return None
        
    def delete(self, segid, coll):
        collect = self.collects[coll]
        try:
            condition = {MAIN_KEY: segid}
            ret = collect.find_one(condition)
            if ret is None: 
                print(f"[MongoDB] {segid} not exists in {coll}.")      
                return False
            collect.delete_one(condition)
            print(f"[MongoDB] Successfully deleted clip data from {segid}")
            return True
        except Exception as e:
            print(f"[MongoDB] Failed to delete clip data from {segid} due to {e}")
            tb.print_exc()
            return False
    
    def __del__(self):
        self.status = False
        self.client.close()

def construct_info(clip_path):
    info = {}
    meta_file = os.path.join(clip_path, "meta.json")

    try:
        with open(meta_file, 'r') as f:
            clip_meta = json.load(f)
        info[MAIN_KEY] = clip_meta['seg_uid']
        info[INTERVAL_KEY] = int(float(clip_meta['time_interval']) / 1000)
        info[DISTANCE_KEY] = int(clip_meta['distance'])
        info[CAMERAS_KEY] = clip_meta['cameras']
        info[DATE_KEY] = clip_meta['date']
        
        # if RECORD_KEY in clip_meta and clip_meta[RECORD_KEY] is not None:
        #     record = clip_meta[RECORD_KEY]
        #     # info['weather'] = translate(record['weather'])
        #     info['windpower'] = record['windpower']            
        #     info['plate_number'] = record['plate_number']
        # else:
        #     info['weather'] = ""
        #     info['windpower'] = ""          
        #     info['plate_number'] = ""

        info['weather'] = ""
        info['windpower'] = ""          
        info['plate_number'] = ""

        info['car'] = clip_meta['car']

        info['clip_path'] = clip_path
        if "trajectory_center" in clip_meta and clip_meta['trajectory_center'] is not None:
            info['centerX'] = clip_meta['trajectory_center'][0]
            info['centerY'] =  clip_meta['trajectory_center'][1]
        else:
            info['centerX'] = 0
            info['centerY'] = 0

        if "trajectory_grids" in clip_meta and clip_meta['trajectory_grids'] is not None:
            info['segGrid'] = clip_meta['trajectory_grids']
    
    except  Exception as e:
        print(f"[MongoDB] Failed to construct info due to {e}")
        tb.print_exc()
    return info
def insert_clips(db_handle:PymongoHelper, clip_paths):
    infos = []
    for clip in clip_paths:
        info = construct_info(clip)
        infos.append(info)
    return db_handle.insert_many(infos, "maps")
        


def update_clip(db_handle:PymongoHelper, clip_id, info:dict):
    info.update({
        MAIN_KEY: clip_id
    })
    return db_handle.update(info, 'maps')


def query_clip(db_handle:PymongoHelper, clip_id):
    return db_handle.query(clip_id, "maps")


def delete_clip(db_handle:PymongoHelper, clip_id):
    return db_handle.delete(clip_id, "maps")

def delete_all_clip(db_handle:PymongoHelper):
    collection = db_handle.collects['maps']
    collection.delete_many({"centerX": {"$gte": 0}})
    collection.delete_many({"centerX": {"$lte": 0}})

def delete_night_data(db_handle:PymongoHelper):
    collection = db_handle.collects['maps']
    for document in collection.find():
        remove_status = False
        mark_str = "/data_autodrive/users/cczheng2"
        import pdb; pdb.set_trace()
        if mark_str in document["clip_path"]:
            remove_status = True
        if not os.path.exists(document["clip_path"]):
            remove_status = True
        if (night_time(document["seg_uid"])):
            remove_status = True
        
        if (remove_status):
            delete_clip(db_handle, document["seg_uid"])
        

def data_writing(db_handle:PymongoHelper):
    paths = []
    # paths.append("/data_autodrive/users/cczheng2/GridSeg/sihao_1482/20231007/cczheng_tianzige/20230919/")
    #paths.append("/data_cold2/origin_data/sihao_8j998/custom_seg/brlil_chedaoxianyewan/20231030-bt/")
    paths.append("/data_cold2/origin_data/sihao_8j998/custom_seg/brli_chedaoxianyewan/20231031-bt/")
    for path in paths:
        insert_folder_clip(db_handle, path)
    del(db_handle)

def cal_distance(p1:list, p2:list):
    distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return distance

def query_center_area(db_handle:PymongoHelper, center:list, length):
    area = [(center[0] - length), (center[0] + length) , (center[1] - length), (center[1] + length)]
    condition = {"centerX": {"$gte": area[0], "$lte": area[1]}, "centerY": {"$gte": area[2], "$lte": area[3]}}
    collect = db_handle.collects['maps']
    results = collect.find(condition)
    result_dict = {}
    for result in results:
        if "segGrid" not in result:
            continue
        if "clip_lane_annotation" not in result:
            result["clip_lane_annotation"] = ""

        key = result["seg_uid"]
        value = { "clip_path": result["clip_path"], "centerX": result["centerX"], 
            "centerY": result["centerY"] , "segGrid": result["segGrid"] , "match":0 ,
            "clip_lane_annotation": result["clip_lane_annotation"] }
        result_dict[key] = value

    update_dict = {}
    for k, v in result_dict.items():
        dict_center = [v["centerX"], v["centerY"]]
        distance = cal_distance(dict_center, center)
        if (distance < length):
            update_dict[k] = v
    return update_dict

def match_grids(update_dict, seg_id, grids, max_num):
    history_dict = {}
    for k, v in update_dict.items(): #去除相同的历史数据
        if (k != seg_id):
            history_dict[k] = v
    for k, v in history_dict.items():
        history_grid = v["segGrid"]
        element = [x for x in history_grid if x in grids]
        history_dict[k]["match"] = len(element)

    grid_dict = {}
    for k, v in history_dict.items():
        grid_dict[v["clip_path"]] = v["match"]
    sorted_grid = sorted(grid_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_list = []
    for seg_grid in sorted_grid:
        if (len(sorted_list) < max_num):
            sorted_list.append(seg_grid[0])

    return sorted_list

def match_database(db_handle:PymongoHelper, seg_folder, render, length=500):
    meta_path = os.path.join(seg_folder, "meta.json")
    meta_data = json.load(open(meta_path))
    update_dict = query_center_area(db_handle, meta_data["trajectory_center"], length)

    car = meta_data["car"]
    date = meta_data["date"]
    date = date.split("_")[0]
    car_date = car + "_" + date

    seg_folder_split = seg_folder.split("/")
    for char in reversed(seg_folder_split):
        if (len(char) != 0):
            seg_id = char
            break
    sorted_list = match_grids(update_dict, seg_id, meta_data["trajectory_grids"], 2)

    for k, v in update_dict.items():
        if car_date in k:
            if v["clip_path"] not in sorted_list:
                sorted_list.append(v["clip_path"])

    meta_data["multi"] = {}
    meta_data["multi"]["current"] = seg_folder
    meta_data["multi"]["correlation"] = 0
    meta_data["multi"]["export_start"] = 1
    meta_data["multi"]["error"] = 1
    
    meta_data["multi"]["render"] = render
    meta_data["multi"]["match"] = sorted_list
    return meta_data

def get_database_sample(db_handle:PymongoHelper, seg_folder, length=500):
    meta_path = os.path.join(seg_folder, "meta.json")
    meta_data = json.load(open(meta_path))
    update_dict = query_center_area(db_handle, meta_data["trajectory_center"], length)

    car = meta_data["car"]
    date = meta_data["date"]
    date = date.split("_")[0]
    car_date = car + "_" + date

    seg_folder_split = seg_folder.split("/")
    for char in reversed(seg_folder_split):
        if (len(char) != 0):
            seg_id = char
            break
    sorted_list = match_grids(update_dict, seg_id, meta_data["trajectory_grids"], 2)

    for k, v in update_dict.items():
        if car_date in k:
            if v["clip_path"] not in sorted_list:
                sorted_list.append(v["clip_path"])
    return sorted_list

def cal_match_score(update_dict, grids):
    for k, v in update_dict.items():
        history_grid = v["segGrid"]
        element = [x for x in history_grid if x in grids]
        update_dict[k]["match"] = len(element)/len(grids)
    return update_dict

def get_database_distance(db_handle:PymongoHelper, seg_folder, length=500, low_value=0.5, max_num=100):
    meta_path = os.path.join(seg_folder, "meta.json")
    meta_data = json.load(open(meta_path))
    update_dict = query_center_area(db_handle, meta_data["trajectory_center"], length)
    update_dict = cal_match_score(update_dict, meta_data["trajectory_grids"])
    update_dict = sorted(update_dict.items(), key=lambda x: x[1]["match"], reverse=True)

    sorted_list = []
    for k, v in update_dict:
        if (v["match"] > low_value) and (len(sorted_list) < max_num):
            sorted_list.append(v["clip_path"])
    
    return sorted_list 

def get_increment_annotation_data(db_handle:PymongoHelper, seg_folder, length=500, low_value=0.5, max_num=100):
    meta_path = os.path.join(seg_folder, "meta.json")
    meta_data = json.load(open(meta_path))
    update_dict = query_center_area(db_handle, meta_data["trajectory_center"], length)
    update_dict = cal_match_score(update_dict, meta_data["trajectory_grids"])
    update_dict = sorted(update_dict.items(), key=lambda x: x[1]["match"], reverse=True)

    sorted_list = []
    for k, v in update_dict:
        if (v["match"] > low_value) and (os.path.exists(v["clip_lane_annotation"])):
            sorted_list.append(v["clip_path"])
    return sorted_list 
