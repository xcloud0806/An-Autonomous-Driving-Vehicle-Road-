import sys, os
sys.path.append("../utils")
from db_utils import query_seg
import pandas as pd
import json
import numpy as np
from loguru import logger
from multiprocessing.pool import Pool
from overpy import Overpass

import pymongo
from pymongo.errors import PyMongoError
import traceback as tb
from datetime import datetime
from copy import deepcopy

MONGO_USER = "brli"
MONGO_PASS = "lerinq1w2E#R$"
MONGO_IP = "172.30.35.11"
MONGO_PORT = 27017

RECORD_KEY="record"
MAIN_KEY="segid"

sunrise = {
    "1_3":{ "start": "0700", "finish": "1822"},
    "4_6":{ "start": "0546", "finish": "1901"},
    "7_9":{ "start": "0604", "finish": "1858"}, 
    "10_12":{ "start": "0642", "finish": "1742"}
}

logger.add("export_segs_mongodb.log", rotation="10 MB", level="INFO")


xlsx_files = [
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_1st_2nd.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0516.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_3rd.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0522.xlsx"
]
loss_json = [
    "/data_autodrive/users/brli/dev_raw_data/refined/loss_list_0522.json",
    "/data_autodrive/users/brli/dev_raw_data/refined/loss_list_0516.json",
    "/data_autodrive/users/brli/dev_raw_data/refined/loss_list_0523.json",
    "/data_autodrive/users/brli/dev_raw_data/refined/loss_list_0527.json"
]

PREANNO_ROOT = "/data_autodrive/auto/label_4d/result/post_delete/"


def multi_process_error_callback(error):
    # get the current process
    process = os.getpid()
    # report the details of the current process
    print(f"Callback Process: {process}, Exeption {error}", flush=True)


def night_time(name):
    night_state = True
    time_list = name.split("_")
    year_month = ""
    day_time = ""
    for time in time_list:
        time_split = time.split("-")
        if len(time_split) == 4:
            year_month = time_split[0]
            day_time = time_split[1] + time_split[2]

    if len(year_month)==0 or len(day_time)==0:
        return night_state
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
                logger.error(f"[MongoDB] block timed out: {exc!r}")
            else:
                logger.error(f"[MongoDB] failed with non-timeout error: {exc!r}")
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
                logger.error(f"[MongoDB] Some segs already exist in the collection {coll}")                
                return False        
            ret = collect.insert_many(infos)
            logger.info(f"[MongoDB] Successfully inserted clip data from {len(infos)} segs")
            return True
        except Exception as e:
            logger.error(f"[MongoDB] Failed to insert clip data due to {e}")
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
                logger.error(f"[MongoDB] {segid} not exists.")      
                return False
            collect.update_one(condition, new_values)
            logger.info(f"[MongoDB] Successfully updated clip data from {segid}")
            return True
        except Exception as e:
            logger.error(f"[MongoDB] Failed to update clip data from {segid} due to {e}")
            tb.print_exc()
            return False
        
    def query(self, segid, coll):
        collect = self.collects[coll]
        try:
            condition = {MAIN_KEY: segid}
            ret = collect.find_one(condition)
            if ret is None: 
                logger.info(f"[MongoDB] {segid} not exists in {coll}.")      
                return {}
            return ret
        except Exception as e:
            logger.error(f"[MongoDB] Failed to query clip data from {segid} due to {e}")
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
            logger.info(f"[MongoDB] Successfully deleted clip data from {segid}")
            return True
        except Exception as e:
            logger.error(f"[MongoDB] Failed to delete clip data from {segid} due to {e}")
            tb.print_exc()
            return False
    
    def __del__(self):
        self.status = False
        self.client.close()

def judge_China(lon, lat):
    if lon < 70 or lon > 140:
        return True
    if lat < 0 or lat > 55:
        return True
    return False

def get_ways_name(wgs84_lon_lst: list, wgs84_lat_lst: list):
    wgs84_lon_arr = np.array(wgs84_lon_lst)
    wgs84_lat_arr = np.array(wgs84_lat_lst)

    wgs84_lst = [
        np.min(wgs84_lat_lst),
        np.min(wgs84_lon_lst),
        np.max(wgs84_lat_lst),
        np.max(wgs84_lon_lst),
    ]
    lat_0 = wgs84_lst[0]
    lon_0 = wgs84_lst[1]
    lat_1 = wgs84_lst[2]
    lon_1 = wgs84_lst[3]
    query_str = "[out:json][timeout:30];way[highway]({},{},{},{});(._;>;);out;".format(
        lat_0, lon_0, lat_1, lon_1
    )

    api = Overpass()
    result = api.query(query_str)
    if len(result.ways) == 0:
        return {}
    ret = {}
    for way in result.ways:
        way_id = way.id
        if way.tags.get("highway"):
            name = way.tags.get("name:en")
            highway = way.tags.get("highway")
            if name not in ret:
                ret[way_id] = {
                    "highway": highway,
                    "name:en": name,
                    "name:zh": way.tags.get("name"),
                    "infos": way.tags,
                    "count": 1,
                }
            else:
                ret[name]["count"] += 1
    return ret

def get_city_name(wgs84_loc: list):
    latitude = wgs84_loc[0]
    longitude = wgs84_loc[1]
    api = Overpass()

    # 创建 Overpass 查询
    query = f"""
        node({latitude},{longitude})[place=city];
        out geom;
    """

    # 执行查询
    result = api.query(query)

    # 检查是否找到任何结果
    cities = [node for node in result.nodes if 'place' in node.tags and node.tags['place'] == 'city']

    # 返回第一个匹配的城市名称
    if cities:
        return cities[0].tags['name']
    else:
        return None

def get_main_way_by_gnss_json(gnss_json: str):
    if not os.path.exists(gnss_json):
        return []
    gnss_fp = open(gnss_json, "r")
    gnsses = json.load(gnss_fp)
    tss = list(gnsses.keys())
    # indexs = [0, len(tss)/4, len(tss)/2, len(tss)*3/4, len(tss)-1]
    indexs = [
        0,
        len(tss) / 8,
        len(tss) / 4,
        len(tss) * 3 / 8,
        len(tss) / 2,
        len(tss) * 5 / 8,
        len(tss) * 3 / 4,
        len(tss) * 7 / 8,
        len(tss) - 1,
    ]
    pre_lon = float(gnsses[tss[0]]["longitude"])
    pre_lat = float(gnsses[tss[0]]["latitude"])
    city = get_city_name([pre_lat, pre_lon])
    ways = {}
    for idx in indexs[1:]:
        ts = tss[int(idx)]
        loc = gnsses[ts]
        lon = float(loc["longitude"])
        lat = float(loc["latitude"])
        if judge_China(lon, lat):
            continue

        cur_ts = int(ts)
        way = get_ways_name([pre_lon, lon], [pre_lat, lat])
        for name, info in way.items():
            if name not in ways:
                ways[name] = info
            else:
                ways[name]["count"] += info["count"]
    # 按照count 对way 进行排序，获取way name
    way_names = sorted(ways.items(), key=lambda x: x[1]["count"], reverse=True)
    roads = {}
    for k, v in way_names:
        if v["name:en"] is None and v["name:zh"] is None:
            continue

        if v["name:zh"] not in roads:
            roads[v["name:zh"]] = v
        else:
            roads[v["name:zh"]]["count"] += v["count"]
    max_cnt = 0
    for k, v in roads.items():
        if v["count"] > max_cnt:
            max_cnt = v["count"]
    ret = []
    for k, v in roads.items():
        if v["count"] == max_cnt:
            ret.append(v)
    return city, ret

def parse_infos_with_excels(xlsx_files: list):
    ret = {}
    total_cnt = 0
    for xlsx_file in xlsx_files:
        df = pd.read_excel(xlsx_file, skiprows=1)
        _total = df.shape[0]
        total_cnt += _total
        # pool = Pool(processes=16)
        for idx, row in df.iterrows():
            segid, objcnt, speed, daynight, task, car, deploy_subfix = row
            # (objcnt, speed, daynight, task, car, deploy_subfix)
            ret[segid] = {
                "objcnt": objcnt,
                "speed": speed,
                "daynight": daynight,
                "task": task,
                "car": car,
                "deploy_subfix": str(deploy_subfix),
                "segid": segid,
            }
    logger.info(f"EXCEL total_count: {total_cnt}")
    return ret

def update_seg_infos(seg_infos, db_handle:PymongoHelper):
    ret = {}
    preanno_batches = os.listdir(PREANNO_ROOT)
    for batch in preanno_batches:
        annos_path = os.path.join(PREANNO_ROOT, batch)
        cars = os.listdir(annos_path)
        cars.sort()        
        for car in cars:
            car_annos_root = os.path.join(annos_path, car)
            subfixes = os.listdir(car_annos_root)
            subfixes.sort()
            for subfix in subfixes:
                subfix_path = os.path.join(car_annos_root, subfix)
                segs = os.listdir(subfix_path)
                segs.sort()
                for seg in segs:
                    seg_path = os.path.join(subfix_path, seg)
                    if not os.path.isdir(seg_path):
                        continue
                    if not os.path.exists(
                        os.path.join(seg_path, "annotations", "result.json")
                    ):
                        continue
                    if seg in ret:
                        logger.info(f"[{seg}] already updated")
                        continue

                    if seg not in seg_infos:
                        logger.info(f"[{seg}] not in excel")
                        continue

                    seg_info = seg_infos[seg]
                    speed = float(seg_info["speed"])
                    daynight = "night" if night_time(seg) else "day"
                    task = seg_info["task"]

                    res = query_seg([seg])
                    res_cnt = res[0]

                    seg_road_name = ""
                    seg_road_type = ""
                    seg_road = []
                    if res_cnt > 0:
                        seg_content = res[1][0]
                        seg_clip_obs_path = seg_content["pathMap"]["obstacle3dAnnoDataPath"]
                        car_name = seg_content["calibrationCar"]
                        gnss_json_path = seg_content["gnssFilePath"]
                        veh_json_path = seg_content['vehicleDriverFilePath']
                        if not gnss_json_path or not os.path.exists(gnss_json_path):
                            logger.info(
                                f"[{car}.{subfix}.{seg}] gnss_json_path: {gnss_json_path} not exists"
                            )
                            continue
                        logger.info(f"[{car}.{subfix}.{seg}]")

                        # seg_road = get_main_way_by_gnss_json(gnss_json_path)
                        # if len(seg_road) > 0:
                        #     logger.info(f"[{seg}] seg_road: {seg_road}")
                        #     seg_road_name = seg_road[0].get("name:zh")
                        #     seg_road_type = seg_road[0].get("highway")

                        result_json = os.path.join(seg_path, "annotations", "result.json")
                        with open(result_json, "r") as fp:
                            result = json.load(fp)
                        frame_annos = result["researcherData"]
                        frame_cnt = len(frame_annos)
                        bbox_cnt = 0
                        for _f in frame_annos:
                            _anno = _f["frame_annotations"]
                            _bbox = len(_anno)
                            bbox_cnt += _bbox
                        ret[seg] = [
                            speed,
                            daynight,
                            task,
                            car,
                            subfix,
                            frame_cnt,
                            bbox_cnt,
                            seg_clip_obs_path,
                            seg_road_name,
                            seg_road_type,
                            seg_road,
                            gnss_json_path,
                            veh_json_path
                        ]
                    else:
                        ret[seg] = [
                            speed,
                            daynight,
                            task,
                            car,
                            subfix,
                            0,
                            0,
                            None,
                            seg_road_name,
                            seg_road_type,
                            seg_road,
                            gnss_json_path,
                            veh_json_path
                        ]
                    insert_info = {
                        "segid": seg,
                        "speed": speed,
                        "daynight": daynight,
                        "task": task,
                        "car": car,
                        "deploy_subfix": subfix,
                        "frame_cnt": frame_cnt,
                        "bbox_cnt": bbox_cnt,                        
                        "seg_road_name": seg_road_name,
                        "seg_road_type": seg_road_type,
                        "seg_road": seg_road,
                        "gnss_file": gnss_json_path,
                        "veh_file": veh_json_path,
                        "city": ""
                    }
                    if not db_handle.insert(insert_info, "infos"):
                        db_handle.update(insert_info, "infos")
    return ret

def export_mongodb():
    db_handle = PymongoHelper(MONGO_IP, MONGO_PORT, "segments", ["infos"])

    seg_infos = parse_infos_with_excels(xlsx_files)
    seg_infos = update_seg_infos(seg_infos, db_handle)

    def parse_loss_json():
        ret = {}
        for f in loss_json:
            with open(f, "r") as fp:
                losss = json.load(fp)
            for k, v in losss.items():
                segid = os.path.basename(k)
                loss = v
                ret[segid] = v
        return ret
    loss = parse_loss_json()   
    
    for segid, info in seg_infos.items():
        (
            speed,
            daynight,
            task,
            car,
            deploy_subfix,
            frame_cnt,
            bbox_cnt,
            seg_obs_path,
            seg_road_name,
            seg_road_type,
            seg_road,
            gnss_json_path,
            veh_json_path
        ) = info
        try:
            city, roads = get_main_way_by_gnss_json(gnss_json_path)
        except Exception as e:
            logger.error(f"[{segid}] get_main_way_by_gnss_json error: {e}")
            city = ""
            roads = []

        updated_info = {
            "segid": segid,
            "speed": speed,
            "daynight": daynight,
            "task": task,
            "car": car,
            "deploy_subfix": deploy_subfix,
            "frame_cnt": frame_cnt,
            "bbox_cnt": bbox_cnt,
            # "seg_obs_path": seg_obs_path,
            "seg_road_name": seg_road_name,
            "seg_road_type": seg_road_type,
            "seg_road": seg_road,
            "city": city,
            "roads": roads
        }        

        if segid in loss:
            updated_info["loss"] = loss[segid]
            db_handle.update(updated_info, "infos")

if __name__ == "__main__":
    export_mongodb()
