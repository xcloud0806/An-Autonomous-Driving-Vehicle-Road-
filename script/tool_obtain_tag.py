import csv
import sys
import json
import os
import math
import traceback
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from haversine import haversine, Unit
from datetime import datetime


OBSTACLES_TOKEN = "65a7b2466ce00b33a6e215f3" # 城市障碍物token
URBAN_DATA_COLLECTION_SCENARIOS = "65a7af04fc37cb3794c80c94"  # 城市数据采集场景token
WAY_TYPE1 = "drivable space"
WAY_TYPE2 = "annotatable area"
WAY_TYPE3 = "errorline"
DEGREE_THRESHOLD = 2.0  # 坡度绝对值，大于2degree认为是存在上下坡路段
TAG_PATH = "./utils/tag.json"


def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class TagTree:
    def __init__(self, tag_path):
        self.__tag_path = tag_path
        self.__id_2_node = {}
        self.__tag = None
        self.__big_class_dict = {} # 最外层id2item
        self.__root = None
        self.__initialize()

    def __initialize(self):
        self.__read_tag_info()
        self.__set_big_cls()
        self.__modify_root_struct()  
        self.__set_parent_id_entry()

    def __read_tag_info(self):
        try:
            with open(self.__tag_path, "r") as f:
                self.__tag = json.load(f)
        except Exception as e:
            print(f"Caught an exception of type {type(e).__name__}: {e}")
            traceback.print_exc()

    ## __set_big_cls()任务如下：
    # 建立id 到 根节点 的索引关系   
    def __set_big_cls(self):
        for item in self.__tag:
            self.__big_class_dict[item["id"]] = item

    def __modify_root_struct(self):  # v3.0
        urban_data_collection_scenarios_node = self.__big_class_dict[URBAN_DATA_COLLECTION_SCENARIOS]
        node = urban_data_collection_scenarios_node["children"]
        season = deepcopy(self.__modify_season_struct(node))
        if season:
            node.append(season) # 不添加None值
    
    def __modify_season_struct(self, node):
        season = None
        for item in node:
            if item["enName"] =="time":
                for i in range(len(item["children"])):
                    if item["children"][i]["enName"] == "season":
                        season = item["children"].pop(i)
                        break
                break
        return season

    def __set_parent_id_entry(self):
        for item in self.__tag:
            self.__set_parent_id(item, None)  # 获取大类
    
    ## __set_parent_id()任务如下:
    # 1、为每个节点设置父节点属性
    # 2、初始化每个节点的flag为0
    # 3、建立一个id2node的关系列表
    def __set_parent_id(self, node, id):
        node["parent_id"] = id  # 设置根节点的父节点的id为None
        if(node["children"]!=None):
            for item in node["children"]:
                self.__set_parent_id(item, node["id"])
        node["flag"] = 0   # 标记所有的叶子节点为0
        self.__id_2_node[node["id"]] = node  # 存储所有的树节点，便于查找

    def is_base_id(self, id):
        node = self.__id_2_node[id]
        obstacles_token = OBSTACLES_TOKEN  # "城市障碍物"
        if node["parent_id"]==None:
            return True
        while(node["parent_id"]!=None):
            if node["parent_id"] == obstacles_token:
                return False
            else:
                node = self.__id_2_node[node["parent_id"]]
        return True

    def build_base_info_tree(self, base_tag:list):
        if self.__big_class_dict is not None:
            self.__root = self.__big_class_dict[base_tag[0]]
            self.__build_base_info_tree(base_tag[1:]) # 细化填充基础信息，标记等级为2
        else:
            print("self.__big_class_dict is None")
    
    def __build_base_info_tree(self, base_info:list):   # 基础信息标记为2
        for id in base_info:
            node = self.__id_2_node[id]
            node["flag"] = 2
            while(node["parent_id"] != None):
                node = self.__id_2_node[node["parent_id"]]
                node["flag"] = 2

    def build_other_info_tree(self, base_info:list):   # 非基础信息标记为1
        if  base_info: # base_info也可能是空的，空的就跳过基础信息构建
            for id in base_info:
                node = self.__id_2_node[id]
                node["flag"] = 1
                while(node["parent_id"] != None):
                    node = self.__id_2_node[node["parent_id"]]
                    node["flag"] = 1
        else:
            print("The base_info is None!")
    
    def todict_v1(self):
        result = OrderedDict()
        # result["info"] = self.__root["enName"]
        i = 0
        for item in self.__root["children"]:
            i += 1
            key = None
            content = []
            key = item["enName"].replace(" ", "_")
            self.__todict_v1(item, content)
            if item["flag"] == 2:
                result[key] = deepcopy(content[0])
            else:
                result[key] = deepcopy(content)
        return result 
     
    def __todict_v1(self, node, content:list):
        if node["flag"] != 0:
            if (not node["children"]):  # 叶子节点
                content.append(deepcopy(node["enName"]))

            if node["flag"] == 1:
                node["flag"] = 0  # 标记清理
            for item in node["children"]:
                self.__todict_v1(item, content)                
        else: 
            return

class TagInfo:
    def __init__(self, tag_info_path) -> None:
        self.info = None
        self.__tag_info_path = tag_info_path

    def read_tag_info(self):
        try:
            with open(self.__tag_info_path, "r") as f:
                self.info = json.load(f)
        except Exception as e:
            print(f"Failed execect func: {sys._getframe().f_code.co_name}!")
            print(f"Caught an exception of type {type(e).__name__}: {e}")
            traceback.print_exc()

class Gnss:
    def __init__(self, gnss_path) -> None:
        self.__gnss_dict_inner = dict()
        self.gnss_dict = OrderedDict()
        self.__gnss_path = gnss_path

    def __dict2orderDict(self):
        time_keys = self.__gnss_dict_inner.keys()
        time_keys = sorted(time_keys)
        for key in time_keys:
            self.gnss_dict[key] = self.__gnss_dict_inner[key]

    def read_gnss(self):
        try:
            with open(self.__gnss_path, "r") as f:
                self.__gnss_dict_inner = json.load(f)
            self.__dict2orderDict()
        except Exception as e:
            print(f"Failed execect func: {sys._getframe().f_code.co_name}!")
            print(f"Caught an exception of type {type(e).__name__}: {e}")
            traceback.print_exc()

class Vehicle:
    def __init__(self, vehicle_path) -> None:
        self.__vehicle_dict = dict()
        self.vehicle_dict = OrderedDict()
        self.__vehicle_path= vehicle_path

    def __dict2orderDict(self):
        time_keys = self.__vehicle_dict.keys()
        time_keys = sorted(time_keys)
        for key in time_keys:
            self.vehicle_dict[key] = self.__vehicle_dict[key]

    def read_vehicle(self):
        try:
            with open(self.__vehicle_path, "r") as f:
                self.__vehicle_dict = json.load(f)
            self.__dict2orderDict()
        except Exception as e:
            print(f"Failed execect func: {sys._getframe().f_code.co_name}!")
            print(f"Caught an exception of type {type(e).__name__}: {e}")
            traceback.print_exc()

class SegDataTag:
    def __init__(
            self, tag_tree:TagTree, 
            tag_info:TagInfo, 
            vehicle:Vehicle, 
            gnss:Gnss, 
            meta_path, 
            annotation_path
            ) -> None:
        
        self.__meta_path = meta_path
        self.__meta = self.__read_meta()
        self.__anno_path = annotation_path
        self.__anno = self.__read_anno()

        self.__TagInfo = tag_info
        self.__TagTree = tag_tree
        self.__Vehicle = vehicle
        self.__Gnss = gnss

        self.__base_info_token = []
        self.__other_info_token = [] 
        self.__vehicle_spd = 0  # 单位是：km/h
        self.__has_slope_state = False # True为存在上下坡路段，Fasle反之

    def __read_meta(self):
        try:
            with open(self.__meta_path, "r") as f:
                meta = json.load(f)
            return meta
        except Exception as e:
            print(f"Failed execect func: {sys._getframe().f_code.co_name}!")
            print(f"Caught an exception of type {type(e).__name__}: {e}")
            traceback.print_exc()

    def __read_anno(self):
        try:
            with open(self.__anno_path, "r") as f:
                anno = json.load(f)
                return anno
        except Exception as e:
            print(f"Failed execect func: {sys._getframe().f_code.co_name}!")
            print(f"Caught an exception of type {type(e).__name__}: {e}")
            traceback.print_exc()

    def __load_base_tag(self):
        try:
            base_tag_tokens = self.__TagInfo.info["tags"][0]["tagIds"]
            for id in base_tag_tokens:
                if self.__TagTree.is_base_id(id):  # 判断id对应的条目是否是基础信息（通用）
                    self.__base_info_token.append(id)
        except Exception as e:
            print(f"Failed execect func: {sys._getframe().f_code.co_name}!")
            print(f"Caught an exception of type {type(e).__name__}: {e}")
            traceback.print_exc()
           
    def __load_other_tag(self):
        try:
            seg_time_start = float(self.__meta["frames"][0]["gnss"])
            seg_time_interval = float(self.__meta["time_interval"])
            seg_time_end = seg_time_start + seg_time_interval
            for tag in self.__TagInfo.info["tags"]:
                tag_time = float(tag["timestamp"])*1000  # 单位变换
                if seg_time_start < tag_time < seg_time_end:
                    for id in tag["tagIds"]:
                        if self.__TagTree.is_base_id(id):
                            continue
                        else:
                            self.__other_info_token.append(id)
        except Exception as e:
            print(f"Failed execect func: {sys._getframe().f_code.co_name}!")
            print(f"Caught an exception of type {type(e).__name__}: {e}")
            traceback.print_exc()

    def __load_vehicle_spd(self):
        try:
            speed = []
            seg_time_start = float(self.__meta["frames"][0]["gnss"])
            seg_time_interval = float(self.__meta["time_interval"])
            seg_time_end = seg_time_start + seg_time_interval
            for timestamp in self.__Vehicle.vehicle_dict.keys():
                if seg_time_start < float(timestamp) < seg_time_end:
                    speed.append(self.__Vehicle.vehicle_dict[timestamp]["vehicle_spd"])
            top95 = math.ceil(len(speed)*0.95)    
            self.__vehicle_spd = speed[top95]
        except Exception as e:
            print(f"Failed execect func: {sys._getframe().f_code.co_name}!")
            print(f"Caught an exception of type {type(e).__name__}: {e}")
            traceback.print_exc()

    def __analyzing_slope_status_based_on_gnss(self):
        try:
            has_slope_state = False
            sample_step = 100  # 1s采集一个样本
            key_node = []
            length = len(self.__Gnss.gnss_dict.keys()) // sample_step
            for i in range(length):
                key = list(self.__Gnss.gnss_dict.keys())[i*sample_step]
                key_node.append(key)
            # 计算两两关键点之间的坡度值
            key_node_slop = []
            for i in range(1, len(key_node)):
                previous_key = key_node[i-1]
                previous_longitude = float(self.__Gnss.gnss_dict[previous_key]["longitude"])
                previous_latitude = float(self.__Gnss.gnss_dict[previous_key]["latitude"])
                previous_altitude = float(self.__Gnss.gnss_dict[previous_key]["altitude"])
                current_key = key_node[i]
                current_longitude = float(self.__Gnss.gnss_dict[current_key]["longitude"])
                current_latitude = float(self.__Gnss.gnss_dict[current_key]["latitude"])
                current_altitude = float(self.__Gnss.gnss_dict[current_key]["altitude"])
                pa = (previous_latitude, previous_longitude)
                pb = (current_latitude, current_longitude)
                dist = haversine(pa, pb, unit=Unit.METERS)
                if abs(dist) <= 0.01:
                    continue
                height = current_altitude - previous_altitude
                degree = math.atan(height/dist) * 180 / math.pi
                key_node_slop.append(abs(degree))

            key_node_slop_array = np.abs(key_node_slop)
            if key_node_slop_array.max() > DEGREE_THRESHOLD:
                has_slope_state = True
            self.__has_slope_state = has_slope_state
        except Exception as e:
            print(f"Failed execect func: {sys._getframe().f_code.co_name}!")
            print(f"Caught an exception of type {type(e).__name__}: {e}")
            traceback.print_exc()

    def get_annotations_tag_(self):
        lane_key= "lane"
        lane_scenes = self.__anno[lane_key]["image_type"]["scene"].split(",")
        wayShuxing = set()
        for way in self.__anno[lane_key]["way"]:
            temp = way["type"]["shuxing"]
            if temp == WAY_TYPE1 or temp == WAY_TYPE2: 
                continue
            elif temp == WAY_TYPE3:
                wayShuxing.clear()
                break
            wayShuxing.add(temp)
        wayShuxing = list(wayShuxing)

        obstacle = "obstacle"
        obstacle_set = set()
        for timestamp in self.__anno[obstacle]["annotations"].keys():
            for object in self.__anno[obstacle]["annotations"][timestamp]:
                obstacle_set.add(object["class_name"])
        
        obstacles = list(obstacle_set)
        return lane_scenes, wayShuxing, obstacles

    def __update_anno_file(self):
        self.__anno["clip_info"]["data_tags"] = {}
        self.__TagTree.build_base_info_tree(self.__base_info_token)
        self.__TagTree.build_other_info_tree(self.__other_info_token)
        self.__anno["clip_info"]["data_tags"] = self.__TagTree.todict_v1()
        lane_scenes, wayShuxing, obstacles = self.get_annotations_tag_()
        self.__anno["clip_info"]["data_tags"]["lane_scenes"] = lane_scenes
        self.__anno["clip_info"]["data_tags"]["way_type"] = wayShuxing
        self.__anno["clip_info"]["data_tags"]["urban_obstacles"] = obstacles
        self.__anno["clip_info"]["data_tags"]["vehicle_spd"] = self.__vehicle_spd
        self.__anno["clip_info"]["data_tags"]["has_slope_state"] = self.__has_slope_state

    def __save_file(self):
        ss = json.dumps(self.__anno, ensure_ascii=False, default=dump_numpy)
        with open(self.__anno_path, "w") as fp:
            fp.write(ss)

    def run(self):
        self.__load_base_tag()
        self.__load_other_tag()
        self.__load_vehicle_spd()
        self.__analyzing_slope_status_based_on_gnss()
        self.__update_anno_file()
        self.__save_file()

def obtain_seg_tag(run_config, tag_path = TAG_PATH, annotation_train_or_test = None):
    if os.path.exists(TAG_PATH):
        print(TAG_PATH)
    seg_root_path = run_config["preprocess"]["segment_path"]
    frames_path = run_config['preprocess']['frames_path']
    clip_submit = run_config['deploy']['clip_submit']
    data_subfix = run_config['deploy']['data_subfix']
    seg_config = run_config["preprocess"]
    spec_clips = seg_config.get("spec_clips", None)
    train_or_test = []
    if annotation_train_or_test != None:
        train_or_test.append(annotation_train_or_test)
    else:
        train_or_test = ["annotation_train", "annotation_test"]
    tag_tree = TagTree(tag_path)
    if not os.path.exists(seg_root_path):
        print(f"{seg_root_path} NOT Exist...")
        sys.exit(0)
    seg_names = os.listdir(seg_root_path)
    seg_names.sort()
    skiped_segs = set()
    dealed_segs = set()
    print(f"\n{str(datetime.now())}:prepare tag data, {seg_root_path}, start... ")
    for item in train_or_test:
        annotation_path = os.path.join(clip_submit, item, data_subfix)
        for i, _seg in enumerate(seg_names):
            if spec_clips is not None:
                go_on = False
                for clip in spec_clips:
                    if clip in _seg:
                        go_on = True
                        break
                if not go_on:
                    continue            
            seg_dir = os.path.join(seg_root_path, _seg)
            if not os.path.exists(os.path.join(seg_dir, "meta.json")):
                skiped_segs.add(_seg)
                continue

            meta_path = os.path.join(seg_dir, "meta.json")
            anno_path = os.path.join(annotation_path, _seg, "annotation.json")
            if not os.path.exists(anno_path):
                skiped_segs.add(_seg)
                continue
            print("\tdealing with seg {}".format(_seg))
            dealed_segs.add(_seg)
            clip_data = _seg.split("_")[2]
            tag_info_path = os.path.join(frames_path, clip_data, "tag_info.json")
            tag_info = TagInfo(tag_info_path)
            tag_info.read_tag_info()
            
            vehicle_path = os.path.join(seg_dir, "vehicle.json")
            vehicle = Vehicle(vehicle_path)
            vehicle.read_vehicle()

            gnss_path = os.path.join(seg_dir, "gnss.json")
            gnss = Gnss(gnss_path)
            gnss.read_gnss()

            seg_data_tag = SegDataTag(tag_tree, tag_info, vehicle, gnss, meta_path, anno_path)
            seg_data_tag.run()
    for _seg in list(skiped_segs - dealed_segs):
        print("\tskip seg {} for not seleted".format(_seg))
    print(f"{str(datetime.now())}:prepare tag data, {seg_root_path}, end... ")   


if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)
    
    with open(config_file, 'r') as fp:
        run_config = json.load(fp)
    
    obtain_seg_tag(run_config, TAG_PATH)
    