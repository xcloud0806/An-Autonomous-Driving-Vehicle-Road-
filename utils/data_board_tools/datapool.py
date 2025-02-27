import os
import json
import shutil
from collections import OrderedDict
from typing import Union, Optional
TARGET_FILE_NAME = "databoard.json"


# def ordered_dict_serializer(o):
#     if isinstance(o, OrderedDict):
#         return dict(o)
#     raise TypeError("Object of type '{}' is not JSON serializable".format(type(o)))

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

class Exectime:
    def __init__(self):
        self.__exec_times = OrderedDict()  # 存储函数执行时间的字典
        self.enabled = True

    def __call__(self, func):
        if not self.enabled:
            return func
        
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()  # 单位是s
            res = func(*args, **kwargs)
            end_time = time.time()  # 单位是s
            exec_time = end_time - start_time
            exec_time = "{:.6f}".format(float(exec_time))  # 保存六位小数

            func_name = func.__name__
            if func_name not in self.__exec_times:
                self.__exec_times[func_name] = []
            self.__exec_times[func_name].append(exec_time)  # 存储每个函数的执行时间
            return res
        return wrapper

    def get_time_cost(self):
        return self.__exec_times

    def print_total_exec_times(self):
        print("\nTotal Exec Times:")
        # all_time_cost = 0.0
            
        ## 这段内容是时间的顺序输出，将其改为逆序输出
        # for func_name, exec_time in self.__exec_times.items():
        #     exec_times_str = ', '.join(map(str, exec_time))
        #     # all_time_cost += sum(exec_time)
        #     # func_name.ljust(15) 会将 func_name 的长度左对齐填充到15个字符，确保整体对齐。
        #     print("\tFuncName: {} \t==> Exec(unit:{}): {}".format(func_name.ljust(15), self.unit, exec_times_str))
            
        ## 逆序输出
        for func_name, exec_time in reversed(self.__exec_times.items()):
            exec_times_str = ', '.join(map(str, exec_time))
            # all_time_cost += sum(exec_time)
            # func_name.ljust(15) 会将 func_name 的长度左对齐填充到15个字符，确保整体对齐。
            print("\tFuncName: {} \t==> Exec(unit:s): {}".format(func_name.ljust(15),  exec_times_str))
        print("\nEnd!") 

class SegAnno():
    """记录pack_anno节点的看板数据,包括标注信息的统计"""
    def __init__(self):
        self.initialize()
        
    def initialize(self):
        # 属性信息
        self.__annotation_attribute = None
        self.__lane_annotaion = None
        self.__obstacle_annotaion = None
        self.__curvature_radius = None
        # 统计信息
        self.__num_of_bbox = 0
        self.__num_of_key_frame = 0 
        self.__class_and_num_of_obstacle = OrderedDict()
        self.__type_and_num_of_lane = OrderedDict()
    
    def parse_annotation(self, annotation:dict):
        self.__set_annotation_attribute(annotation)
        self.__set_type_and_num_of_lane(annotation)
        self.__set_class_and_num_of_obstacle(annotation)  
        self.__set_curvature_r(annotation)  
    
    def __set_annotation_attribute(self, annotation):
        self.__annotation_attribute =  annotation["clip_info"]["datasets"]
    
    def __set_type_and_num_of_lane(self, annotation:dict):
        lane_class = annotation["lane"]["image_type"]["reconstruction-error"]
        if lane_class not in self.__type_and_num_of_lane.keys():
            self.__type_and_num_of_lane[lane_class] = 0
        self.__type_and_num_of_lane[lane_class] += 1
    
    def __set_class_and_num_of_obstacle(self, annotation:dict):
        for frame in annotation["obstacle"]["annotations"].keys():
            self.__num_of_key_frame += 1
            for box in annotation["obstacle"]["annotations"][frame]:
                self.__num_of_bbox += 1
                obstacle_class = box["class_name"]
                if obstacle_class not in self.__class_and_num_of_obstacle.keys():
                    self.__class_and_num_of_obstacle[obstacle_class] = 0
                self.__class_and_num_of_obstacle[obstacle_class] += 1
    
    def __set_curvature_r(self, annotation:dict):
        self.__curvature_radius = float(annotation["clip_info"]["data_tags"]["curvature_radius"])
    
    def set_annotation_lane_attribute(self, lane_flag:bool):
        self.__lane_annotaion = lane_flag

    def set_annotation_obstacle_attribute(self, obstacle_flag:bool):
        self.__obstacle_annotaion = obstacle_flag
    
    # 对于统计信息设置对外接口       
    def get_num_of_bbox(self):
        return self.__num_of_bbox
    
    def get_num_of_key_frame(self):
        return self.__num_of_key_frame
    
    def get_type_and_num_of_lane(self):
        return self.__type_and_num_of_lane

    def get_class_and_num_of_obstacle(self):
        return self.__class_and_num_of_obstacle 
    
    def get_curvature_radius(self):
        return self.__curvature_radius
    
    # 对于单段数据的汇总信息设置对外输出接口
    def get_variables_dict(self):
        variables_dict = {
                # 属性信息
                "annotation_attribute": self.__annotation_attribute, 
                "lane_annotaion": self.__lane_annotaion,
                "obstacle_annotaion": self.__obstacle_annotaion,
                # 统计信息
                "num_of_bbox": self.__num_of_bbox,
                "num_of_key_frame": self.__num_of_key_frame,
                "class_and_num_of_obstacle": self.__class_and_num_of_obstacle,
                "type_and_num_of_lane": self.__type_and_num_of_lane
                }
        return variables_dict

class SegLmdb():
    """用于记录lmdb_pack的一些统计数据"""
    def __init__(self) -> None:
        self.initialize()
        
    def initialize(self):
        self.__lmdb_size = 0
        self.__lmdb_hash = 0
        self.__frame_count = OrderedDict()
        self.__total_count = 0
    
    def parse_lmdb_info(self, lmdb_info:dict):
        self.__lmdb_size = lmdb_info["lmdb_size"]
        self.__lmdb_hash = lmdb_info["lmdb_hash"]
        self.__frame_count = lmdb_info["frame_cnt"]
        self.__total_count = lmdb_info["total_cnt"]

    def get_lmdb_size(self):
        return self.__lmdb_size

    def get_lmdb_hash(self):
        return self.__lmdb_hash

    def get_frame_count(self):
        return self.__frame_count
    
    def get_total_count(self):
        return self.__total_count
    
    def get_variables_dict(self):
        variables_dict = {
            "lmdb_size": self.__lmdb_size,
            "lmdb_hash": self.__lmdb_hash,
            "total_count": self.__total_count,
            "frame_count": self.__frame_count
        }
        return variables_dict

class SegBase():
    """用于记录gen_seg节点的基础seg数据"""
    def __init__(self) -> None:
        self.initialize()

    def initialize(self):
        self.__distance = 0.0
        self.__time_start = 0
        self.__time_interval = 0
        self.__num_of_image_lost = 0
    
    def set_distance(self, value:Union[int, float, None]):
        """设置seg持续行驶里程"""
        self.__distance = value
    
    def set_time_start(self, value:int):
        """设置Segs数据起始时间戳"""
        self.__time_start = value
    
    def set_time_interval(self, value:Union[int, float]):
        """设置Seg数据持续时间"""
        self.__time_interval = value
    
    def set_num_of_image_lost(self, value:int):
        self.__num_of_image_lost = value
    
    def get_distance(self):
        return self.__distance
    
    def get_time_interval(self):
        return self.__time_interval
    
    def get_num_of_image_lost(self):
        return self.__num_of_image_lost

    def get_variables_dict(self) -> dict:
        variables_dict = {
            "distance": self.__distance,
            "time_start": self.__time_start,
            "time_interval": self.__time_interval,
            "num_of_image_lost": self.__num_of_image_lost,
        }
        return variables_dict

class SegPrepareAnno():
    def __init__(self) -> None:
        self.initialize()
        
    def initialize(self):
        self.__exist_loadable_meta = None
        self.__skip_reconstruct = None
        self.__reconstruct_ok = None
        self.__pose_normality = None
        self.__frames_lost_to_limit = None
        self.__obstacle_data_ok = None
        self.__lane_data_ok = None
        self.__obstacle_data_attribute = None

    def set_exist_loadable_meta(self, value:Optional[bool]):
        self.__exist_loadable_meta = value
        
    def set_pose_normality(self, value:Optional[bool]):
        self.__pose_normality = value
        
    def set_skip_reconstruct(self, value:Optional[bool]):
        self.__skip_reconstruct = value
    
    def set_reconstruct_ok(self, value:Optional[bool]):
        self.__reconstruct_ok = value
  
    def set_frames_lost_to_limit(self, value:Optional[bool]):
        self.__frames_lost_to_limit = value
        
    def set_obstacle_data_ok(self, value:Optional[bool]):
        self.__obstacle_data_ok = value
        
    def set_lane_data_ok(self, value:Optional[bool]):
        self.__lane_data_ok = value
        
    def set_obstacle_data_attribute(self, value:Optional[str]):
        self.__obstacle_data_attribute = value
        
    # def get_exist_loadable_meta(self):
    #     return self.__exist_loadable_meta
    
    # def get_pose_normality(self):
    #     return self.__pose_normality
    
    # def get_skip_reconstruct(self):
    #     return self.__skip_reconstruct
    
    # def get_frames_lost_to_limit(self):
    #     return self.__frames_lost_to_limit
    
    # def get_obstacle_data_ok(self):
    #     return self.__obstacle_data_ok
    
    # def get_lane_data_ok(self):
    #     return self.__lane_data_ok
    
    def get_obstacle_data_attribute(self):
        return self.__obstacle_data_attribute

    def get_variables_dict(self) -> dict:
        variables_dict = {
                "exist_loadable_meta" : self.__exist_loadable_meta,
                "pose_normality" : self.__pose_normality,
                "skip_reconstruct" : self.__skip_reconstruct,
                "reconstruct_ok": self.__reconstruct_ok,
                "frames_lost_to_limit" : self.__frames_lost_to_limit,
                "obstacle_data_ok" : self.__obstacle_data_ok, 
                "lane_data_ok" : self.__lane_data_ok,
                "obstacle_data_attribute" : self.__obstacle_data_attribute,
        }
        return variables_dict
    
class BaseClass:
    def __init__(self) -> None:
        self.__name = None
    
    def get_name(self) -> str:
        return self.__name
    
    def set_name(self, name):
        self.__name = name
 
class ClipMatch:
    """主要用于node_match_frame节点的重要数据统计,数据多是clip的属性"""
    def __init__(self) -> None:
        self.initialize()
        
    def initialize(self):
        self.__cameras_lost = None
        self.__bpearl_lidar_lost = None
        self.__inno_lidar_lost = None
        self.__num_of_lidar_frame = 0
        self.__num_of_lidar_frame_lost = 0
        self.__num_of_image_frame_lost = 0
        self.__num_of_bpearl_lidars_frame_lost = 0
        self.__num_of_inno_lidars_frames_lost = 0
        self.__match_type = None
    
    def set_match_type(self, match_type:str):
        """设置匹配模式为："match",或者为"raw"."""
        self.__match_type = match_type
        
    def set_num_of_lidar_frame(self, num:int = 0):
        """设置激光雷达的总帧数,默认值是0"""
        self.__num_of_lidar_frame = num
    
    def set_num_of_image_frame_lost(self, num:int = 0):
        """设置所有的相机丢帧的总数量,默认为0"""
        self.__num_of_image_frame_lost = num
    
    def set_num_of_lidar_frame_lost(self, num:int = 0):
        """设置雷达的丢帧数量"""
        self.__num_of_lidar_frame_lost = num
        
    def set_num_of_bpearl_lidars_frame_lost(self, num:Optional[int] = None):
        """设置所有的补盲雷达丢帧的总量,None值则代表不存在补盲雷达"""
        self.__num_of_bpearl_lidars_frame_lost = num
    
    def set_num_of_inno_lidars_frames_lost(self, num:Optional[int] = None):
        """设置所有的鹰眼雷达丢帧的总数量,None值则代表不存在鹰眼雷达"""
        self.__num_of_inno_lidars_frames_lost = num
        
    def add_camera_lost(self, cam:list = []):
        """记录应该有但是缺失了的相机的名称, 默认为空"""
        if cam != None:
            if not isinstance(self.__cameras_lost, list):
                self.__cameras_lost = list()
            self.__cameras_lost.extend(cam)
    
    def add_bpearl_lidar_lost(self, lidar:Optional[list] = None):
        """
            bpearl(补盲雷达), 该接口记录应该有但是缺失了的bpearl传感器名称;
            具体说明,同 add_inno_lidar_lost 方法
        """
        if lidar != None:
            if not isinstance(self.__bpearl_lidar_lost, list):
                self.__bpearl_lidar_lost = list()
            self.__bpearl_lidar_lost.extend(lidar)
    
    def add_inno_lidar_lost(self, lidar:Optional[list] = None):
        """
            inno(鹰眼雷达),该接口记录应该有但是缺失了的inno传感器名称;
            若是None值, 则代表没有用到鹰眼相机;
            若是list类型,不论是不是空,都代表使用了 inno 传感器，
                若为空列表,则代表使用了鹰眼雷达,但是没有缺失应有的传感器
        """
        if lidar != None:
            if not isinstance(self.__inno_lidar_lost, list):
                self.__inno_lidar_lost = list()
            self.__inno_lidar_lost.extend(lidar)
    
    def get_match_type(self) -> str:
        return self.__match_type
    
    def get_variables_dict(self) -> dict:
        variables_dict = {
            "cameras_lost": self.__cameras_lost,
            "bpearl_lidar_lost": self.__bpearl_lidar_lost,
            "inno_lidar_lost": self.__inno_lidar_lost,
            "num_of_lidar_frame": self.__num_of_lidar_frame,
            "num_of_lidar_frame_lost": self.__num_of_lidar_frame_lost,
            "num_of_image_frame_lost": self.__num_of_image_frame_lost,
            "num_of_bpearl_lidars_frame_lost":
                self.__num_of_bpearl_lidars_frame_lost,
            "num_of_bpearl_inno_frame_lost":
                self.__num_of_inno_lidars_frames_lost
        }
        return variables_dict

class Seg(BaseClass):
    def __init__(self) -> None:
        super().__init__()
        self.initialize()
        
    def initialize(self):
        self._anno:SegAnno = None
        self._lmdb:SegLmdb = None
        self._base:SegBase = None
        self._prepare_anno:SegPrepareAnno = None
    
    def add_anno(self, anno:SegAnno):
        self._anno = anno
    
    def add_lmdb(self, lmdb_class:SegLmdb):
        self._lmdb = lmdb_class
        
    def add_base(self, base:SegBase):
        self._base = base
        
    def add_prepare_anno(self, prepare_data:SegPrepareAnno):
        self._prepare_anno = prepare_data
        
class ClipGen:
    """主要用于node_gen_segments节点的重要数据统计,数据多是clip的属性"""
    def __init__(self) -> None:
        self.initialize()
    
    def initialize(self):
        self.__seg_nums_of_clip = 0
        self.__distance_of_clip = 0.0
        self.__time_interval_of_clip = 0.0
        self.__image_num_lost_of_clip = 0
        self.__seg_data = OrderedDict()
        self.__has_seg_flag = True
    
    def add_seg(self, seg:Seg):
        self.__seg_nums_of_clip += 1
        self.__distance_of_clip += seg._base.get_distance()
        self.__time_interval_of_clip += seg._base.get_time_interval()
        self.__image_num_lost_of_clip += seg._base.get_num_of_image_lost()
        self.__seg_data[seg.get_name()] = seg._base.get_variables_dict()
    
    def set_has_seg_flag(self, flag:bool):
        self.__has_seg_flag = flag
    
    def get_has_seg_flag(self):
        return self.__has_seg_flag
    
    def get_seg_nums_of_clip(self):
        return self.__seg_nums_of_clip
    
    def get_distance_of_clip(self):
        return self.__distance_of_clip

    def get_time_interval_of_clip(self):
        return self.__time_interval_of_clip
    
    def get_image_num_lost_of_clip(self):
        return self.__image_num_lost_of_clip
    
    def get_variables_dict(self) -> dict:
        variables_dict = {
            "seg_nums_of_clip": self.__seg_nums_of_clip,
            "distance_of_clip": self.__distance_of_clip,
            "time_interval_of_clip": self.__time_interval_of_clip,
            "image_num_lost_of_clip": self.__image_num_lost_of_clip,
            "seg_data": self.__seg_data,
        }
        return variables_dict
    
class Clip(BaseClass):
    def __init__(self) -> None:
        super().__init__()
        self.initialize()
    
    def initialize(self):
        self._clip_match:ClipMatch = None
        self._clip_gen:ClipGen = None
    
    def add_clip_match(self, clip_match:ClipMatch):
        self._clip_match = clip_match
    
    def add_clip_gen(self, clip_gen:ClipGen):
        self._clip_gen = clip_gen
    
@singleton        
class Car(BaseClass):
    def __init__(self) -> None:
        super().__init__()
        self.__datadate = None
        self.exectime = Exectime()
        self.__clip_match_data = OrderedDict()
        self.__clip_gen_data = OrderedDict()
        self.__seg_anno = OrderedDict()
        self.__seg_lmdb = OrderedDict()
        self.__seg_prepare_anno = OrderedDict()
    
    def add_clip_match_data(self, clip:Clip):
        if clip.get_name() not in self.__clip_match_data.keys():
            self.__clip_match_data[clip.get_name()] = OrderedDict()
        self.__clip_match_data[clip.get_name()][
            clip._clip_match.get_match_type()
        ] = clip._clip_match.get_variables_dict()
    
    def add_clip_gen_data(self, clip:Clip):
        if "clip_with_no_seg" not in self.__clip_gen_data.keys():
            self.__clip_gen_data["clip_with_no_seg"] = list()
        
        if not clip._clip_gen.get_has_seg_flag():
            self.__clip_gen_data["clip_with_no_seg"].append(clip.get_name())
        
        if "num_of_clips_all" not in self.__clip_gen_data.keys():
            self.__clip_gen_data["num_of_clips_all"] = 0
        self.__clip_gen_data["num_of_clips_all"] += 1
        
        if "num_of_segs_all" not in self.__clip_gen_data.keys():
            self.__clip_gen_data["num_of_segs_all"] = 0
        self.__clip_gen_data["num_of_segs_all"] += (
            clip._clip_gen.get_seg_nums_of_clip()
        )
        
        if "distance_all" not in self.__clip_gen_data.keys():
            self.__clip_gen_data["distance_all"] = 0.0
        self.__clip_gen_data["distance_all"] += (
            clip._clip_gen.get_distance_of_clip()
        )
        
        if "time_interval_all" not in self.__clip_gen_data.keys():
            self.__clip_gen_data["time_interval_all"] = 0.0
        self.__clip_gen_data["time_interval_all"] += (
            clip._clip_gen.get_time_interval_of_clip()
        )
        
        if "image_frame_lost_all" not in self.__clip_gen_data.keys():
            self.__clip_gen_data["image_frame_lost_all"] = 0
        self.__clip_gen_data["image_frame_lost_all"] += (
            clip._clip_gen.get_image_num_lost_of_clip()           
        )
        
        if "clip" not in self.__clip_gen_data.keys():
            self.__clip_gen_data["clip"] = OrderedDict()
            
        self.__clip_gen_data["clip"][clip.get_name()] = (
            clip._clip_gen.get_variables_dict()
        )
    
    def add_seg_anno(self, seg:Seg):
        if "num_of_segs_all" not in self.__seg_anno.keys():
            self.__seg_anno["num_of_segs_all"] = 0
        self.__seg_anno["num_of_segs_all"] += 1
        
        if "num_of_bbox_all" not in self.__seg_anno.keys():
            self.__seg_anno["num_of_bbox_all"] = 0
        self.__seg_anno["num_of_bbox_all"] += seg._anno.get_num_of_bbox()
        
        if "num_of_key_frams_all" not in self.__seg_anno.keys():
            self.__seg_anno["num_of_key_frams_all"] = 0
        self.__seg_anno["num_of_key_frams_all"] += (
            seg._anno.get_num_of_key_frame()
        )
        
        if "class_and_num_of_obstacle_all" not in self.__seg_anno.keys():
            self.__seg_anno["class_and_num_of_obstacle_all"] = OrderedDict()
        for key in seg._anno.get_class_and_num_of_obstacle().keys():
            # 判断语句太长了,这里把.keys()去掉了, 效果是一样的
            if key not in self.__seg_anno["class_and_num_of_obstacle_all"]:
                self.__seg_anno["class_and_num_of_obstacle_all"][key] = 0
            self.__seg_anno["class_and_num_of_obstacle_all"][key] += (
                seg._anno.get_class_and_num_of_obstacle()[key]
            )

        if "type_and_num_of_lane_all" not in self.__seg_anno.keys():
            self.__seg_anno["type_and_num_of_lane_all"] = OrderedDict()
        for key in seg._anno.get_type_and_num_of_lane().keys():
            if key not in self.__seg_anno["type_and_num_of_lane_all"].keys():
                self.__seg_anno["type_and_num_of_lane_all"][key] = 0
            self.__seg_anno["type_and_num_of_lane_all"][key] += (
                seg._anno.get_type_and_num_of_lane()[key]
            )

        if "seg" not in self.__seg_anno.keys():
            self.__seg_anno["seg"] = OrderedDict()
        self.__seg_anno["seg"][seg.get_name()] = seg._anno.get_variables_dict()
        
        wandao = "{}-curvate_radius_statis".format(self.get_name())
        if wandao not in self.__seg_anno.keys():
            self.__seg_anno[wandao] = OrderedDict()
            self.__seg_anno[wandao]["r<25m"] = 0
            self.__seg_anno[wandao]["25<=r<50"]= 0
            self.__seg_anno[wandao]["50<=r<100"]= 0
            self.__seg_anno[wandao]["100<=r<150"]= 0
            self.__seg_anno[wandao]["150<=r<200"]= 0
            self.__seg_anno[wandao]["200<=r<500"]= 0
            self.__seg_anno[wandao]["r>=500"]= 0
        radius = seg._anno.get_curvature_radius()
        if radius < 25: self.__seg_anno[wandao]["r<25m"] += 1
        if 25 <= radius < 50: self.__seg_anno[wandao]["25<=r<50"] += 1
        if 50 <= radius < 100: self.__seg_anno[wandao]["50<=r<100"] += 1
        if 100 <= radius < 150: self.__seg_anno[wandao]["100<=r<150"] += 1
        if 150 <= radius < 200: self.__seg_anno[wandao]["150<=r<200"] += 1
        if 200 <= radius < 500: self.__seg_anno[wandao]["200<=r<500"] += 1
        if radius >= 500: self.__seg_anno[wandao]["r>=500"] += 1

    def add_seg_lmdb(self, seg:Seg):
        if "size_of_lmdb_all" not in self.__seg_lmdb.keys():
            self.__seg_lmdb["size_of_lmdb_all"] = 0
        self.__seg_lmdb["size_of_lmdb_all"] += int(seg._lmdb.get_lmdb_size())

        if "total_count_all" not in self.__seg_lmdb.keys():
            self.__seg_lmdb["total_count_all"] = 0
        self.__seg_lmdb["total_count_all"] += int(seg._lmdb.get_total_count())

        if "frame_count_all" not in self.__seg_lmdb.keys():
            self.__seg_lmdb["frame_count_all"] = OrderedDict()
        for key in seg._lmdb.get_frame_count().keys():
            if key not in self.__seg_lmdb["frame_count_all"].keys():
                self.__seg_lmdb["frame_count_all"][key] = 0
            self.__seg_lmdb["frame_count_all"][key] += (
                int(seg._lmdb.get_frame_count()[key])
            )
        # print(self.__seg_lmdb)
        if "seg" not in self.__seg_lmdb.keys():
            self.__seg_lmdb["seg"] = OrderedDict()
        # self.__seg_anno["seg"][seg.get_name()] = seg.get_variables_dict()
        self.__seg_lmdb["seg"][seg.get_name()] = seg._lmdb.get_variables_dict()   
        
    def add_seg_prepare_anno(self, seg:Seg):
        if "seg_num_all" not in self.__seg_prepare_anno.keys():
            self.__seg_prepare_anno["seg_num_all"] = 0
        self.__seg_prepare_anno["seg_num_all"] += 1
        
        if "test_all" not in self.__seg_prepare_anno.keys():
            self.__seg_prepare_anno["test_all"] = 0
        if seg._prepare_anno.get_obstacle_data_attribute() == "test":
            self.__seg_prepare_anno["test_all"] += 1

        if "train_all" not in self.__seg_prepare_anno.keys():
            self.__seg_prepare_anno["train_all"] = 0
        if seg._prepare_anno.get_obstacle_data_attribute() == "test":
            self.__seg_prepare_anno["train_all"] += 1
            
        if "not_train_or_test" not in self.__seg_prepare_anno.keys():
            self.__seg_prepare_anno["not_train_or_test"] = list()
        if seg._prepare_anno.get_obstacle_data_attribute() == None:
            self.__seg_prepare_anno["not_train_or_test"].append(
                seg.get_name()
            )
        if "segs" not in self.__seg_prepare_anno.keys():
            self.__seg_prepare_anno["segs"]= OrderedDict()
        self.__seg_prepare_anno["segs"][seg.get_name()] = seg._prepare_anno.get_variables_dict()
        
    def set_datadate(self, date:str):
        self.__datadate = date
    
    def get_datadate(self):
        return self.__datadate
    
    def get_clip_match_data(self):
        return self.__clip_match_data
    
    def get_clip_gen_data(self):
        return self.__clip_gen_data
    
    def get_seg_anno(self):
        return self.__seg_anno

    def get_seg_lmdb(self):
        return self.__seg_lmdb
    
    def get_seg_prepare_anno(self):
        return self.__seg_prepare_anno
    
    
class DataPool:
    def __init__(self, car:Car, config_file:str, node_name:str) -> None:
        self.__car = car
        self.__config_file = config_file
        self.__work_tmp_dir = os.path.dirname(config_file)
        self.__node_name = node_name
        self.__initialize()
        
    def __initialize(self):
        self.__data = OrderedDict()
        # self.__target_path = os.path.join(
        #     os.path.dirname(os.path.abspath(__file__)),
        #     "../../log", 
        #     f"{self.__car.get_name()}_{self.__car.get_datadate()}")
        self.__target_path = os.path.join(self.__work_tmp_dir, "data_board")
        self.__target_file = os.path.join(self.__target_path, TARGET_FILE_NAME)
    
    def run(self):
        self.__create_target_path()
        # self.__backup_config_file()
        self.__parse_information()
        self.__initialize_target_file()
        self.__update_target_file()
        self.__initialize()
    
    def __create_target_path(self):
        if not os.path.exists(self.__target_path):
            os.makedirs(self.__target_path)
    
    def __backup_config_file(self):
        source = os.path.abspath(self.__config_file)
        destination = self.__target_path
        shutil.copy2(source, destination)
    
    def __parse_information(self):
        self.__data[self.__node_name] = OrderedDict()
        self.__data[self.__node_name]["car_name"] = self.__car.get_name()
        self.__data[self.__node_name]["datadate"] = self.__car.get_datadate()
        self.__data[self.__node_name]["time_cost"] = (
            self.__car.exectime.get_time_cost()
        )
        if self.__node_name == "node_match_frame":
            self.__data[self.__node_name]["clip"] = (
                self.__car.get_clip_match_data()
            )
        if self.__node_name == "node_pack_annotation":
            for key in self.__car.get_seg_anno():
                self.__data[self.__node_name][key] = (
                    self.__car.get_seg_anno()[key]
                )
        if self.__node_name == "node_lmdb_pack":
            for key in self.__car.get_seg_lmdb():
                self.__data[self.__node_name][key] = (
                    self.__car.get_seg_lmdb()[key]
                )
        if self.__node_name == "node_gen_segments":
            for key in self.__car.get_clip_gen_data():
                self.__data[self.__node_name][key] = (
                    self.__car.get_clip_gen_data()[key]
                )
        if self.__node_name == "node_prepare_anno_data":
            for key in self.__car.get_seg_prepare_anno():
                self.__data[self.__node_name][key] = (
                    self.__car.get_seg_prepare_anno()[key]
                )
            
    def __initialize_target_file(self):
        if not os.path.exists(self.__target_file):
            with open(self.__target_file, 'w') as file:
                file.write('{}')
            print("%s".format(FileExistsError))
            
        if not os.path.getsize(self.__target_file):
            with open(self.__target_file, "w") as file:
                file.write("{}")
    
    def __update_target_file(self):
        with open(self.__target_file, "r" ,encoding="utf-8") as json_file:
            file_content = json_file.read()
            if not file_content:
                file_content = OrderedDict()
            else:
                file_content = json.loads(file_content)
        file_content[self.__node_name] = self.__data[self.__node_name]
        file_content = json.dumps(file_content, indent=4)
        # file_content = json.dumps(file_content, indent=4, default=ordered_dict_serializer)
        with open(self.__target_file, "w", encoding="utf-8") as json_file:
            json_file.write(file_content)
        print("Databoard of {} saved in file: {}".format(self.__node_name, self.__target_path))
            
        
    
