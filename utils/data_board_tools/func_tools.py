import os
import json
import numpy as np
from .datapool import ClipMatch, ClipGen, Clip, SegBase, SegPrepareAnno, SegAnno,Seg, Car
from typing import Optional
data_car = Car()

MAX_LOST_LIMIT = 2
DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]


def fill_seg_anno(seg_name:str, annotation, lane_flag:bool, obstacle_flag:bool):
    anno = SegAnno()
    anno.parse_annotation(annotation)
    anno.set_annotation_lane_attribute(lane_flag)
    anno.set_annotation_obstacle_attribute(obstacle_flag)
    seg = Seg()
    seg.set_name(seg_name)
    seg.add_anno(anno)
    data_car.add_seg_anno(seg)
    anno.initialize()
    seg.initialize()

def fill_clip_match(
    clip_name:str,
    match_type:str,
    num_of_lidar_frame:int = 0,
    num_of_lidar_frame_lost:int = 0,
    num_of_image_frame_lost:int = 0,
    num_of_bpearl_lidars_frame_lost:Optional[int] = None,
    num_of_inno_lidars_frames_lost:Optional[int] = None,
    camera_lost:list = [],
    inno_lidar_lost:Optional[list] = None,
    bpearl_lidar_lost:Optional[list] = None,
)-> None:
    """统计match_frames过程中的关键指标"""
    clip_match = ClipMatch()
    clip_match.set_match_type(match_type)
    clip_match.set_num_of_lidar_frame(num_of_lidar_frame)
    clip_match.set_num_of_bpearl_lidars_frame_lost(
        num_of_bpearl_lidars_frame_lost
    )
    clip_match.set_num_of_inno_lidars_frames_lost(
        num_of_inno_lidars_frames_lost
    )
    clip_match.set_num_of_lidar_frame_lost(num_of_lidar_frame_lost)
    clip_match.set_num_of_image_frame_lost(num_of_image_frame_lost)
    clip_match.add_camera_lost(camera_lost)
    clip_match.add_inno_lidar_lost(inno_lidar_lost)
    clip_match.add_bpearl_lidar_lost(bpearl_lidar_lost)
    
    clip = Clip()
    clip.set_name(clip_name)
    clip.add_clip_match(clip_match)
    data_car.add_clip_match_data(clip)
    clip.initialize()
    clip_match.initialize()

def fill_clip_gen(
    clip_name:str,
    segs:Optional[list],
)-> None:
    """针对每个clip提取多个seg的数据"""
    clip_gen = ClipGen()
    if len(segs) == 0:
        clip_gen.set_has_seg_flag(False)
    else:
        for seg_data in segs:
            distance = float(seg_data["distance"])
            time_start = int(seg_data["frames"][0]["lidar"]["timestamp"])
            time_interval = float(seg_data["time_interval"])
            lost_image_num = int(seg_data["lost_image_num"])
            seg_uid = seg_data["seg_uid"]
            
            seg_base = SegBase()
            seg_base.set_distance(distance)
            seg_base.set_time_start(time_start)
            seg_base.set_time_interval(time_interval)
            seg_base.set_num_of_image_lost(lost_image_num)
            
            seg = Seg()
            seg.add_base(seg_base)
            seg.set_name(seg_uid)
            clip_gen.add_seg(seg)
            
            seg_base.initialize()
            seg.initialize()
    clip = Clip()
    clip.add_clip_gen(clip_gen)
    clip.set_name(clip_name)
    data_car.add_clip_gen_data(clip=clip)
    clip.initialize()

def fill_seg_prepare_anno(
    seg_root_path,
    skip_reconstruct,
    seg_anno_dst,
):
    seg_names = os.listdir(seg_root_path)
    for i, _seg in enumerate(seg_names):
        def get_key_info():
            exist_loadable_meta = None
            pose_normality = None
            reconstruct_ok = None
            frames_lost_to_limit = None
            obstacle_data_ok = None
            lane_data_ok = None
            obstacle_data_attribute=None
            
            # 1. 更新exist_loadable_meta
            seg_dir = os.path.join(seg_root_path, _seg)
            seg_meta_json = os.path.join(seg_dir, "meta.json")
            
            if not os.path.exists(seg_meta_json):
                exist_loadable_meta = False
                return (
                    exist_loadable_meta,
                    pose_normality,
                    reconstruct_ok,
                    frames_lost_to_limit,
                    obstacle_data_ok,
                    lane_data_ok,
                    obstacle_data_attribute,
                    )
            try:
                meta = json.load(open(seg_meta_json, "r"))
                exist_loadable_meta = True
            except:
                exist_loadable_meta = False
                return (
                    exist_loadable_meta,
                    pose_normality,
                    reconstruct_ok,
                    frames_lost_to_limit,
                    obstacle_data_ok,
                    lane_data_ok,
                    obstacle_data_attribute,
                    )
            # 2. 更新 pose_normality
            first_lidar_pose = np.array(meta['frames'][0]['lidar']['pose']).astype(np.float32)
            dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
            if (first_lidar_pose==dft_pose_matrix).all():
                pose_normality = False
                return (
                    exist_loadable_meta,
                    pose_normality,
                    reconstruct_ok,
                    frames_lost_to_limit,
                    obstacle_data_ok,
                    lane_data_ok,
                    obstacle_data_attribute,
                    
                )
            else:
                pose_normality = True
                
            # 3. 在函数外部更新了skip_resconstruct   
            # 4. 更新 reconstruct_ok
            reconstruct_path = os.path.join(seg_dir, "reconstruct")
            if not os.path.exists(reconstruct_path) and not skip_reconstruct:
                reconstruct_ok = False
                return (
                    exist_loadable_meta,
                    pose_normality,
                    reconstruct_ok,
                    frames_lost_to_limit,
                    obstacle_data_ok,
                    lane_data_ok,
                    obstacle_data_attribute,
                    )
            
            # 5. 更新 frames_lost_to_limit
            enable_cameras = meta['cameras']
            calib = meta['calibration']
            calib_sensors = calib['sensors']
            cameras = [item for item in enable_cameras if item in calib_sensors ]
            max_lost_limit = len(cameras) * MAX_LOST_LIMIT
            if meta['lost_image_num'] > (max_lost_limit):
                frames_lost_to_limit = True
                print("333333333333333")
                return (
                exist_loadable_meta,
                pose_normality,
                reconstruct_ok,
                frames_lost_to_limit,
                obstacle_data_ok,
                lane_data_ok,
                obstacle_data_attribute,
                )
            else:
                frames_lost_to_limit = False
            
            # 6. 更新 obstacle_data_ok
            # 7. 更新 lane_data_ok
            meta = json.load(seg_meta_json)
            lane_dst_path = seg_anno_dst[_seg]['lane']
            obs_dst_path = seg_anno_dst[_seg]['obs']
            # 8.更新obstacle_data_attribute
            if _seg in seg_anno_dst:
                obstacle_data_attribute = seg_anno_dst[_seg]['mode']
            key_lane_file = os.path.join(lane_dst_path, "transform_matrix.json")
            key_obstacle_file = obs_dst_path, "{}_infos.json".format(_seg)
            if skip_reconstruct:
                if not os.path.exists(key_obstacle_file):
                    obstacle_data_ok = False
                else:
                    with open(key_obstacle_file, "r") as info_fp:
                        info_json_dict = json.load(info_fp)
                    if len(info_json_dict['frames']) < 5:
                        obstacle_data_ok = False
                    obstacle_data_ok = True
            else:
                if not os.path.exists(key_lane_file):
                    lane_data_ok = False
                else:
                    lane_data_ok = True
                
                if not os.path.exists(
                    os.path.join(key_obstacle_file)
                ):
                    obstacle_data_ok = False
                else:
                    with open(key_obstacle_file, "r") as info_fp:
                        info_json_dict = json.load(info_fp)
                    if len(info_json_dict['frames']) < 5:
                        obstacle_data_ok = False
                    obstacle_data_ok = True
            print("55555555555555555")
            return (
                exist_loadable_meta,
                pose_normality,
                reconstruct_ok,
                frames_lost_to_limit,
                obstacle_data_ok,
                lane_data_ok,
                obstacle_data_attribute,
            )
        (
            exist_loadable_meta,
            pose_normality,
            reconstruct_ok,
            frames_lost_to_limit,
            obstacle_data_ok,
            lane_data_ok,
            obstacle_data_attribute
        ) = get_key_info()
        seg_prepare_anno = SegPrepareAnno()
        seg_prepare_anno.set_exist_loadable_meta(exist_loadable_meta)
        seg_prepare_anno.set_pose_normality(pose_normality)
        seg_prepare_anno.set_skip_reconstruct(skip_reconstruct)
        seg_prepare_anno.set_reconstruct_ok(reconstruct_ok)
        seg_prepare_anno.set_frames_lost_to_limit(frames_lost_to_limit)
        seg_prepare_anno.set_obstacle_data_ok(obstacle_data_ok)
        seg_prepare_anno.set_lane_data_ok(lane_data_ok)
        seg_prepare_anno.set_obstacle_data_attribute(obstacle_data_attribute)
        print("obstacle_data_attribute:", obstacle_data_attribute)
        seg = Seg()
        seg.set_name(_seg)
        seg.add_prepare_anno(seg_prepare_anno)
        data_car.add_seg_prepare_anno(seg=seg)
        seg.initialize()
        seg_prepare_anno.initialize()


