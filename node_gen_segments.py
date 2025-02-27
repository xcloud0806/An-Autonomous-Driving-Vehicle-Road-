from python import handle_ifly_frame
from utils import combine_calib, CarMeta

import numpy as np
import json
from datetime import datetime
import traceback as tb
import os,sys
from loguru import logger
import copy
from collections import Counter  

RECORD_FILE = "record.json"
META_FILE = "car_meta.json"
MAX_FRAMES = 80
PICK_INTERVAL = 5 # 10 * 0.5
MDC_CAR = ["chery_10034", "chery_04228", "chery_18047", "chery_18049", "chery_48160", "chery_14520"]
def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def gen_clip_segs(source, segment, car_name, calib_path, car_meta:CarMeta, seg_mode="distance", seg_val=None):
    try:
        calib_info = combine_calib(calib_path)
        seg_infos = []
        print(f"......\t{source} cut seg start... {str(datetime.now())}")
        
        if 'iflytek' in car_meta.dc_system_version :
            if seg_mode == "distance":
                # 基于距离的重建最大不超过700M，也不能超过5分钟
                seg_distance = 700
                if seg_val is not None:
                    seg_distance = seg_val
                seg_infos = handle_ifly_frame(source, segment, car_name, distance=seg_distance, time_interval=300)
            elif seg_mode == "time":
                time_interval = 120
                if seg_val is not None:
                    time_interval = seg_val
                seg_infos = handle_ifly_frame(source, segment, car_name, distance=0, time_interval=time_interval)
            elif seg_mode == "motorway": 
                seg_distance = 1000
                seg_infos = handle_ifly_frame(source, segment, car_name, distance=seg_distance, time_interval=60)
            elif seg_mode == "city" or seg_mode == "common":
                seg_distance = 700
                seg_infos = handle_ifly_frame(source, segment, car_name, distance=seg_distance, time_interval=120)
            elif seg_mode == "hpp" or seg_mode == "hpp_luce":
                seg_distance = 70
                if seg_val is not None:
                    seg_distance = seg_val
                # 这里改动是从2024年8月12日开始，采集数据为司机定制采集，不再通过距离分段
                seg_infos = handle_ifly_frame(source, segment, car_name, distance=0, time_interval=0, seg_mode=seg_mode)
            elif seg_mode == "none" or seg_mode == 'null' \
                or seg_mode == 'trigger' or seg_mode == 'luce' \
                or seg_mode == 'test' or seg_mode == 'aeb'  \
                or seg_mode == 'traffic_light':
                seg_infos = handle_ifly_frame(source, segment, car_name, distance=0, time_interval=0, seg_mode=seg_mode)
    except Exception as e:
        logger.exception(f"Try cut {source} segment failed. as exeption {e}")
        tb.print_exc()
        sys.exit(1)
        return []

    print(f"......\t{source} cut seg end... {str(datetime.now())}")
    print("\tcut to {} segs.".format(len(seg_infos)))
    if len(seg_infos) == 0:
        return []

    record_file = os.path.join(source, RECORD_FILE)
    record = None
    try:
        if os.path.exists(record_file):
            fp = open(record_file, "r")
            record = json.load(fp)

        if not os.path.exists(segment):
            os.umask(0o002)
            os.makedirs(segment, mode=0o775, exist_ok=True)
    except Exception as e:
        logger.exception(f"Caught an exception of type {type(e).__name__}: {e}")
        sys.exit(1)

    clip_segs = []
    print(f"......\t{source} gen seg info start... {str(datetime.now())}")
    for seg_idx, seg_info in enumerate(seg_infos):
        seg, gnss, vech = seg_info
        seg['calibration'] = calib_info
        seg['record'] = record
        seg['data_tags'] = []
        seg['data_system'] = car_meta.dc_system_version
        seg['car'] = car_meta.car_name
        seg['other_sensors_info'] = car_meta.other_sensors_info
        seg['bpearl_lidars'] = car_meta.bpearl_lidars
        seg['inno_lidars'] = car_meta.inno_lidars
        seg['vision_slot'] = car_meta.vision_slot
        seg['vision_slot_interval'] = car_meta.vision_slot_timeinterval

        seg_id = seg['seg_uid']
        try:
            os.makedirs(os.path.join(segment, seg_id), exist_ok=True)   

            vech_json = os.path.join(os.path.join(segment, seg_id, "vehicle.json"))
            with open(vech_json, "w") as fp:
                ss = json.dumps(vech, ensure_ascii=False, default=dump_numpy)
                fp.write(ss)

            gnss_json = os.path.join(os.path.join(segment, seg_id, "gnss.json"))
            with open(gnss_json, "w") as fp:
                ss = json.dumps(gnss, ensure_ascii=False, default=dump_numpy)
                fp.write(ss)

            pre_anno_json = os.path.join(os.path.join(segment, seg_id, "pre_anno.json"))
            with open(pre_anno_json, "w") as fp:
                pre_anno = {
                    "point_cloud":[]
                }
                ss = json.dumps(pre_anno, ensure_ascii=False, default=dump_numpy)
                fp.write(ss)
        except Exception as e:
            logger.exception(f"Caught an exception of type {type(e).__name__}: {e}")
            sys.exit(1)
                    
        clip_segs.append(seg)
    print(f"......\t{segment} gen seg info end... {str(datetime.now())}")
    return clip_segs

def main_gen_segs(config:dict):
    seg_config = config['preprocess']
    frames_path = seg_config["frames_path"]
    tgt_seg_path = seg_config["segment_path"]
    calib_path = seg_config['calibration_path']
    seg_mode = seg_config['seg_mode']
    seg_value = seg_config['seg_value']
    car_name = seg_config['car']
    spec_clips = seg_config.get("spec_clips", None)

    car_meta = CarMeta()
    try:
        car_meta_file = os.path.join(calib_path, "car_meta.json")
        with open(car_meta_file, 'r') as fp:
            car_meta_dict = json.load(fp)    
            car_meta.from_json_iflytek(car_meta_dict)
    except Exception as e:
            logger.exception(f"Caught an exception of type {type(e).__name__}: {e}")
            sys.exit(1)

    clip_segs = []
    clips = os.listdir(frames_path)
    clips.sort()

    avail_seg_distance = 0.0
    avail_seg_time = 0.0
    avail_seg_frame = 0
    segs = []

    for clip in clips:
        if spec_clips is not None and clip not in spec_clips:
            continue
        if not clip.startswith("202"):
            continue
        print("Start cut seg {}......".format(clip))
        clip_frame = os.path.join(frames_path, clip)
        curr_clip_segs = gen_clip_segs(clip_frame, tgt_seg_path, car_name, calib_path, car_meta, seg_mode, seg_value)
        clip_segs.extend(curr_clip_segs)    

        for seg in curr_clip_segs:
            seg_id = seg['seg_uid']
            hz_10 = 0
            hz_20 = 0
            for camera in seg['cameras']:
                camera_path = os.path.join(seg['frames_path'], camera)
                if not os.path.exists(camera_path):
                    continue
                imgs_list = os.listdir(camera_path)
                img_ts_list = [filename.split('.')[0] if '.' in filename else filename for filename in imgs_list] 
                img_ts_list = sorted(map(int, img_ts_list))
                ts_diffs = [img_ts_list[i + 1] - img_ts_list[i] for i in range(len(img_ts_list) - 1)]
                # 统计每种差值的数量
                ts_diffs_count = Counter(ts_diffs)
                # 获取统计数量最多的差值  
                most_common_difference = ts_diffs_count.most_common(1)
                if most_common_difference[0][0] == 100: # 此时相机的数据是10HZ
                    hz_10 += 1
                elif most_common_difference[0][0] == 50: # 此时相机的数据是20HZ
                    hz_20 += 1
            if hz_10 > hz_20:
                seg['frames'] = seg['raws'] 
                seg['lost_image_num'] = seg['raw_pair_lost_image_num']
            segs.append(seg_id)
            avail_seg_distance += seg['distance']
            avail_seg_time += seg['time_interval']
            avail_seg_frame += len(seg['frames'])
            try:
                meta_json = os.path.join(os.path.join(tgt_seg_path, seg_id, "meta.json"))     
                meta_ss = json.dumps(seg, ensure_ascii=False, default=dump_numpy)
                with open(meta_json, "w") as fp:
                    fp.write(meta_ss)
            except Exception as e:
                logger.exception(f"Caught an exception of type {type(e).__name__}: {e}")
                sys.exit(1)

    logger.info(f"......\t{frames_path} total {len(clips)} clips cut segs to {len(clip_segs)} segments end...")
    logger.info(f"\t...total clips' time is {avail_seg_time}")
    logger.info(f"\t...total clips' distance is {avail_seg_distance}")
    logger.info(f"\t...total clips' frame is {avail_seg_frame}")
    logger.info(f"\t...total segments id is {segs}")
    return clip_segs

if __name__ ==  "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)
    
    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "node_gen_segments.log"))

    with open(config_file, 'r') as fp:
        run_config = json.load(fp)
    clip_segs = main_gen_segs(run_config)
    seg_config = run_config['preprocess']
    spec_clips = seg_config.get("spec_clips", None)
    if spec_clips is not None:
        dump_config = copy.deepcopy(run_config)
        spec_segids = []
        for seg in clip_segs:
            spec_segids.append(seg['seg_uid'])
        dump_config['preprocess']['spec_segments'] = spec_segids

        with open(config_file, "w") as wfp:
            ss = json.dumps(dump_config, ensure_ascii=False, default=dump_numpy, indent=4)
            wfp.write(ss)
