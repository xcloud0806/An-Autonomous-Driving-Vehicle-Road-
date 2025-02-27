import sys, os
import json
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from collections import OrderedDict
from python import match_frame_fdc, match_frame
from utils import CarMeta
import shutil
from utils import Seg, Clip, Car, DataPool
data_car = Car()
RECORD_JSON = 'record.json'
TAG_INFO_JSON = 'tag_info.json'
FRAME_LOST_LIMIT = 50
LIDAR_LOST_LIMIT = 2
HPP_CAR_LIST = [
    "sihao_y7862",
    "sihao_7xx65",
    "sihao_19cp2"
]
logging.basicConfig(level=logging.DEBUG,  # 设置日志级别为DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 日志格式
# 将handler添加到日志记录器
logger = logging.getLogger(__file__)

def get_sensors(car_meta, cameras, only_check_lidar):
    sensor_list = []
    sensor_list.append(car_meta.lidar_name)
    sensor_list.extend(car_meta.inno_lidars)
    sensor_list.extend(car_meta.bpearl_lidars)
    if not only_check_lidar:
        sensor_list.extend(cameras)
    sensor_to_threshold = {}
    for sensor in sensor_list:
        if "lidar" in sensor:
            sensor_to_threshold[sensor] = (10, 5) # 采样频率10hz，容忍度5ms
        if "around" in sensor:
            sensor_to_threshold[sensor] = (20, 15) # 采样频率20hz，容忍度15ms 
        if "surround" in sensor:
            sensor_to_threshold[sensor] = (20, 10) # TODO:确认阈值
    return sensor_to_threshold
        

def frame_lost_self_check(sensors:dict, clip_path):
    frame_of_sensor_lost = OrderedDict()
    for sensor, (hz, threshold) in sensors.items():
        sensor_data_path = os.path.join(clip_path, sensor)
        if "bpearl" in sensor_data_path and not os.path.exists(sensor_data_path):
            continue
        if not os.path.exists(sensor_data_path):
            return False
        sensor_data = sorted(os.listdir(sensor_data_path))
        period = 1000* 1 / hz       # 时间
        previous_timestamp = None
        for data in sensor_data:
            current_timestamp = int(data.split(".")[0])
            if previous_timestamp != None:
                if abs(current_timestamp - previous_timestamp - period) > threshold:
                    if sensor not in frame_of_sensor_lost:
                        frame_of_sensor_lost[sensor] = 0
                    # import pdb; pdb.set_trace()
                    temp = round((current_timestamp - previous_timestamp + period)/period) -1 # check bug
                    frame_of_sensor_lost[sensor] += temp
            previous_timestamp = current_timestamp

    frame_lost_all = 0
    for key in frame_of_sensor_lost:
        frame_lost_all += frame_of_sensor_lost[key]
    
    if frame_lost_all >= FRAME_LOST_LIMIT:
        logger.warning("frame_lost_all:{}".format(frame_lost_all))
        for key in frame_of_sensor_lost:
            frame_lost_all += frame_of_sensor_lost[key]
            logger.warning("sensor:{}, lost frames:{}".format(key, frame_of_sensor_lost[key]))
        return False
    else:
        if "lidar" in frame_of_sensor_lost:
            lidar_lost_cnt = frame_of_sensor_lost['lidar']
            if lidar_lost_cnt > LIDAR_LOST_LIMIT:
                logger.warning("lidar lost frames:{}".format(lidar_lost_cnt))
                return False       
        return True

@data_car.exectime
def main_match_frames(run_config):
    match_config = run_config['preprocess']
    frame_path = match_config['frames_path']
    calib_path = match_config["calibration_path"]
    seg_mode = match_config["seg_mode"]
    spec_clips = match_config.get("spec_clips", None)

    car_meta = CarMeta()
    try:
        car_meta_file = os.path.join(calib_path, "car_meta.json")
        with open(car_meta_file, 'r', encoding='utf-8') as fp:
            car_meta_dict = json.load(fp)    
            car_meta.from_json_iflytek(car_meta_dict)
    except Exception as e:
        logger.exception(f"Caught an exception of type {type(e).__name__}: {e}")
        sys.exit(1) 

    car = car_meta.car_name
    data_car.set_name(car)
    data_car.set_datadate(os.path.basename(frame_path))
    clip_list = os.listdir(frame_path)
    clip_list = [clip_name for clip_name in clip_list if os.path.isdir(os.path.join(frame_path, clip_name)) and clip_name != "discard"]
    clip_skip_num = 0
    
    for clip in clip_list: 
        if spec_clips is not None and clip not in spec_clips:
            continue
        logger.info("Dealing with clip:{}".format(clip))
        clip_path = os.path.abspath(os.path.join(frame_path, clip))
        cameras = []
        for cam in car_meta.cameras:
            image_target_path = os.path.join(clip_path, cam)
            if os.path.exists(image_target_path):
                cameras.append(cam)
        
        # try to fix ofilm_camera_front_120_8M
        camera_front_8M = os.path.join(clip_path, "ofilm_camera_front_120_8M")
        if os.path.exists(camera_front_8M) and "ofilm_surround_front_120_8M" in car_meta.cameras:
            surr_front_8M =  os.path.join(clip_path, "ofilm_surround_front_120_8M")
            if not os.path.exists(surr_front_8M):
                os.rename(camera_front_8M, surr_front_8M)
                cameras.append("ofilm_surround_front_120_8M")
        
        only_check_lidar = False
        if seg_mode in ["time", "luce", "hpp_luce", "aeb"]:
            only_check_lidar = True
        sensors_to_offset = get_sensors(car_meta, cameras, only_check_lidar)
        result = frame_lost_self_check(sensors_to_offset, clip_path) # 各传感器在容忍度范围内 检查是否丢帧
        if not result:
            clip_skip_num += 1
            logger.warning("Skip current clip: {}".format(clip))
            continue

        try:
            if car_meta.dc_system_version == 'iflytek_expo':
                match_frame_fdc(clip_path, cameras, calib_path, seg_mode)
            else:
                match_frame(car, clip_path, "", clip_path, car_meta)
        except Exception as e:
            logger.exception(f"Caught an exception of type {type(e).__name__}: {e}")
            sys.exit(1)
            
        record_path = os.path.join(clip_path, TAG_INFO_JSON)
        record_clip_outpath = os.path.join(clip_path, RECORD_JSON)
        if os.path.exists(record_path) and not os.path.exists(record_clip_outpath):            
            shutil.copy(record_path, record_clip_outpath)
    
    total_clip_num = len(clip_list) if spec_clips is None else len(spec_clips)
    clip_skip_rate = clip_skip_num / total_clip_num
    if clip_skip_rate > 0.1:
        logger.error(
            f"clip skip num:{clip_skip_num}, "
            f"all clip num:{total_clip_num}, "
            f"clip skip rate:{clip_skip_rate:.3f}, more than 10%.\n"
            f"program terminated!"
        )
        if seg_mode != "luce" and seg_mode != "hpp_luce" and seg_mode != "aeb":
            return False
        elif clip_skip_rate > 0.5:
            # in luce mode, raise fatal error when clip skip rate > 50%
            return False
    else:
        logger.info(
            f"clip skip num:{clip_skip_num}, "
            f"all clip num:{total_clip_num}, "
            f"clip skip rate:{clip_skip_rate:.3f}, less than 10%.\n"
            f"program go on!"
        )
    return True

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)
    
    with open(config_file, 'r') as fp:
        run_config = json.load(fp)
    
    work_tmp_dir = os.path.dirname(config_file)
    handler = RotatingFileHandler(os.path.join(work_tmp_dir, "node_match_frame.log"), maxBytes=20*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    ret = main_match_frames(run_config)
    node_name = os.path.splitext(os.path.basename(__file__))[0]
    datapool = DataPool(data_car, config_file, node_name)
    datapool.run()
    if not ret:
        sys.exit(1)

    

