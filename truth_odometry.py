import sys
import os
import time

curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_path)
import sys
from loguru import logger

sys.path.append(f"{curr_dir}/lib/python3.8/site_packages")
import odometry

import json
import math
import numpy as np
from multiprocessing.pool import Pool
from multiprocessing import Event
import multiprocessing as mp

DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

def error_callback(e, error_event):  
    # 捕获子进程中的异常并设置事件  
    logger.error(f"Error occurred: {e}")
    error_event.set()  # 通知主进程有异常发生  

def dump_numpy_nan(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and math.isnan(obj):
        return "NaN"
    elif obj is None:
        return "null"
    else:
        return obj

def convert_nan_to_string(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_nan_to_string(value)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = convert_nan_to_string(data[i])
    elif isinstance(data, float) and np.isnan(data):
        data = 'NaN'
    return data

def judge_odometry_failed(pair_pose_list, gnss_info_list):
    if isinstance(pair_pose_list, str):
        pair_pose_list = json.loads(pair_pose_list)

    status = False
    pose_list = []
    time_list = []
    gnss_speed_list = []
    gnss_status_list = []
    for li, pair in enumerate(pair_pose_list["frames"]):
        if pair["lidar"]["pose"] == DEFAULT_POSE_MATRIX:
            status = True
        pose_list.append(np.array(pair["lidar"]["pose"]))
        time_list.append(pair["lidar"]["timestamp"])
        gnss_speed = gnss_info_list[str(pair["gnss"])]["speed"]
        gnss_status = gnss_info_list[str(pair["gnss"])]["gps_status"]
        gnss_speed_list.append(float(gnss_speed))
        gnss_status_list.append(gnss_status)
    
    speeds_list = []
    speeds_time_list = []
    car_gnss_speed_list = []
    for index in range(1, len(pose_list)):
        pose_inv = np.linalg.pinv(pose_list[index-1])
        rela_pose = pose_inv @ pose_list[index]
        trans = rela_pose[:3, 3]
        trans_norm = np.linalg.norm(trans, ord=2)
        time = (time_list[index] - time_list[index-1]) * 0.001
        car_speed = trans_norm / time
        if (car_speed > 40):
            status = True
        speeds_list.append(car_speed)
        speeds_time_list.append(time_list[index])
        car_gnss_speed_list.append(gnss_speed_list[index])
        speed_err = abs(car_speed - gnss_speed_list[index])
        if (speed_err > 3) and (gnss_status_list[index] == '42'):
            status = True
    
    for index in range(1, len(speeds_list)):
        speed_norm = speeds_list[index] - speeds_list[index-1]
        time = (speeds_time_list[index] - speeds_time_list[index-1]) * 0.001
        car_acc = speed_norm / time
        if (car_acc > 30):
            status = True
    return status

def LidarOdeometryConstruct(seg_folder):
    try:
        meta_json = os.path.join(seg_folder, "meta.json")
        with open(meta_json, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
            meta = convert_nan_to_string(meta)
            meta_json_str = json.dumps(meta, ensure_ascii=False, default=dump_numpy_nan)
        logger.info(f"Curr handle {meta['seg_uid']}")
        meta_ss = odometry.LidarOdeometryConstruct(seg_folder, meta_json_str)
    except Exception as e:
        logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
        raise e        
    dst_json_path = os.path.join(seg_folder, "meta.json")

    if len(meta_ss) == 0:
        logger.warning(f"{seg_folder} Odeometry Failed...")
        # raise Exception(f"{seg_folder} Odeometry Failed...")
    else:
        try:
            gnss_path = os.path.join(seg_folder, "gnss.json")
            gnss_info_list = json.load(open(gnss_path, 'r'))
            if not judge_odometry_failed(meta_ss, gnss_info_list):
                with open(dst_json_path, "w") as fp:
                    fp.write(meta_ss)
            else:
                logger.error(f"{seg_folder} Odeometry Speed Acc Failed...")
        except Exception as e:
            logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
            raise e                

def LidarWheelConstruct(seg_folder):
    try:
        meta_json = os.path.join(seg_folder, "meta.json")
        with open(meta_json, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
            meta = convert_nan_to_string(meta)
            meta_json_str = json.dumps(meta, ensure_ascii=False, default=dump_numpy_nan)
        logger.info(f"Curr handle {meta['seg_uid']}")

        gnss_json = os.path.join(seg_folder, "gnss.json")
        with open(gnss_json, "r", encoding="utf-8") as fp:
            gnss = json.load(fp)
            gnss_json_str = json.dumps(gnss)

        vehicle_json = os.path.join(seg_folder, "vehicle.json")
        with open(vehicle_json, "r", encoding="utf-8") as fp:
            vehicle = json.load(fp)
            vehicle_json_str = json.dumps(vehicle)

        meta_ss = odometry.LidarWheelConstruct(
            seg_folder, meta_json_str, gnss_json_str, vehicle_json_str)
    except Exception as e:
        logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
        raise e        
    dst_json_path = os.path.join(seg_folder, "meta.json")

    if len(meta_ss) == 0:
        logger.warning(f"{seg_folder} Odeometry Failed...")
        # raise ValueError(f"{seg_folder} Odeometry Failed...")
    else:
        try:
            gnss_path = os.path.join(seg_folder, "gnss.json")
            gnss_info_list = json.load(open(gnss_path, 'r'))
            if not judge_odometry_failed(meta_ss, gnss_info_list):
                with open(dst_json_path, "w") as fp:
                    fp.write(meta_ss)
            else:
                logger.error(f"{seg_folder} Odeometry Speed Acc Failed...")
        except Exception as e:
            logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
            raise e

def ParkingOdeometryConstruct(seg_folder):
    meta_json = os.path.join(seg_folder, "meta.json")
    try:
        with open(meta_json, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
            meta = convert_nan_to_string(meta)
            meta_json_str = json.dumps(meta, ensure_ascii=False, default=dump_numpy_nan)
        logger.info(f"Curr handle {meta['seg_uid']}")
        meta_ss = odometry.ParkingOdeometryConstruct(seg_folder, meta_json_str)
    except Exception as e:
        logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
        raise e
        
    dst_json_path = os.path.join(seg_folder, "meta.json")

    if len(meta_ss) == 0:
        logger.warning(f"{seg_folder} Odeometry Failed...")
        # raise ValueError(f"{seg_folder} Odeometry Failed...")
    else:
        try:
            gnss_path = os.path.join(seg_folder, "gnss.json")
            gnss_info_list = json.load(open(gnss_path, 'r'))
            if not judge_odometry_failed(meta_ss, gnss_info_list):
                with open(dst_json_path, "w") as fp:
                    fp.write(meta_ss)
            else:
                logger.error(f"{seg_folder} Odeometry Speed Acc Failed...")
        except Exception as e:
            logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
            raise e
def is_spd_increasing(lst):
    return all(x <= y for x, y in zip(lst, lst[1:]))

def vehicle_data_right(seg_folder):
    status = True
    vehicle_path = os.path.join(seg_folder, "vehicle.json")
    vehicle_info = json.load(open(vehicle_path, 'r'))
    if len(vehicle_info) < 10:
        status = False

    rl_whl_spd = []
    for k, frame in vehicle_info.items():
        rl_whl_spd.append(float(frame["rl_whl_spd_e_sum"]))
    if not (is_spd_increasing(rl_whl_spd)):
        status = False
    return status

#先进行纯雷达里程计计算，若失败则采用IMU+轮速+雷达
def LidarWheelOdometryConstruct(seg_folder):
    LidarOdeometryConstruct(seg_folder)
    meta_path = os.path.join(seg_folder, "meta.json")
    gnss_path = os.path.join(seg_folder, "gnss.json")
    pair_pose_list = json.load(open(meta_path, 'r'))
    gnss_info_list = json.load(open(gnss_path, 'r'))
    if judge_odometry_failed(pair_pose_list, gnss_info_list):
        if vehicle_data_right(seg_folder):
            LidarWheelConstruct(seg_folder)
        else:
            logger.error(f"{seg_folder} vehicle data is err...")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        logger.error(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "node_gen_segments.log"))

    segment_path = run_config["preprocess"]["segment_path"]
    if not os.path.exists(segment_path):
        logger.error(f"No Segment Path [{segment_path}]")
        sys.exit(1)

    force = (run_config["preprocess"]["force"] == "True")

    seg_config = run_config["preprocess"]
    spec_clips = seg_config.get("spec_clips", None)

    folders = os.listdir(segment_path)
    if len(folders) == 0:
        logger.error(f"No Segment Folder [{segment_path}]")
        sys.exit(1)

    folders.sort()
    seg_interval = int(run_config["preprocess"]["interval"])
    
    logger.info(f"{segment_path} curr handle [{len(folders)}] segments with every_{seg_interval} clips.")
    # 创建一个事件对象，用于通知主进程  
    error_event = Event()
    pool = Pool(processes=8)
    try:
        for idx, sub in enumerate(folders):
            if spec_clips is not None:
                go_on = False
                for clip in spec_clips:
                    if clip in sub:
                        go_on = True
                        break
                if not go_on:
                    continue
            if idx % seg_interval != 0:
                continue
            seg_folder = os.path.join(segment_path, sub)
            meta_json = os.path.join(seg_folder, "meta.json")
            if not os.path.exists(meta_json):
                logger.error(f"{seg_folder}/meta.json Not Exists.")
                raise ValueError(f"{seg_folder}/meta.json Not Exists.")
            
            with open(meta_json, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            segid = meta["seg_uid"]
            first_lidar_pose = np.array(meta["frames"][0]["lidar"]["pose"]).astype(
                np.float32
            )
            dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
            pattern = run_config["odometry"]["pattern"]
            if (first_lidar_pose == dft_pose_matrix).all():
                # logger.info(f"{segid} have not been processed.")
                pass
            else:
                if not force:
                    logger.warning(f"{segid} not selected.")
                    continue
            
            if pattern == 0:
                logger.info(f"{segid} Start Processing LidarOdeometryConstruct..")
                pool.apply_async(LidarOdeometryConstruct, (seg_folder,),
                                 error_callback=lambda e: error_callback(e, error_event))
            elif pattern == 1:
                logger.info(f"{segid} Start Processing LidarWheelConstruct..")
                pool.apply_async(LidarWheelOdometryConstruct, (seg_folder,),
                                 error_callback=lambda e: error_callback(e, error_event))
            elif pattern == 2:
                logger.info(f"{segid} Start Processing ParkingOdeometryConstruct..")
                pool.apply_async(ParkingOdeometryConstruct, (seg_folder,),
                                error_callback=lambda e: error_callback(e, error_event))
            else:
                logger.error(f"Wrong Pattern {pattern} for {segid} odometry.")

        pool.close()  # 关闭进程池，防止提交新任务
        pool.join()  # 等待子进程终止
        # 等待所有任务完成或检测到异常  
        while True:  
            if error_event.is_set():  # 如果检测到异常  
                logger.error("Terminating all processes due to an error.")  
                pool.terminate()  # 终止所有子进程  
                sys.exit(1)  # 退出主程序  
            break  # 如果没有异常，跳出循环 

    except Exception as e:  
        # 捕获主进程中的异常  
        logger.error(f"Error occurred in main process: {e}")  
        pool.terminate()  # 终止所有子进程  
        pool.join()  # 等待子进程终止  
        sys.exit(1)  # 退出主程序  

    avail_segs = 0
    valid_segs = []
    for seg in folders:
        if spec_clips is not None:
            go_on = False
            for clip in spec_clips:
                if clip in seg:
                    go_on = True
                    break
            if not go_on:
                continue        
        seg_folder = os.path.join(segment_path, seg)
        meta_json = os.path.join(seg_folder, "meta.json")
        with open(meta_json, "r", encoding="utf-8") as fp:
            meta = json.load(fp)

        segid = meta['seg_uid']
        first_lidar_pose = np.array(meta["frames"][0]["lidar"]["pose"]).astype(
            np.float32
        )
        dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
        if (first_lidar_pose == dft_pose_matrix).all():
            # logger.info(f"{segid} not selected .")
            continue
        avail_segs += 1
        valid_segs.append(segid)
    logger.info(f"AT end total calculate {avail_segs} segs' pose.")
    logger.info(f"Valid segments' list is \n\t {valid_segs}")
