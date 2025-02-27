import os
import argparse
import csv
import numpy as np
from loguru import logger

MDC_CARS = ['10034', '48160', '04228', '18047']
INNO_CARS = ['48160', '36gl1']
HPP_CARS = ['23gc9', '48160', '72kx6']
MISS_THRESHOLD = 10 # unit is millseconds

def parse_args():
    parser = argparse.ArgumentParser(description="Init work node")
    # clip decoder params
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/data_autodrive/自动驾驶/hangbai/Traffic_Light/annotations",
    )
    parser.add_argument("--car", "-c", type=str, default="chery_e0y_10034")
    args = parser.parse_args()
    return args

def check_lost(frame_path, car_name):
    lidar_name = 'lidar'
    use_h265 = False
    use_mdc = False
    has_inno = False
    sensors = [
        "ofilm_surround_front_120_8M",
        "ofilm_surround_rear_100_2M",
        "ofilm_surround_front_left_100_2M",
        "ofilm_surround_front_right_100_2M",
        "ofilm_surround_rear_left_100_2M",
        "ofilm_surround_rear_right_100_2M",
    ]
    # if "10034" in car_name or "48160" in car_name:
    for _mdc_car in MDC_CARS:
        if _mdc_car in car_name:
            sensors = [
                "ofilm_surround_front_120_8M",
                "ofilm_surround_front_left_100_2M",
                "ofilm_surround_front_right_100_2M",
                "ofilm_surround_rear_left_100_2M",
                "ofilm_surround_rear_right_100_2M",
            ]
            use_h265 = True
            use_mdc = True
    for _hpp_car in HPP_CARS:
        if _hpp_car in car_name:
            sensors.extend(
                [
                    "ofilm_around_front_190_3M",
                    "ofilm_around_rear_190_3M",
                    "ofilm_around_left_190_3M",
                    "ofilm_around_right_190_3M",
                ]
            )
    for _inno_car in INNO_CARS:
        if _inno_car in car_name:
            has_inno = True
    _gnss_csv = "gnss.csv"
    _veh_csv = "vehicle.csv"

    def path_exist_check(frame_path):
        ret = True
        paths = [
            os.path.join(frame_path, _gnss_csv),
            os.path.join(frame_path, _veh_csv),
            os.path.join(frame_path, lidar_name)
        ]
        if has_inno:
            paths.append(os.path.join(frame_path, "inno_lidar"))
        paths.extend([os.path.join(frame_path, sensor) for sensor in sensors])
        for path in paths:
            if not os.path.exists(path):
                logger.error(f"{path} not exists")
                ret =  False
        return ret
    
    if not path_exist_check(frame_path):
        return False
    
    lidars = os.listdir(os.path.join(frame_path, lidar_name))
    lidars.sort()
    lidar_ts = [int(l.split(".")[0]) for l in lidars]
    lidar_ts_arr = np.array(lidar_ts)
    lidar_num = len(lidars)
    def sensor_frame_num_check(sensors, lidar_num):
        ret = True
        for s in sensors:
            s_num = len(os.listdir(os.path.join(frame_path, s)))            
            if "10034" in car_name: # camera in chery_10034 is 10HZ
                if s_num < (lidar_num - 4):
                    logger.error(f"{s} frame num {s_num} mismatch to lidar {lidar_num}")
                    ret = False
            elif use_h265: # MDC cars use H265 codec
                if s_num < (2 * lidar_num - 30):
                    logger.error(f"{s} frame num {s_num} mismatch to lidar {lidar_num}")
                    ret = False
            elif s_num < (2 * lidar_num - 8):# lidar is 10HZ, camera is 20HZ
                logger.error(f"{s} frame num {s_num} mismatch to lidar {lidar_num}")
                ret = False
        return ret
    
    def check_lidar_timestamp_correct(lidar_ts_arr, key='Lidar'):
        a_arr = lidar_ts_arr[:-1]
        b_arr = lidar_ts_arr[1:]
        judge_arr = np.logical_and(np.abs(a_arr - b_arr) < 105, np.abs(a_arr - b_arr) > 95)
        if not np.all(judge_arr):
            logger.error(f"{key} timestamp not correct")
            poss = np.where(np.logical_not(judge_arr))[0].tolist()
            for p in poss:
                logger.error(f"frame<{p}> {a_arr[p]} vs {b_arr[p]}")
            return False
        return True
    
    if not check_lidar_timestamp_correct(lidar_ts_arr):
        return False
    
    if has_inno:
        inno_lidars = os.listdir(os.path.join(frame_path, "inno_lidar"))
        inno_lidars.sort()
        inno_lidar_ts = [int(l.split(".")[0]) for l in inno_lidars]
        inno_lidar_ts_arr = np.array(inno_lidar_ts)
        if not check_lidar_timestamp_correct(inno_lidar_ts_arr, "inno_lidar"):
            return False
    
    if not sensor_frame_num_check(sensors, lidar_num):
        return False
    
    def timestamp_correct_check(sensors, lidar_ts_arr):
        ret = True
        lidar_base_ts = lidar_ts_arr[10:] # skip 10 head frames for skip image lost frame
        lidar_base_num = len(lidar_base_ts)
        total_miss_cnt = 0
        for s in sensors:
            images = os.listdir(os.path.join(frame_path, s))
            images.sort()
            images_ts = [int(i.split(".")[0]) for i in images]
            images_ts_arr = np.array(images_ts)
            img_num = len(images)

            pre_stat = True
            miss_cnt = 0
            for idx, ts in enumerate(lidar_base_ts.tolist()):
                miss_time_threshold = MISS_THRESHOLD
                if 'around' in s:
                    miss_time_threshold = 15
                match_idx = np.abs(np.abs(images_ts_arr - ts -25)).argmin()
                if np.abs(images_ts_arr[match_idx] - ts - 25) > MISS_THRESHOLD:
                    logger.error(f"frame<{idx}> {images_ts_arr[match_idx]} vs {ts}")
                    miss_cnt += 1
                    if not pre_stat:
                        logger.error(f"frame<{idx}&{idx-1}> continuously lost.") # 连续丢了2帧数据
                        ret = False
                    else:
                        pre_stat = False
                else:
                    pre_stat = True
            
            if miss_cnt > 5:
                logger.error(f"{s} miss {miss_cnt} frames")
                ret = False
            
            total_miss_cnt += miss_cnt
            
            if "10034" in car_name: # camera in chery_10034 is 10HZ
                _ts_arr = images_ts_arr[1:] - 100
            else:
                _ts_arr = images_ts_arr[1:] - 50
            # _ts_arr = images_ts_arr[1:] - 50
            diff = np.abs(images_ts_arr[:-1] - _ts_arr)
            diff_sum = np.sum(diff)
            # logger.debug(f"{s} timestamp diff sum {diff_sum}")
            if diff_sum > (img_num * 0.2):
                logger.error(f"{s} timestamp not correct")
                ret = False
        
        if total_miss_cnt > (len(sensors) * 2):
            logger.error(f"all {len(sensors)} cameras total miss {total_miss_cnt} frames")
            ret = False

        _ts_arr = lidar_ts_arr[1:] - 100
        diff = np.abs(lidar_ts_arr[:-1] - _ts_arr)
        if np.sum(diff) > (lidar_num):
            logger.error(f"lidar timestamp not correct")
            ret = False
        return ret

    if not timestamp_correct_check(sensors, lidar_ts_arr):
        return False

    return True


if __name__ == "__main__":
    logger.add(f"logs/lost_frame_check.log", rotation="50 MB", level="INFO")
    args = parse_args()
    query = args.input
    name = os.path.basename(query)
    
    clips = os.listdir(query)
    clips = [c for c in clips if os.path.isdir(os.path.join(query, c))]
    total = len(clips)
    pass_cnt = 0
    fail_cnt = 0
    fail_lst = []
    for clip in clips:
        if not os.path.isdir(os.path.join(query, clip)):
            continue     

        if check_lost(os.path.join(query, clip), args.car):
            logger.info(f"{clip} check pass")
            pass_cnt += 1
        else:
            logger.error(f"{clip} check failed")                        
            fail_cnt += 1
            fail_lst.append(clip)
    logger.info(f"{name} total: {total}, pass: {pass_cnt}, fail: {fail_cnt}")
    logger.warning(f"FAILED: \n \t{fail_lst}")
