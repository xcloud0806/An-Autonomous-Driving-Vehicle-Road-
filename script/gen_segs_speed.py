import os
import pandas as pd
import sys
import shutil
import numpy as np
import json
from datetime import datetime
import time
import cv2
sys.path.append("../utils")
from calib_utils import  load_calibration, load_bpearls, undistort
from db_utils import query_seg
from loguru import logger

root_path = "/data_autodrive/users/xbchang2/临时文件存放/导出数据/custom"

if __name__ == "__main__":
    seg_list = []
    car_list = os.listdir(root_path)
    for car in car_list:
        car_path = os.path.join(root_path, car)
        dates = os.listdir(car_path)
        for date in dates:
            date_path = os.path.join(car_path, date)
            segs = os.listdir(date_path)
            for seg in segs:
                seg_list.append(seg)

    result = query_seg(seg_list)
    res_cnt = result[0]
    
    table = []

    if res_cnt > 0:
        seg_contents = result[1]
        for seg_content in seg_contents:
            segid = seg_content["id"]
            logger.info(f">>>{segid}")
            veh_file = seg_content["vehicleDriverFilePath"]
            if not os.path.exists(veh_file):
                logger.error(f"{veh_file} not exists")
                continue

            with open(veh_file, "r") as f:
                veh_content = json.load(f)

            speeds = []
            for k, v in veh_content.items():
                speed_kph = v['vehicle_spd']
                speeds.append(speed_kph)

            speeds_array = np.array(speeds)
            sorted_speeds = np.sort(speeds_array)
            idx = int(len(speeds_array) * 0.95 - 1)
            table.append([segid, float(sorted_speeds[idx])])

    df = pd.DataFrame(table, columns=["segid", "speed"])
    df.to_csv("./speed_table.csv", index=False)
