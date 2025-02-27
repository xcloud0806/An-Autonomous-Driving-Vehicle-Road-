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
sys.path.append("../lib/python3.8/site_packages")
import pcd_iter as pcl
from calib_utils import  load_calibration, load_bpearls, undistort
from db_utils import query_seg
from loguru import logger
from multiprocessing.pool import Pool

xlsx_file = "/data_autodrive/users/brli/dev_raw_data/total_objcnt_speed_1st_2nd.xlsx"
logger.add("gen_all_segs_frame_num.log", rotation="10 MB")

DST_ROOT = "/data_autodrive/auto/label_4d/second"
DST_INFO_ROOT = "/data_autodrive/auto/label_4d/second/info"

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
            #(objcnt, speed, daynight, task, car, deploy_subfix)
            ret[segid] = {
                "objcnt": objcnt,
                "speed": speed,
                "daynight": daynight,
                "task": task,
                "car": car,
                "deploy_subfix": str(deploy_subfix),
                "segid": segid
            }
    logger.info(f"EXCEL total_count: {total_cnt}")
    return ret

def gen_frames_num_by_spec_segs():
    df = pd.read_excel(xlsx_file, skiprows=1)
    use_count = 0
    total_count = df.shape[0]
    lst = []
    total_deal_objcnt = 0
    for idx, row in df.iterrows():
        segid, objcnt, speed, daynight, task, car, deploy_subfix = row
        if float(objcnt) < 4 or float(objcnt) > 65:
            continue
        if float(speed) < 0.1 and task != "frwang_chengshilukou" and task != "frwang_chadaohuichu":
            logger.warning(f"{task}.{segid} speed too low")
            continue
        res = query_seg([segid])
        res_cnt = res[0]
        
        if res_cnt > 0:
            seg_content = res[1][0]
            seg_clip_obs_path = seg_content['pathMap']['obstacle3dAnnoDataPath']
            subfix = seg_content['collectionDataDate']
            car_name = seg_content['calibrationCar']
            dst_path = os.path.join(DST_ROOT, car_name, subfix, segid)
            dst_info_dir = os.path.join(DST_INFO_ROOT, car_name, subfix, segid)
            dst_info = os.path.join(DST_INFO_ROOT, car_name, subfix, segid, f"{segid}_infos.json")
            if os.path.exists(dst_info):
                with open(dst_info, 'r') as f:
                    dst_info_content = json.load(f)
                    frame_cnt = len(dst_info_content['frames'])
                    
            if segid in deal_segs:
                frame_obj_cnt = int(frame_cnt * objcnt)
                total_deal_objcnt += frame_obj_cnt

            res_row = [segid, objcnt, speed, daynight, task, car, deploy_subfix, frame_cnt, int(frame_cnt * objcnt)]
            lst.append(res_row)
    # res dump csv
    logger.info(f"Already deal {len(deal_segs)} segs with {total_deal_objcnt} objects.")
    res_df = pd.DataFrame(lst, columns=['segid', 'objcnt', 'speed', 'daynight', 'task', 'car', 'deploy_subfix', 'frame_cnt', 'total_objcnt'])
    res_df.to_csv("res.csv", index=False)

def gen_frames_num_by_annos():
    seg_infos = parse_infos_with_excels([xlsx_file])

    annos_path = "/data_autodrive/users/xbchang2/临时文件存放/文件传输中转/111_512016/refined-all/"
    cars = os.listdir(annos_path)
    cars.sort()
    idx = 0
    tables = []
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
                if not os.path.exists(os.path.join(seg_path, "annotations", "result.json")):
                    continue

                seg_info = seg_infos[seg]
                speed = 3.6 * float(seg_info['speed'])
                daynight = seg_info['daynight']
                task = seg_info['task']
                
                logger.info(f"[{idx}] <-> [{car}.{subfix}.{seg}]")
                result_json = os.path.join(seg_path, "annotations", "result.json")
                with open(result_json, 'r') as fp:
                    result = json.load(fp)
                    
                frame_annos = result['researcherData']
                frame_cnt = len(frame_annos)
                bbox_cnt = 0
                for _f in frame_annos:
                    _anno = _f['frame_annotations']
                    _bbox = len(_anno)
                    bbox_cnt += _bbox
                avg_cnt = bbox_cnt / frame_cnt
                logger.info(f"\t {car}.{subfix}.{seg} frame_cnt: {frame_cnt}, bbox_cnt: {bbox_cnt}")
                tables.append([seg, speed, daynight, task, car, subfix, frame_cnt, bbox_cnt, avg_cnt])

    df = pd.DataFrame(tables, columns=['segid', 'speed', 'daynight', 'task', 'car', 'subfix', 'frame_cnt', 'bbox_cnt', 'avg_cnt'])
    df.to_csv("res.csv", index=False)

if __name__ == "__main__":
    gen_frames_num_by_annos()




