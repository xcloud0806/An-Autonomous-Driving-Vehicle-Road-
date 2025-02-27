import sys, os
import json

import argparse
import os
import math
import pandas
import numpy as np
from utils import CarMeta
import traceback as tb
from concurrent.futures import ThreadPoolExecutor, wait
from utils import fill_clip_match


def correct_images_ts(clip_path, output_root, cameras, car_meta:CarMeta):      
    def call_ifly(car_meta: CarMeta, output_root, cameras):
        offset_detail = car_meta.offset_detail
        pool = ThreadPoolExecutor(max_workers=4)
        task_list = []
        for cam in cameras:
            camera_name = cam
            image_target_path = os.path.join(output_root, camera_name)
            
            imgs = os.listdir(image_target_path)
            imgs.sort()
            img_ts = []
            for img in imgs:
                ts = int(os.path.splitext(img)[0])
                img_ts.append(ts)
            base, n = offset_detail[cam]
            for ts in img_ts:
                correct_ts = ts - 50 * n
                src = os.path.join(image_target_path, "{}.jpeg".format(ts))
                dst = os.path.join(image_target_path, "{}.jpg".format(correct_ts))
                if not os.path.exists(src):
                    # print("CANNOT rename {} as not exist.".format(src))
                    continue
                #os.rename(src, dst)
                _task = pool.submit(os.rename, src, dst)
                task_list.append(_task)
        pool.shutdown()
        wait(task_list)

    if 'iflytek' in car_meta.dc_system_version:
        call_ifly(car_meta, output_root, cameras)

def match(car, clip_path, cameras, output_root, car_meta:CarMeta):
    # 统计数据初始化，数据看板相关
    clip_name = os.path.basename(clip_path)
    match_type = "match"
    num_of_lidar_frame = 0
    num_of_lidar_frame_lost = 0
    num_of_image_frame_lost = 0
    num_of_inno_lidars_frames_lost = None
    num_of_bpearl_lidars_frame_lost = None
    camera_lost = []
    inno_lidar_lost = None
    bpearl_lidar_lost = None
    
    lidar_name = car_meta.lidar_name
    offset_detail = car_meta.offset_detail

    match_res = {}
    cams_list = cameras

    pcd_dir = os.path.join(output_root, lidar_name)
    pcds = os.listdir(pcd_dir)
    if len(pcds) < 10:
        print("{} lidar is empty!!!".format(output_root))
        return
    pcds.sort()
    pcd_ts = []
    for pcd in pcds:
        ts = int(pcd[:-4])
        pcd_ts.append(ts)  
    num_of_lidar_frame = len(pcd_ts)

    for cam in cams_list:
        camera_name = cam
        base, n = offset_detail[cam]
        cam_res = []
        image_target_path = os.path.join(output_root, camera_name)        
        if cam not in cameras or not os.path.exists(image_target_path):
            camera_lost.append(cam)
            for idx, pts in enumerate(pcd_ts):
                cam_res.append(0) 
            match_res[cam] = cam_res
            continue
        
        imgs = os.listdir(image_target_path)
        imgs.sort()
        img_ts = []
        for img in imgs:
            ts = int(os.path.splitext(img)[0])
            img_ts.append(ts)
        img_ts_arr = np.array(img_ts)
        for pts in pcd_ts:
            match_idx = abs(img_ts_arr - pts - base).argmin()
            if abs(img_ts_arr[match_idx]- pts - base)>10:
                cam_res.append(0)
                num_of_image_frame_lost += 1
            else:
                cam_res.append(str(img_ts_arr[match_idx]))
        match_res[cam] = cam_res
    
    if len(car_meta.bpearl_lidars) > 0:
        num_of_bpearl_lidars_frame_lost = 0
        bpearl_lidar_lost = []
        for bpearl in car_meta.bpearl_lidars:
            bpearl_res = []
            bpearl_target_path = os.path.join(output_root, bpearl)  
            if not os.path.exists(bpearl_target_path):
                bpearl_lidar_lost.append(bpearl)
                for idx, pts in enumerate(pcd_ts):
                    bpearl_res.append(0) 
                match_res[bpearl] = bpearl_res
                continue
            bpcds = os.listdir(bpearl_target_path)
            bpcds.sort()
            bpcd_ts = []
            for p in bpcds:
                ts = int(os.path.splitext(p)[0])
                bpcd_ts.append(ts)
            bpcd_ts_arr = np.array(bpcd_ts)
            for pts in pcd_ts:
                match_idx = abs(bpcd_ts_arr - pts).argmin()
                if abs(bpcd_ts_arr[match_idx]- pts)>5:
                    bpearl_res.append(0)
                    num_of_bpearl_lidars_frame_lost += 1
                else:
                    bpearl_res.append(str(bpcd_ts_arr[match_idx]))
            match_res[bpearl] = bpearl_res
    
    if len(car_meta.inno_lidars) > 0:
        num_of_inno_lidars_frames_lost = 0
        inno_lidar_lost = []
        for inno in car_meta.inno_lidars:
            base = 25
            inno_res = []
            inno_target_path = os.path.join(output_root, inno)  
            if not os.path.exists(inno_target_path):
                inno_lidar_lost.append(inno)
                for idx, pts in enumerate(pcd_ts):
                    inno_res.append(0) 
                match_res[inno] = inno_res
                continue
            innos = os.listdir(inno_target_path)
            innos.sort()
            inno_ts = []
            for i in innos:
                ts = int(os.path.splitext(i)[0])
                inno_ts.append(ts)
            inno_ts_arr = np.array(inno_ts)
            for pts in pcd_ts:
                match_idx = abs(inno_ts_arr - pts - base).argmin()
                if abs(inno_ts_arr[match_idx]- pts - base)>5:
                    inno_res.append(0)
                    num_of_inno_lidars_frames_lost += 1
                else:
                    inno_res.append(str(inno_ts_arr[match_idx]))
            match_res[inno] = inno_res
    
    record = {}
    pcd_ts_str = []
    for _ts in pcd_ts:
        pcd_ts_str.append("{:010f}".format(_ts))
    record[lidar_name] = pcd_ts_str
    for idx, cam in enumerate(cams_list):
        camera_name = cam
        # print("{} length is {}".format(cam, len(match_res[cam])))
        record[camera_name] = match_res[cam]
    if len(car_meta.bpearl_lidars) > 0:
        for bpearl in car_meta.bpearl_lidars:
            record[bpearl] = match_res[bpearl]
    if len(car_meta.inno_lidars) > 0:
        for inno in car_meta.inno_lidars:
            record[inno] = match_res[inno]
    
    dframe = pandas.DataFrame(record)
    output = os.path.join(output_root, "matches.csv")
    dframe.to_csv(output, index=False, encoding='utf8')
        
    fill_clip_match(
        clip_name = clip_name,
        match_type = match_type,
        num_of_lidar_frame = num_of_lidar_frame,
        num_of_lidar_frame_lost = num_of_lidar_frame_lost,
        num_of_image_frame_lost = num_of_image_frame_lost,
        num_of_inno_lidars_frames_lost = num_of_inno_lidars_frames_lost,
        num_of_bpearl_lidars_frame_lost = num_of_bpearl_lidars_frame_lost,
        camera_lost = camera_lost,
        inno_lidar_lost = inno_lidar_lost,
        bpearl_lidar_lost = bpearl_lidar_lost,
    )

def match_raw(car, clip_path, cameras, output_root, car_meta:CarMeta):
    # 统计数据初始化，数据看板相关
    clip_name = os.path.basename(clip_path)
    match_type = "raw"
    num_of_lidar_frame = 0
    num_of_lidar_frame_lost = 0
    num_of_image_frame_lost = 0
    num_of_inno_lidars_frames_lost = None
    num_of_bpearl_lidars_frame_lost = None
    camera_lost = []
    inno_lidar_lost = None
    bpearl_lidar_lost = None
    
    lidar_name = car_meta.lidar_name
    offset_detail = car_meta.offset_detail

    match_res = {}
    cams_list = cameras

    pcd_dir = os.path.join(output_root, lidar_name)
    pcds = os.listdir(pcd_dir)
    if len(pcds) < 10:
        print("{} lidar is empty!!!".format(output_root))
        return
    pcds.sort()
    pcd_ts = []
    for pcd in pcds:
        ts = int(pcd[:-4])
        pcd_ts.append(ts)    
    num_of_lidar_frame = len(pcd_ts)

    for cam in cams_list:
        camera_name = cam
        base, n = offset_detail[cam]
        cam_res = []
        image_target_path = os.path.join(output_root, camera_name)        
        if cam not in cameras or not os.path.exists(image_target_path):
            camera_lost.append(cam)
            for idx, pts in enumerate(pcd_ts):
                cam_res.append(0) 
            match_res[cam] = cam_res
            continue
        
        imgs = os.listdir(image_target_path)
        imgs.sort()
        img_ts = []
        for img in imgs:
            ts = int(os.path.splitext(img)[0])
            img_ts.append(ts)
        img_ts_arr = np.array(img_ts)
        for pts in pcd_ts:
            match_idx = abs(img_ts_arr - pts - 25).argmin()
            if abs(img_ts_arr[match_idx]- pts - 25)>10:
                cam_res.append(0)
                num_of_image_frame_lost += 1
            else:
                cam_res.append(str(img_ts_arr[match_idx]))
        match_res[cam] = cam_res
        
    record = {}
    pcd_ts_str = []
    for _ts in pcd_ts:
        pcd_ts_str.append("{:010f}".format(_ts))
    record[lidar_name] = pcd_ts_str
    for idx, cam in enumerate(cams_list):
        camera_name = cam
        # print("{} length is {}".format(cam, len(match_res[cam])))
        record[camera_name] = match_res[cam]
    dframe = pandas.DataFrame(record)
    output = os.path.join(output_root, "raw.csv")
    dframe.to_csv(output, index=False, encoding='utf8')

    fill_clip_match(
        clip_name = clip_name,
        match_type = match_type,
        num_of_lidar_frame = num_of_lidar_frame,
        num_of_lidar_frame_lost = num_of_lidar_frame_lost,
        num_of_image_frame_lost = num_of_image_frame_lost,
        num_of_inno_lidars_frames_lost = num_of_inno_lidars_frames_lost,
        num_of_bpearl_lidars_frame_lost = num_of_bpearl_lidars_frame_lost,
        camera_lost = camera_lost,
        inno_lidar_lost = inno_lidar_lost,
        bpearl_lidar_lost = bpearl_lidar_lost,
    )

def match_frame(car, clip_path, cams:str, output_root, car_meta:CarMeta):
    if os.path.exists(os.path.join(output_root, "matches.csv")):
        print("### Decode {} Done.".format(output_root))
        return 
    cameras = []
    for cam in car_meta.cameras:
        image_target_path = os.path.join(output_root, cam)
        if os.path.exists(image_target_path):
            cameras.append(cam)

    print("### Decode {}, with cameras {}.".format(output_root, cameras))
    # 3. correct images' timestamp
    correct_images_ts(clip_path, output_root, cameras, car_meta)
    # 4. create matches.csv
    try:
        match(car, clip_path, cameras, output_root, car_meta)
        match_raw(car, clip_path, cameras, output_root, car_meta)
    except Exception as e:
        print(f"{car}.{clip_path} match failed as {e} \n")
        tb.print_exc()
        