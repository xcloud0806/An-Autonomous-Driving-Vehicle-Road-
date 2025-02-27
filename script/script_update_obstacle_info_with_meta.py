import os
import pandas as pd
import sys
import shutil
import numpy as np
import json
from datetime import datetime
import time
import cv2
import copy
from loguru import logger

sys.path.append("../utils")
sys.path.append("../lib/python3.8/site_packages")
import pcd_iter as pcl
from calib_utils import load_calibration, load_bpearls, undistort
from prepare_clip_infos import gen_datasets


MAX_LOST_LIMIT = 2
INFO_FILE = "infos.json"
MAX_FRAMES = 100
PICK_INTERVAL = 5 # 10 * 0.5
DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
# 10034 is huawei camera test car with 10hz images
BYPASS_CARS = [
    "chery_10034",
    "chery_04228"
]

def update_obstacle_info(dst_path, updated_path, segment_path):
    meta_json = os.path.join(segment_path, "meta.json")

    all_frames = {}
    with open(meta_json, "r") as fp:
        meta = json.load(fp)
        
        enable_cameras = meta["cameras"]
        segid = meta["seg_uid"]
        car_name = meta["car"]
        calib = meta["calibration"]
        calib_sensors = calib["sensors"]
        cameras = [item for item in enable_cameras if item in calib_sensors]

        first_lidar_pose = np.array(meta["frames"][0]["lidar"]["pose"]).astype(
            np.float32
        )
        dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
        if (first_lidar_pose == dft_pose_matrix).all():
            logger.warning(f"{segid} not selected .")
            return

        for f in meta["frames"]:
            lidar = f["lidar"]
            all_frames[str(lidar["timestamp"])] = f

    obstacle_info_file = os.path.join(dst_path, "{}_infos.json".format(segid))
    if not os.path.exists(obstacle_info_file):
        logger.warning(f"{segid} obstacle not exist.")
        return
    with open(obstacle_info_file, "r") as fp:
        info = json.load(fp)

    dst_info = {}
    dst_info["calib_params"] = calib
    dst_info["frames"] = []
    frames = info["frames"]
    for f in frames:
        frame_info = {}
        pf = f["pc_frame"]
        lidar_ts = pf["timestamp"]
        pcd_idx = pf["frame_id"]
        frame_idx = int(pcd_idx.split(".")[0])
        raw_frame = all_frames[lidar_ts]

        lidar = raw_frame["lidar"]
        pose = lidar["pose"]

        images = raw_frame["images"]
        img_info = {}
        for cam in cameras:
            cam_calib = load_calibration(meta, cam)
            if cam in images:
                cam_img = images[cam]
                img_info[cam] = {}
                lidar_to_camera = np.concatenate(
                    (cam_calib["extrinsic"], np.array([[0, 0, 0, 1]]))
                )
                camera_to_world = np.matmul(
                    np.array(pose), np.linalg.pinv(lidar_to_camera)
                )
                img_info[cam]["pose"] = camera_to_world.tolist()
                img_info[cam]["timestamp"] = str(cam_img["timestamp"])
                img_info[cam]["frame_id"] = frame_idx
        frame_info["image_frames"] = img_info

        pc_info = {}
        pc_info["pose"] = pose
        pc_info["timestamp"] = str(lidar["timestamp"])
        pc_info["frame_id"] = "{}.pcd".format(frame_idx)

        frame_info["pc_frame"] = pc_info
        dst_info["frames"].append(frame_info)

        updated_info_file = os.path.join(updated_path, "{}_infos.json".format(segid))
        with open(updated_info_file, "w") as fp:
            json.dump(dst_info, fp)


def node_main(run_config):
    logger.info("Starting node_update_obstacle_anno_info")
    seg_config = run_config["preprocess"]
    tgt_seg_path = seg_config["segment_path"]
    car_name = seg_config["car"]
    seg_mode = seg_config["seg_mode"]
    pre_anno_cfg = run_config['annotation']
    clip_obstacle = pre_anno_cfg["clip_obstacle"]
    clip_obstacle_test = pre_anno_cfg["clip_obstacle_test"]

    seg_root_path = tgt_seg_path
    if not os.path.exists(seg_root_path):
        logger.warning(f"{seg_root_path} NOT Exist...")
        sys.exit(0)
    seg_names = os.listdir(seg_root_path)
    seg_names.sort()
    seg_anno_dst = {}
    logger.info(f"......\t{tgt_seg_path} prepare_obstatcle... {str(datetime.now())}")
    for i, _seg in enumerate(seg_names):
        seg_dir = os.path.join(seg_root_path, _seg)
        if not os.path.exists(os.path.join(seg_dir, "meta.json")):
            logger.warning("\tskip seg {} for not seleted".format(_seg))
            continue
        
        seg_meta_json = open(os.path.join(seg_dir, "meta.json"))
        try:
            meta = json.load(seg_meta_json)
        except Exception as e:
            logger.error(f"{_seg}_meta.json load error.")
            os.remove(os.path.join(seg_dir, "meta.json"))
            continue
        gnss_json = os.path.join(seg_dir, "gnss.json")
        # tt_mode, _, _ = get_road_name(meta, gnss_json, test_road_gnss_file, odometry_mode)
        tt_mode, _, _ = gen_datasets(meta, gnss_json)
        obs_dst_path = os.path.join(clip_obstacle, _seg)
        updated_obs_path = obs_dst_path.replace("clip_obstacle", "clip_obstacle_info")
        if seg_mode == "luce" or seg_mode == "test" or tt_mode == "test":
            obs_dst_path = os.path.join(clip_obstacle_test, _seg)
            tt_mode = "test"
            updated_obs_path = obs_dst_path.replace(
                "clip_obstacle_test", "clip_obstacle_test_info"
            )

        os.makedirs(updated_obs_path, exist_ok=True)
        update_obstacle_info(obs_dst_path, updated_obs_path, seg_dir)


if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        logger.error(f"{config_file} Not Exists.")
        sys.exit(1)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "update_obstacle.log"))

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    node_main(run_config)

    # configs = [
    #     "/data_autodrive/users/xuanliu7/work_tmp/sihao_36gl1/ztwen_inno_data/sihao_36gl1_20240718_54a4c5473/run_sihao_36gl1_20240718.cfg",
    #     "/data_autodrive/users/xuanliu7/work_tmp/sihao_36gl1/ztwen_inno_data/sihao_36gl1_20240719_9f678d1c5/run_sihao_36gl1_20240719.cfg",
    #     "/data_autodrive/users/xuanliu7/work_tmp/sihao_36gl1/ztwen_inno_data/sihao_36gl1_20240721_a90a614f3/run_sihao_36gl1_20240721.cfg",
    #     "/data_autodrive/users/xuanliu7/work_tmp/sihao_36gl1/ztwen_inno_data/sihao_36gl1_20240722_1_ad0e68484/run_sihao_36gl1_20240722_1.cfg",
    #     "/data_autodrive/users/xuanliu7/work_tmp/sihao_36gl1/ztwen_inno_data/sihao_36gl1_20240722_2_ee9ce09ff/run_sihao_36gl1_20240722_2.cfg",
    #     "/data_autodrive/users/xuanliu7/work_tmp/sihao_36gl1/ztwen_inno_data/sihao_36gl1_20240722_3_64f525074/run_sihao_36gl1_20240722_3.cfg",
    #     "/data_autodrive/users/xuanliu7/work_tmp/sihao_36gl1/ztwen_inno_data/sihao_36gl1_20240723_1_9d613b0d5/run_sihao_36gl1_20240723_1.cfg",
    #     "/data_autodrive/users/xuanliu7/work_tmp/sihao_36gl1/ztwen_inno_data/sihao_36gl1_20240723_2_d9b7cb0d4/run_sihao_36gl1_20240723_2.cfg",
    #     "/data_autodrive/users/xuanliu7/work_tmp/sihao_36gl1/ztwen_inno_data/sihao_36gl1_20240723_3_9c143a404/run_sihao_36gl1_20240723_3.cfg",
    #     "/data_autodrive/users/xuanliu7/work_tmp/sihao_36gl1/ztwen_inno_data/sihao_36gl1_20240724_c3e54213d/run_sihao_36gl1_20240724.cfg",
    # ]

    # for cfg in configs:
    #     logger.info(f"Starting node_update_obstacle_anno_info for {cfg}")
    #     with open(cfg, "r") as fp:
    #         run_config = json.load(fp)
    #         node_main(run_config)
