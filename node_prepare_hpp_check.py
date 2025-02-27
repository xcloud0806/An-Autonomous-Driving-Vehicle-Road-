import os
import json
import shutil
import numpy as np
from utils import  load_calibration, load_bpearls, undistort, project_lidar2img, db_update_seg, gen_datasets
from multiprocessing.pool import Pool
from datetime import datetime
from loguru import logger
import cv2
import sys

MAX_LOST_LIMIT = 2
INFO_FILE = "infos.json"
MAX_FRAMES = 80
PICK_INTERVAL = 5 # 10 * 0.5
DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

def prepare_hpp_check(segment_path, dst_path):
    meta_json = os.path.join(segment_path, "meta.json")
    meta_fp = open(meta_json, "r")
    meta = json.load(meta_fp)
    seg_id = meta['seg_uid']

    reconstruct_path = os.path.join(segment_path, "multi_reconstruct")
    if not os.path.exists(reconstruct_path):
        reconstruct_path = os.path.join(segment_path, "reconstruct")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path, mode=0o775, exist_ok=True )
    # if os.path.isdir(reconstruct_path) and os.path.exists(os.path.join(reconstruct_path, "transform_matrix.json")):
    if os.path.isdir(reconstruct_path) and len(os.listdir(reconstruct_path)) >= 5: # 5: images, rgb, height, semantic, transform_matrix
        dirs = os.listdir(reconstruct_path)
        recon_jpgs = []
        for _dir in dirs:
            dir_i = os.path.join(reconstruct_path, _dir)
            if os.path.isfile(dir_i) and _dir.endswith(".jpg"):
                img = cv2.imread(dir_i)
                if img is not None:
                    recon_jpgs.append(img)

        res_arr = np.hstack(recon_jpgs)
        cv2.imwrite(os.path.join(dst_path, f"{seg_id}.jpg"), res_arr)
    meta_fp.close()

def node_main(run_config):
    seg_config = run_config["preprocess"]
    tgt_seg_path = seg_config["segment_path"]
    car_name = seg_config['car']
    seg_mode = seg_config['seg_mode']
    rec_cfg = run_config["reconstruction"]
    skip_reconstruct = False
    if rec_cfg['enable'] != "True":
        skip_reconstruct = True
    pre_anno_cfg = run_config['annotation']
    clip_lane = pre_anno_cfg['clip_lane']
    clip_lane_check = clip_lane.replace("clip_lane", "clip_lane_check")
    deploy_cfg = run_config["deploy"]
    subfix = deploy_cfg['data_subfix']
    spec_clips = seg_config.get("spec_clips", None)
    odometry_mode = run_config["odometry"]["pattern"]
    seg_root_path = tgt_seg_path
    if not os.path.exists(seg_root_path):
        logger.error(f"{seg_root_path} NOT Exist...")
        sys.exit(1)
    seg_names = os.listdir(seg_root_path)
    seg_names.sort()
    seg_anno_dst = {}
    logger.info(f"......\t{tgt_seg_path} prepare_lane3d_check... {str(datetime.now())}")
    for i, _seg in enumerate(seg_names):
        if spec_clips is not None:
            go_on = False
            for clip in spec_clips:
                if clip in _seg:
                    go_on = True
                    break
            if not go_on:
                continue        
        seg_dir = os.path.join(seg_root_path, _seg)
        meta_json_path = os.path.join(seg_dir, "meta.json")
        date_seg = _seg.split("_")[2]
        if not os.path.exists(meta_json_path):
            # logger.warning(f"\tskip {_seg} for not seleted, {seg_dir}/meta.json not exist")
            logger.error(f"\tskip {_seg} for not seleted, car_name = {car_name}, {date_seg}")
            #db_update_seg(seg_dir, "", "")
            continue
        seg_meta_json = open(meta_json_path)
        try:
            meta = json.load(seg_meta_json)
            if 'key_frames' not in meta:
                logger.warning(f"{seg_dir} skip. Because no [key_frame] field.")
                logger.error(f"\tskip {_seg} for not seleted, car_name = {car_name}, {date_seg}")
                continue
            sig_frames = meta['key_frames']
            if len(sig_frames) <= 10:
                logger.warning(f"{seg_dir} skip. Because too few key frame.")
                logger.error(f"\tskip {_seg} for not seleted, car_name = {car_name}, {date_seg}")
                continue
            sig_frames_lost = meta.get('key_frames_lost', 0)
            if sig_frames_lost > 2:
                logger.warning(f"{seg_dir} skip. Because too many key frame lost. [{sig_frames_lost}]")
                logger.error(f"\tskip {_seg} for not seleted, car_name = {car_name}, {date_seg}")
                continue
        except Exception as e:
            logger.warning(f"{meta_json_path} load error, {e}")
            logger.error(f"\tskip {_seg} for not seleted, car_name = {car_name}, {date_seg}")
            # os.remove(os.path.join(seg_dir, "meta.json"))
            continue
        reconstruct_path = os.path.join(seg_dir, "reconstruct")
        if not os.path.exists(reconstruct_path) and not skip_reconstruct:
            logger.warning(f"\tskip seg {_seg} for not seleted, {reconstruct_path} not exist ")
            logger.error(f"\tskip {_seg} for not seleted, car_name = {car_name}, {date_seg}")
            continue

        if not skip_reconstruct:      
            logger.info("Prepare Lane Check Data {} in {}".format(_seg, clip_lane))            
            prepare_hpp_check(seg_dir, clip_lane_check)

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        logger.error(f"{config_file} Not Exists.")
        sys.exit(1)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "prepare_lane3d_check.log"))

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    node_main(run_config)
