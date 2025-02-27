import os
import json
import shutil
import numpy as np
from utils import  load_calibration, load_bpearls, undistort, project_lidar2img, db_update_seg, get_road_name
from multiprocessing.pool import Pool
from datetime import datetime
from loguru import logger
import sys
from node_prepare_obstacle_anno_data import prepare_obstacle, prepare_check

MAX_LOST_LIMIT = 2
INFO_FILE = "infos.json"
MAX_FRAMES = 80
PICK_INTERVAL = 5 # 10 * 0.5
DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

def prepare_lane3d(segment_path, dst_path):
    meta_json = os.path.join(segment_path, "meta.json")
    meta_fp = open(meta_json, "r")
    meta = json.load(meta_fp)

    reconstruct_path = os.path.join(segment_path, "multi_reconstruct")
    if not os.path.exists(reconstruct_path):
        reconstruct_path = os.path.join(segment_path, "reconstruct")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path, mode=0o775, exist_ok=True )
    if os.path.isdir(reconstruct_path) and os.path.exists(os.path.join(reconstruct_path, "transform_matrix.json")):
        dirs = os.listdir(reconstruct_path)
        for _dir in dirs:
            dir_i = os.path.join(reconstruct_path, _dir)
            subdir = os.path.basename(dir_i)
            tar_dir = dst_path+'/'+ subdir
            if not os.path.exists(tar_dir):
                # print("{} to {}".format(dir_i, tar_dir))
                if os.path.isdir(dir_i):
                    shutil.copytree(dir_i, tar_dir)
                elif os.path.isfile(dir_i):
                    shutil.copy(dir_i, tar_dir)
    meta_fp.close()

def node_main(run_config):
    if 'multi_seg' not in run_config:
        logger.error(" multi_seg not in config file")
        sys.exit(1)
    if not run_config['multi_seg']['enable']:
        logger.error(" multi_seg not enable")
        sys.exit(1)
    seg_config = run_config["preprocess"]
    tgt_seg_path = seg_config["segment_path"]

    # tgt_coll_path = tgt_seg_path.replace("_seg", "_coll")
    tgt_coll_path = run_config['multi_seg']['multi_info_path']
    if not os.path.exists(tgt_seg_path) or not os.path.exists(tgt_coll_path):
        logger.error(f"{tgt_seg_path} or {tgt_coll_path} NOT Exist...")
        sys.exit(1)
    car_name = seg_config['car']
    seg_mode = seg_config['seg_mode']
    rec_cfg = run_config["reconstruction"]
    skip_reconstruct = False
    if rec_cfg['enable'] != "True":
        skip_reconstruct = True
    pre_anno_cfg = run_config['annotation']
    clip_lane = pre_anno_cfg['clip_lane']
    deploy_cfg = run_config["deploy"]
    subfix = deploy_cfg['data_subfix']

    colls = os.listdir(tgt_coll_path)
    for coll in colls:
        logger.info(f"Processing {coll}")
        coll_path = os.path.join(tgt_coll_path, coll)
        multi_info_json = os.path.join(coll_path, "multi_info.json")
        if not os.path.exists(multi_info_json):
            continue
        with open(multi_info_json, "r") as fp:
            multi_info = json.load(fp)

        multi_info_id = coll
        multi_info_status = True
        if not multi_info[multi_info_id]["multi_odometry_status"] :
            multi_info_status = False
        if not multi_info_status:
            logger.warning(f"{multi_info_id} multi odometry status is False")
            continue
        night_seg_path = multi_info[multi_info_id]["main_clip_path"][0]
        night_seg_id = os.path.basename(night_seg_path)
        if night_seg_path.endswith("/"):
            night_seg_id = night_seg_id[:-1]
        night_recon_path = os.path.join(night_seg_path, "reconstruct")
        if os.path.exists(night_recon_path):
            if os.path.exists(os.path.join(night_recon_path, "transform_matrix.json")):
                # logger.info(f"{night_recon_path} already exists, skip copy.")
                lane_dst_path = os.path.join(clip_lane, night_seg_id)  
                prepare_lane3d(night_seg_path, lane_dst_path)
        else:            
            day_seg_path = multi_info[multi_info_id]["clips_path"][0]
            day_seg_id = os.path.basename(day_seg_path)
            if day_seg_path.endswith("/"):
                day_seg_id = day_seg_id[:-1]
            logger.info(f"Processing {coll} with [{night_seg_id} & {day_seg_id}]")
            if not os.path.exists(night_seg_path) or not os.path.exists(day_seg_path):
                logger.error(f"{night_seg_path} or {day_seg_path} NOT Exist...")
                continue

            day_recon_path = os.path.join(day_seg_path, "reconstruct")            
            shutil.copytree(day_recon_path, night_recon_path)
            lane_dst_path = os.path.join(clip_lane, night_seg_id)  
            prepare_lane3d(night_seg_path, lane_dst_path)

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]        

    if not os.path.exists(config_file):
        logger.error(f"{config_file} Not Exists.")
        sys.exit(1)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "prepare_night_anno.log"))

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    node_main(run_config)
