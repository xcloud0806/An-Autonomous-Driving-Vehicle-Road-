import os
import json
import shutil
import numpy as np
from utils import  load_calibration, load_bpearls, undistort, project_lidar2img, db_update_seg, gen_datasets
from multiprocessing.pool import Pool
from datetime import datetime
from loguru import logger

curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_path)
import sys
sys.path.append(f"{curr_dir}/lib/python3.8/site_packages")
import pcd_iter as pcl
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
    seg_id = meta['seg_uid']

    reconstruct_path = os.path.join(segment_path, "multi_reconstruct")
    if not os.path.exists(reconstruct_path):
        reconstruct_path = os.path.join(segment_path, "reconstruct")
    
    # if os.path.isdir(reconstruct_path) and os.path.exists(os.path.join(reconstruct_path, "transform_matrix.json")):
    if os.path.isdir(reconstruct_path) and len(os.listdir(reconstruct_path)) >= 5: # 5: images, rgb, height, semantic, transform_matrix
        if not os.path.exists(dst_path):
            os.makedirs(dst_path, mode=0o775, exist_ok=True )
        dirs = os.listdir(reconstruct_path)
        for _dir in dirs:
            dir_i = os.path.join(reconstruct_path, _dir)
            subdir = os.path.basename(dir_i)
            tar_dir = dst_path+'/'+ subdir
            if os.path.exists(tar_dir):
                if os.path.isfile(tar_dir):
                    os.remove(tar_dir) 
                elif os.path.isdir(tar_dir): 
                    shutil.rmtree(tar_dir)
            # if not os.path.exists(tar_dir):
                # print("{} to {}".format(dir_i, tar_dir))
            if os.path.isdir(dir_i):
                shutil.copytree(dir_i, tar_dir)
            elif os.path.isfile(dir_i):
                shutil.copy(dir_i, tar_dir)
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
    specs = list()
    if seg_mode == "hpp" and os.path.exists(clip_lane_check):
        for rgbs in os.listdir(clip_lane_check):
            seg_id, _ = os.path.splitext(rgbs)
            specs.append(seg_id)
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
    logger.info(f"......\t{tgt_seg_path} prepare_lane3d... {str(datetime.now())}")
    for i, _seg in enumerate(seg_names):
        if len(specs) > 0 and _seg not in specs:
            continue
        if spec_clips is not None:
            go_on = False
            for clip in spec_clips:
                if clip in _seg:
                    go_on = True
                    break
            if not go_on:
                continue        
        seg_dir = os.path.join(seg_root_path, _seg)
        if not os.path.exists(os.path.join(seg_dir, "meta.json")):
            logger.warning("\tskip seg {} for not seleted".format(_seg))
            #db_update_seg(seg_dir, "", "")
            continue
        seg_meta_json = open(os.path.join(seg_dir, "meta.json"))
        try:
            meta = json.load(seg_meta_json)
        except Exception as e:
            logger.warning(f"{_seg}_meta.json load error.")
            # os.remove(os.path.join(seg_dir, "meta.json"))
            continue
        reconstruct_path = os.path.join(seg_dir, "reconstruct")
        if not os.path.exists(reconstruct_path) and not skip_reconstruct:
            logger.warning("\tskip seg {} for not seleted".format(_seg))
            db_update_seg(seg_dir, "", "")
            continue

        lane_dst_path = os.path.join(clip_lane, _seg)  
        if not skip_reconstruct:      
            logger.info("Prepare Lane Anno Data {} in {}".format(_seg, clip_lane))            
            prepare_lane3d(seg_dir, lane_dst_path)
            db_update_seg(seg_dir, lane_dst_path, "")   


if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        logger.error(f"{config_file} Not Exists.")
        sys.exit(1)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "prepare_lane3d_anno.log"))

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    node_main(run_config)
