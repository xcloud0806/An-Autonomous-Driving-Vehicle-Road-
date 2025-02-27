import os
import json
import shutil
import numpy as np
from utils import  load_calibration, load_bpearls, undistort, project_lidar2img, db_update_seg, gen_datasets
from multiprocessing.pool import Pool
from datetime import datetime
import cv2
from utils import mail_handle
from node_prepare_obstacle_anno_data import prepare_obstacle, prepare_check

curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_path)
import sys
sys.path.append(f"{curr_dir}/lib/python3.8/site_packages")
import pcd_iter as pcl


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
    clip_obstacle =  pre_anno_cfg['clip_obstacle']
    clip_obstacle_test =  pre_anno_cfg['clip_obstacle_test']
    clip_check =  pre_anno_cfg['clip_check']
    test_road_gnss_file = pre_anno_cfg['test_gnss_info']
    deploy_cfg = run_config["deploy"]
    subfix = deploy_cfg['data_subfix']
    
    odometry_mode = run_config["odometry"]["pattern"]
    spec_clips = seg_config.get("spec_clips", None)
    pool = Pool(processes=4)
    seg_root_path = tgt_seg_path
    if not os.path.exists(seg_root_path):
        print(f"{seg_root_path} NOT Exist...")
        sys.exit(0)
    seg_names = os.listdir(seg_root_path)
    seg_names.sort()
    seg_anno_dst = {}
    print(f"......\t{tgt_seg_path} prepare_lane3d&prepare_obstatcle... {str(datetime.now())}")
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
        if not os.path.exists(os.path.join(seg_dir, "meta.json")):
            print("\tskip seg {} for not seleted".format(_seg))
            #db_update_seg(seg_dir, "", "")
            continue
        seg_meta_json = open(os.path.join(seg_dir, "meta.json"))
        try:
            meta = json.load(seg_meta_json)
        except Exception as e:
            print(f"{_seg}_meta.json load error.")
            os.remove(os.path.join(seg_dir, "meta.json"))
            continue
        reconstruct_path = os.path.join(seg_dir, "reconstruct")
        if not os.path.exists(reconstruct_path) and not skip_reconstruct:
            print("\tskip seg {} for not seleted".format(_seg))
            db_update_seg(seg_dir, "", "")
            continue

        lane_dst_path = os.path.join(clip_lane, _seg)  
        if not skip_reconstruct:      
            print("Prepare Lane Anno Data {} in {}".format(_seg, clip_lane))
            prepare_lane3d(seg_dir, lane_dst_path)

        gnss_json = os.path.join(seg_dir, "gnss.json")
        # tt_mode, _, _ = get_road_name(meta, gnss_json, test_road_gnss_file, odometry_mode)
        tt_mode, _, _ = gen_datasets(meta, gnss_json, odometry_mode=odometry_mode)
        obs_dst_path = os.path.join(clip_obstacle, _seg)
        if seg_mode == 'luce' or seg_mode == 'test' or tt_mode == 'test' or seg_mode == 'aeb':
            obs_dst_path = os.path.join(clip_obstacle_test, _seg)
            tt_mode = 'test'
        seg_anno_dst[_seg] = {
            "lane": lane_dst_path,
            "obs": obs_dst_path,
            "mode": tt_mode
        }
        if not os.path.exists(os.path.join(obs_dst_path, "{}_info.json".format(_seg))):            
            print("Prepare Obstacle Anno Data {} in {}".format(_seg, obs_dst_path))
            #prepare_obstacle(seg_dir, obs_dst_path)
            pool.apply_async(prepare_obstacle, args=(seg_dir, obs_dst_path, tt_mode))
        else:
            print("Prepare Obstacle {} Done".format(_seg))

    pool.close()
    pool.join()  
    print(f"......\t{car_name}.{tgt_seg_path} prepare_lane3d&prepare_obstatcle end... {str(datetime.now())}")  
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
        if not os.path.exists(os.path.join(seg_dir, "meta.json")):
            continue
        reconstruct_path = os.path.join(seg_dir, "reconstruct")
        if not os.path.exists(reconstruct_path) and not skip_reconstruct:
            continue

        seg_meta_json = open(os.path.join(seg_dir, "meta.json"))
        meta = json.load(seg_meta_json)
        lane_dst_path = seg_anno_dst[_seg]['lane']
        obs_dst_path = seg_anno_dst[_seg]['obs']
        if skip_reconstruct and not os.path.exists(os.path.join(obs_dst_path, "{}_infos.json".format(_seg))):
            print(f"REMOVE {_seg} for short of data.")
            os.system("rm -rf {}".format(obs_dst_path))
            db_update_seg(seg_dir, "", "")
        elif not skip_reconstruct and not os.path.exists(os.path.join(lane_dst_path, "transform_matrix.json")) \
            or not os.path.exists(os.path.join(obs_dst_path, "{}_infos.json".format(_seg))):
            print(f"REMOVE {_seg} for short of data.")
            mode = seg_anno_dst[_seg]['mode']
            if mode == 'test' or mode == 'luce':
                db_update_seg(seg_dir, "", obs_dst_path)
                print("Produce check images {}".format(_seg))
                prepare_check(clip_check, obs_dst_path, meta)  
            else:
                os.system("rm -rf {} {}".format(lane_dst_path, obs_dst_path))
                db_update_seg(seg_dir, "", "")
        else:
            obs_info_json = os.path.join(obs_dst_path, "{}_infos.json".format(_seg))
            if not os.path.exists(obs_info_json):
                print(f"REMOVE {_seg} for short of data.")
                os.system("rm -rf {}".format(obs_dst_path))
                continue
            with open(obs_info_json, "r") as info_fp:
                info_json_dict = json.load(info_fp)
                if len(info_json_dict['frames']) < 5:
                    print(f"REMOVE {_seg} for too little frames.")
                    os.system("rm -rf {}".format(obs_dst_path))
                    continue
            if skip_reconstruct:
                db_update_seg(seg_dir, "", obs_dst_path)
            else:
                db_update_seg(seg_dir, lane_dst_path, obs_dst_path)            
            print("Produce check images {}".format(_seg))
            prepare_check(clip_check, obs_dst_path, meta)  
    print(f"......\t{tgt_seg_path} prepare_check end... {str(datetime.now())}")

    print(f" ### ### ### ### \n")
    print(f" ### {tgt_seg_path} Done. ### \n")
    print(f" ### ### ### ### \n")

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    node_main(run_config)
