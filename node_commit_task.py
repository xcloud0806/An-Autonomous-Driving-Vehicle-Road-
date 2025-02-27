from utils import (
    get_redis_rcon,
    acquire_lock_with_timeout,
    release_lock,
    RECONSTRUCT_QUEUE,
    RECONSTRUCT_PRIORITY_QUEUE,
    push_msg,
    read_msg,
    RECONSTRUCT_LOCK_KEY,
    MULTICLIPS_LOCK_KEY,
    MULTICLIPS_QUEUE,
    MULTICLIPS_PRIORITY_QUEUE,
)

import os, sys
import json
import numpy as np
import time
import shutil
from loguru import logger

DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

def commit_multiclips_task(run_config, config_file):
    rcon = get_redis_rcon()
    segment_path = run_config["preprocess"]["segment_path"] 
    if not os.path.exists(segment_path):
        print("No Segment Path")
        return
    multi_info_path = run_config["multi_seg"]["multi_info_path"]
    colls = os.listdir(multi_info_path)
    colls.sort()
    for coll in colls:
        if coll.startswith("."):
            continue
        if not os.path.exists(os.path.join(multi_info_path, coll)):
            continue
        if not os.path.exists(os.path.join(multi_info_path, coll, "multi_info.json")):
            continue
        with open(os.path.join(multi_info_path, coll, "multi_info.json"), "r") as f:
            multi_info = json.load(f)          
            if multi_info[coll]["multi_odometry_status"]:    
                msg_info = {"config": config_file, "coll_id": coll}
                while True:
                    lock_val = acquire_lock_with_timeout(rcon, MULTICLIPS_LOCK_KEY)
                    if not lock_val or lock_val is None:
                        time.sleep(1)
                        continue

                    push_msg(rcon, MULTICLIPS_QUEUE, msg_info)
                    release_lock(rcon, MULTICLIPS_LOCK_KEY, lock_val)
                    break
    return 

def node_main(run_config, config_file):
    seg_mode = run_config["preprocess"]['seg_mode']
    segment_path = run_config["preprocess"]["segment_path"]
    if 'multi_seg' in run_config and run_config['multi_seg']['enable'] == 'True':
        return commit_multiclips_task(run_config, config_file)

    force = (run_config["preprocess"]['force'].lower() != 'false')
    if not os.path.exists(segment_path):
        logger.error("segment_path does not exist!")
        sys.exit(1)
    
    seg_config = run_config["preprocess"]
    spec_clips = seg_config.get("spec_clips", None)

    work_config_file = config_file
    folders = os.listdir(segment_path)
    if len(folders) == 0:
        logger.error("segment_path is empty!")
        sys.exit(1)
    if run_config["method"] == "cli":
        if run_config["reconstruction"]["enable"] != "True":
            return
        print("In CLI mode, will send RECONSTRUCT task to redis queue.")
    if run_config['method'] == 'ripple':
        if run_config["reconstruction"]["enable"] != "True":
            return
        print("In RIPPLE mode, will send RECONSTRUCT task to redis queue.")
        task_id = run_config['ripples_platform_demand']['task_id']
        seg_config = run_config["preprocess"]
        car_name = seg_config['car']
        frame_path = seg_config['frames_path']
        subfix = os.path.basename(frame_path)
        work_tmp_path = os.path.join("/data_autodrive/users/xuanliu7/work_tmp_ripple", f"{car_name}_{subfix}_{task_id}" )
        try:
            os.makedirs(work_tmp_path, exist_ok=True, mode=0o777)
            config_file_name = os.path.basename(config_file)
            dst_conf = os.path.join(work_tmp_path, config_file_name)
            shutil.copy(config_file, dst_conf)
            work_config_file = dst_conf
        except Exception as e:
            logger.exception(f"Caught an exception of type {type(e).__name__}: {e}")
            sys.exit(1)

    task_queue = RECONSTRUCT_QUEUE
    task_lock_key = RECONSTRUCT_LOCK_KEY

    if seg_mode == 'luce' or seg_mode == 'hpp_luce' or seg_mode == 'aeb':
        task_queue = RECONSTRUCT_PRIORITY_QUEUE
        task_lock_key = RECONSTRUCT_LOCK_KEY

    rcon = get_redis_rcon()
    for segid in folders:
        if spec_clips is not None:
            go_on = False
            for clip in spec_clips:
                if clip in segid:
                    go_on = True
                    break
            if not go_on:
                continue        
        seg_folder = os.path.join(segment_path, segid)
        meta_file = os.path.join(seg_folder, "meta.json")

        try:
            meta_json = open(meta_file, "r")
            meta = json.load(meta_json)
            if 'hpp' in seg_mode:
                frames_path = run_config["preprocess"]['frames_path']
                seg_name = segid.split("_")[2]
                frames_seg_path = os.path.join(frames_path, seg_name)
                frames_seg_meta_path = os.path.join(frames_seg_path, "meta.json")
                shutil.copy(meta_file, frames_seg_meta_path)
        except Exception as e:
            logger.exception(f"Caught an exception of type {type(e).__name__}: {e}")
            sys.exit(1)
                    
        first_lidar_pose = np.array(meta["frames"][0]["lidar"]["pose"]).astype(
            np.float32
        )
        dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
        if (first_lidar_pose == dft_pose_matrix).all():
            print(f"{segid} not selected .")
            continue
        msg_info = {"config": work_config_file, "specs": [segid]}
        while True:
            lock_val = acquire_lock_with_timeout(rcon, task_lock_key)
            if not lock_val or lock_val is None:
                time.sleep(1)
                continue
            try:
                result = push_msg(rcon, task_queue, msg_info)
                if result > 0:
                    logger.info(f"push_msg {segment_path}/{segid} done, 当前队列长度为: {result}")
                    break
            except Exception as e:                
                logger.warning(f"push_msg {segment_path}/{segid} failed")
                
            release_lock(rcon, task_lock_key, lock_val)

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)

    work_tmp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_tmp_dir, "node_commit_task.log"), rotation="50 MB")

    node_main(run_config, config_file)
