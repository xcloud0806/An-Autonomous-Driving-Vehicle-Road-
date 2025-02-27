import os, sys
import json
import numpy as np
import time
from loguru import logger

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

TIMEOUT_SET=(86400) # three day -> one day
DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

def retry_tasks(config_file, seglist, lock_key, queue_name):
    rcon = get_redis_rcon()
    while True:
        try:
            lock_val = acquire_lock_with_timeout(rcon, lock_key)
            if not lock_val or lock_val is None:
                time.sleep(1)
                continue
            msg_info = {"config": config_file, "specs": seglist}
            push_msg(rcon, queue_name, msg_info)
            break
        except Exception as e:
            logger.error(f"RETRY commit task error: {e}")
            time.sleep(1)

def retry_multi_tasks(config_file):
    rcon = get_redis_rcon()
    msg_info = {"config": config_file}
    while True:
        lock_val = acquire_lock_with_timeout(rcon, MULTICLIPS_LOCK_KEY)
        if not lock_val or lock_val is None:
            time.sleep(1)
            continue

        push_msg(rcon, MULTICLIPS_PRIORITY_QUEUE, msg_info)
        release_lock(rcon, MULTICLIPS_LOCK_KEY, lock_val)
        break

def watch_multiclips_task(run_config, config_file):
    multi_info_path = run_config['multi_seg']["multi_info_path"]
    colls = os.listdir(multi_info_path)
    for coll in colls:
        coll_path = os.path.join(multi_info_path, coll)

def node_main(run_config, config_file):
    if 'multi_seg' in run_config and run_config['multi_seg']['enable'] == 'True':
        return watch_multiclips_task(run_config, config_file)
    work_temp_dir = None
    if run_config["method"] == "cli":
        work_temp_dir = os.path.dirname(config_file)

    if run_config['method'] == 'ripple':
        seg_config = run_config["preprocess"]
        car_name = seg_config['car']
        frame_path = seg_config['frames_path']
        task_id = run_config['ripples_platform_demand']['task_id']
        subfix = os.path.basename(frame_path)
        work_temp_dir = os.path.join("/data_autodrive/users/xuanliu7/work_tmp_ripple", f"{car_name}_{subfix}_{task_id}" )
    segment_path = run_config["preprocess"]["segment_path"]
    if not os.path.exists(segment_path):
        print("No Segment Path")
        return

    folders = os.listdir(segment_path)
    if len(folders) == 0:
        print("No Segment Folder")
        return 
    
    seg_config = run_config["preprocess"]
    rec_cfg = run_config["reconstruction"]
    tgt_seg_path = seg_config["segment_path"]
    if rec_cfg['enable'] != "True":
        print(f"{tgt_seg_path} skip reconstruct.")
        return 
    spec_clips = seg_config.get("spec_clips", None)
    task_queue = RECONSTRUCT_QUEUE
    task_lock_key = RECONSTRUCT_LOCK_KEY
    task_prior_queue = RECONSTRUCT_PRIORITY_QUEUE
    
    total_cnt = 0
    for seg_id in folders:
        if spec_clips is not None:
            go_on = False
            for clip in spec_clips:
                if clip in seg_id:
                    go_on = True
                    break
            if not go_on:
                continue         
        seg_folder = os.path.join(segment_path, seg_id)
        meta_file = os.path.join(seg_folder, "meta.json")
        with open(meta_file, "r") as meta_json:
            meta = json.load(meta_json)
            first_lidar_pose = np.array(meta["frames"][0]["lidar"]["pose"]).astype(
                np.float32
            )
            dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
            if (first_lidar_pose == dft_pose_matrix).all():
                print(f"{seg_id} not selected .")
                continue
            total_cnt += 1

    start_time = time.time()
    last_query_cnt = 0
    retry_flag = False
    while True:
        curr_time = time.time()
        timeout_time = curr_time - start_time
        if timeout_time > TIMEOUT_SET:
            print(f"Curr Task reach TIMOUT {TIMEOUT_SET}.")
            break

        curr_cnt = 0
        miss_segs = []
        for seg_id in folders:
            if spec_clips is not None:
                go_on = False
                for clip in spec_clips:
                    if clip in seg_id:
                        go_on = True
                        break
                if not go_on:
                    continue            
            status_flag = os.path.join(work_temp_dir, "reconstruct_status", f"{seg_id}:DONE")
            if os.path.exists(status_flag):
                curr_cnt += 1
            else:
                miss_segs.append(seg_id)

        if curr_cnt >= total_cnt:
            print(f"{work_temp_dir} reconstruct all[{total_cnt}] DONE.")
            break
        else:
            print(f"Curr query reconstruct status [{curr_cnt}/{total_cnt}] DONE")
            if curr_cnt > (total_cnt - 10):
                print(miss_segs)
                last_query_cnt += 1
                if last_query_cnt > 20 and not retry_flag:
                    retry_tasks(config_file, miss_segs, task_lock_key, task_prior_queue)
                    retry_flag = True
                
            time.sleep(60)
            continue
    logger.info(f"Total [{total_cnt}] segs wait for reconstruct results cost [{time.time() - start_time}]Seconds.")

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "reconstruct_status.log"))

    with open(config_file, "r") as fp:
        run_config = json.load(fp)

    node_main(run_config, config_file)
