import os, sys
import json
import numpy as np
import time

TIMEOUT_SET=(3*86400) # three day
DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

def node_main(run_config, config_file):
    # work_temp_dir = os.path.dirname(config_file)
    task_id = run_config['ripples_platform_demand']['task_id']
    seg_config = run_config["preprocess"]
    car_name = seg_config['car']
    frame_path = seg_config['frames_path']
    subfix = os.path.basename(frame_path)
    work_temp_dir = os.path.join("/data_autodrive/users/xuanliu7/work_tmp_ripple", f"{car_name}_{subfix}_{task_id}" )
    segment_path = run_config["preprocess"]["segment_path"]
    if not os.path.exists(segment_path):
        print("No Segment Path")
        return
    spec_clips = seg_config.get("spec_clips", None)
    folders = os.listdir(segment_path)
    if len(folders) == 0:
        print("No Segment Folder")
        return 
    
    rec_cfg = run_config["reconstruction"]
    tgt_seg_path = seg_config["segment_path"]
    if rec_cfg['enable'] != "True":
        print(f"{tgt_seg_path} skip reconstruct.")
        return 
    
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
        print("reconstruct_status: success")
    else:
        print(f"Curr query reconstruct status [{curr_cnt}/{total_cnt}] ING.")
        print("reconstruct_status: process")
        if curr_cnt > (total_cnt - 10):
            print(miss_segs)

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)

    node_main(run_config, config_file)
