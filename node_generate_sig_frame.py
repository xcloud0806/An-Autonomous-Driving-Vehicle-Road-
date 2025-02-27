import numpy as np
import json
from datetime import datetime
import traceback as tb
import os,sys
from loguru import logger
import copy

RECORD_FILE = "record.json"
META_FILE = "car_meta.json"
MAX_FRAMES = 120
PICK_INTERVAL = 5 # unit 0.1 second
PICK_DISTANCE = 0.1 # unit meter
EXPAND_FRAMES = 50 # parking behavior expand frame data
EXPAND_DISTANCE = 10 # unit is meter

def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def gen_clip_sig_frames(gnss:dict, vehicle:dict, seg:dict, mode:str):
    frames = seg['frames']
    seg_id = seg['seg_uid']
    sig_frames = []
    frame_cnt = len(frames)
    # only use middle 400M for obstacle anno
    start_idx = int(frame_cnt * 0.25)
    end_idx = int(frame_cnt * 0.75)
    if mode == 'luce' or mode == 'test':
        start_idx = int(frame_cnt * 0.05)
        end_idx = int(frame_cnt * 0.95)

    cnt = 0
    pre_speed = []
    for f_idx in range(start_idx, end_idx):
        # use frame every 0.5s
        sample_interval = PICK_INTERVAL
        if f_idx % sample_interval != 0:
            continue
        # total frame not bigger than 80
        if cnt > MAX_FRAMES:
            break
        frame = frames[f_idx]
        frame_gnss = gnss[frame['gnss']]
        speed = float(frame_gnss['speed'])
        # skip static frame
        if speed == 0:
            frame_veh = vehicle[frame['vehicle']]
            speed = float(frame_veh['vehicle_spd'])
        if speed < 0.01 and sum(pre_speed) < 0.1:
            continue

        if len(pre_speed) > 10:
            pre_speed.pop(0)
        pre_speed.append(speed)
        enable_cameras = seg['cameras']
        sig_frame = {}
        lidar = frame['lidar']
        sig_frame['lidar'] = str(lidar['timestamp'])
        sig_frame['frame_idx'] = f_idx
        images = frame['images']
        for cam in enable_cameras:
            if cam in images:
                cam_img = images[cam]                
                sig_frame[cam] = str(cam_img['timestamp'])
        # frame_info = convert_frame(cnt, frame, meta, dst_path, cameras)
        cnt += 1
        sig_frames.append(sig_frame)
    seg['key_frames'] = sig_frames
    logger.info(f"......\t{seg_id} pick {cnt} significant frames.")


def gen_clip_sig_frames_hpp_luce(gnss:dict, vehicle:dict, seg:dict, mode:str):
    frames = seg['frames']
    seg_id = seg['seg_uid']
    sig_frames = []
    sig_frame_lost_cnt = 0
    frame_cnt = len(frames)
    start_idx = int(frame_cnt * 0.05)
    end_idx = int(frame_cnt)
    cnt = 0
    for f_idx in range(start_idx, end_idx):
        frame = frames[f_idx]
        pre_ts = int(frame['lidar']['timestamp'])
        
        enable_cameras = seg['cameras']
        sig_frame = {}        
        sig_frame['lidar'] = str(pre_ts)
        sig_frame['frame_idx'] = f_idx
        images = frame['images']
        img_lost = False
        for cam in enable_cameras:
            if cam in images:
                cam_img = images[cam]                
                sig_frame[cam] = str(cam_img['timestamp'])
            else:
                img_lost = True
        if img_lost:
            sig_frame_lost_cnt += 1
        cnt += 1
        if len(sig_frame) > 0:
            sig_frames.append(sig_frame)

    logger.info(f"......\t{seg_id} pick {cnt} significant frames.")
    seg['key_frames'] = sig_frames
    seg['key_frames_lost'] = sig_frame_lost_cnt

def gen_clip_sig_frames_hpp(gnss:dict, vehicle:dict, seg:dict, mode:str):
    frames = seg['frames']
    seg_id = seg['seg_uid']
    sig_frames = []
    sig_frame_lost_cnt = 0
    frame_cnt = len(frames)
    start_idx = int(frame_cnt * 0.05)
    end_idx = int(frame_cnt)
    
    back_list = []
    for f_idx in range(frame_cnt):
        frame = frames[f_idx]
        frame_veh = vehicle[str(frame['vehicle'])]
        gear = int(frame_veh['gear']) # 2: backward 4:forward
        if gear == 2:
            back_list.append(f_idx)

    # gen start/end index by expand distance    
    if len(back_list) > 0:
        ss_idx = back_list[0]
        ss_frame = frames[ss_idx]
        ss_pose = np.array(ss_frame['lidar']['pose'])
        se_idx = back_list[-1]
        se_frame = frames[se_idx]
        se_pose = np.array(se_frame['lidar']['pose'])
        for f_idx in range(0, ss_idx):
            frame = frames[f_idx]
            frame_pose = frame['lidar']['pose']
            dist = np.linalg.norm(np.array(frame_pose)[:3, 3] - ss_pose[:3, 3])
            if dist < EXPAND_DISTANCE:
                logger.debug(f"....{seg_id} set start index to {f_idx}/{frame_cnt}")
                start_idx = f_idx
                break


        for f_idx in range(se_idx, frame_cnt - 1):
            frame = frames[f_idx]
            frame_pose = frame['lidar']['pose']
            dist = np.linalg.norm(np.array(frame_pose)[:3, 3] - se_pose[:3, 3])
            if dist < EXPAND_DISTANCE:
                logger.debug(f"....{seg_id} set end index to {f_idx}/{frame_cnt}")
                end_idx = f_idx
                break
    
    # gen start/end index by expand frame
    # if len(back_list) > 0:
    #     if len(back_list) < EXPAND_FRAMES:
    #         logger.info(f"{seg_id} back_list too short...")
    #     else:
    #         logger.info(f"{seg_id} has reverse parking with [{len(back_list)}] frames")
    #         start_idx = back_list[0] - EXPAND_FRAMES if back_list[0] > EXPAND_FRAMES else 0
    #         end_idx = back_list[-1] + EXPAND_FRAMES if back_list[-1] < frame_cnt - EXPAND_FRAMES else frame_cnt

    cnt = 0
    pre_speed = []
    pre_ts = 0
    pre_pose = None
    for f_idx in range(start_idx, end_idx):
        frame = frames[f_idx]
        frame_veh = vehicle[str(frame['vehicle'])]
        speed = float(frame_veh['vehicle_spd'])
        if len(pre_speed) > 10:
            pre_speed.pop(0)
        pre_speed.append(speed)
        if speed < 0.01 and sum(pre_speed) < 0.1: # skip static frame
            continue

        enable_cameras = seg['cameras']
        sig_frame = {}
        if pre_ts == 0:
            pre_ts = int(frame['lidar']['timestamp'])
            pre_pose = frame['lidar']['pose']
            sig_frame['lidar'] = str(pre_ts)
            sig_frame['frame_idx'] = f_idx
            images = frame['images']
            for cam in enable_cameras:
                if cam in images:
                    cam_img = images[cam]                
                    sig_frame[cam] = str(cam_img['timestamp'])
            cnt += 1
        else:
            curr_ts = int(frame['lidar']['timestamp'])
            curr_pose = frame['lidar']['pose']
            curr_dist = np.linalg.norm(np.array(curr_pose)[:3, 3] - np.array(pre_pose)[:3, 3])
            if curr_dist > PICK_DISTANCE:
                # logger.info(f"No.{cnt} <->{curr_ts} + {curr_dist} ")
                pre_ts = curr_ts
                pre_pose = curr_pose
                sig_frame['lidar'] = str(curr_ts)
                sig_frame['frame_idx'] = f_idx
                images = frame['images']
                img_lost = False
                for cam in enable_cameras:
                    if cam in images:
                        cam_img = images[cam]                
                        sig_frame[cam] = str(cam_img['timestamp'])
                    else:
                        img_lost = True
                if img_lost:
                    sig_frame_lost_cnt += 1
                cnt += 1
        if len(sig_frame) > 0:
            sig_frames.append(sig_frame)

    logger.info(f"......\t{seg_id} pick {cnt} significant frames.")
    seg['key_frames'] = sig_frames
    seg['key_frames_lost'] = sig_frame_lost_cnt

def node_main(run_config):
    seg_config = run_config["preprocess"]
    seg_mode =  seg_config["seg_mode"]
    frames_path = seg_config['frames_path']
    tgt_seg_path = seg_config["segment_path"]

    spec_clips = seg_config.get("spec_clips", None)

    seg_root_path = tgt_seg_path
    if not os.path.exists(seg_root_path):
        logger.error(f"{seg_root_path} NOT Exist...")
        sys.exit(1)
    seg_names = os.listdir(seg_root_path)
    seg_names.sort()

    for segid in seg_names:
        if spec_clips is not None:
            go_on = False
            for clip in spec_clips:
                if clip in segid:
                    go_on = True
                    break
            if not go_on:
                continue
        logger.info(f"Processing {segid}...")
        seg_path = os.path.join(seg_root_path, segid)
        meta_file = os.path.join(seg_root_path, segid, "updated_meta.json")
        if not os.path.exists(meta_file):
            meta_file = os.path.join(seg_root_path, segid, "meta.json")
        if not os.path.exists(meta_file):
            logger.error(f"{meta_file} NOT Exist...")
            sys.exit(1)

        gnss_file = os.path.join(seg_path, "gnss.json")
        vehi_file = os.path.join(seg_path, "vehicle.json")
        if not os.path.exists(gnss_file) or not os.path.exists(vehi_file):
            logger.error(f"{gnss_file} or {vehi_file} NOT Exist...")
            sys.exit(1)

        try:
            meta = json.load(open(meta_file, 'r'))
            gnss = json.load(open(gnss_file, 'r'))
            vehicle = json.load(open(vehi_file, 'r'))
        except Exception as e:
            logger.exception(f"Caught an exception of type {type(e).__name__}: {e}")
            sys.exit(1)

        if seg_mode == "hpp":
            gen_clip_sig_frames_hpp(gnss, vehicle, meta, seg_mode)
        elif seg_mode == 'hpp_luce':
            gen_clip_sig_frames_hpp_luce(gnss, vehicle, meta, seg_mode)
        else:
            gen_clip_sig_frames(gnss, vehicle, meta, seg_mode)
        
        try:
            with open(meta_file, "w") as fp:
                json.dump(meta, fp, indent=4, default=dump_numpy)
        except Exception as e:
            logger.exception(f"Caught an exception of type {type(e).__name__}: {e}")
            sys.exit(1)
            
if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)
    
    work_temp_dir = os.path.dirname(config_file)
    logger.add(f"{work_temp_dir}/node_gen_sig_frame.log", rotation="10 MB")

    with open(config_file, 'r') as fp:
        run_config = json.load(fp)
    node_main(run_config)