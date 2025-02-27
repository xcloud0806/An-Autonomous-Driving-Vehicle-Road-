import shutil
import sys
sys.path.insert(0, "..")
from python import re_segment_clip
import os, sys
import json
import copy
from loguru import logger

copy_file = [
    "gnss.json",
    "vehicle.json",
    "pre_anno.json"
]
def main_gen_segs(origin_seg_path, dst_root:str):
    if not os.path.exists(origin_seg_path):
        print("[ERROR] origin seg path not exists")
        return
    
    if not os.path.exists(dst_root):
        os.makedirs(dst_root, mode=0o777, exist_ok=True)

    meta_json = os.path.join(origin_seg_path, "meta.json")
    meta = json.load(open(meta_json, "r"))
    seg_uid = meta['seg_uid']
    car_name = meta['car']
    gnss_json = os.path.join(origin_seg_path, "gnss.json")
    gnss_list = json.load(open(gnss_json, "r"))
    cut_segs, segs_dist = re_segment_clip(gnss_list)
    # print(cut_segs)
    if  len(cut_segs) <= 1:
        logger.info(f"{seg_uid} no need to re-segment")
        return
    
    def cut_list(lst:list, start_ts, end_ts):
        ret = []
        for idx, item in enumerate(lst):
            if 'lidar' in item:
                cur_ts = item['lidar']['timestamp']
                if cur_ts > start_ts and cur_ts < end_ts:
                    ret.append(idx)

        max_idx = ret[-1]
        min_idx = ret[0]
        return min_idx, max_idx
        

    for idx, seg_ts in enumerate(cut_segs):
        distance = segs_dist[idx]
        if distance < 250:
            logger.info(f"{seg_uid} cut_{idx} < 250m, distance: {distance}, skip.")
            continue

        start_ts = int(seg_ts[0])
        end_ts = int(seg_ts[-1])
        new_seg_uid = seg_uid.replace("_seg0", f"_seg{idx}")
        new_seg_path = os.path.join(dst_root, new_seg_uid)
        os.makedirs(new_seg_path, mode=0o777, exist_ok=True)
        
        new_seg_meta_file =  os.path.join(new_seg_path, "meta.json")
        new_meta = copy.deepcopy(meta)
        new_meta['distance'] = distance
        new_meta['time_interval'] = end_ts - start_ts
        new_meta['seg_uid'] = new_seg_uid
        logger.info(f"\t {seg_uid} cut_{idx} distance: {distance}, start_ts: {start_ts}, end_ts: {end_ts}, new seg_uid: {new_seg_uid}")

        for f in copy_file:
            src_file = os.path.join(origin_seg_path, f)
            dst_file = os.path.join(new_seg_path, f)
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)

        frames = meta['frames']
        raws = meta['raws']

        cut_idx = cut_list(frames, start_ts, end_ts)
        new_frames = copy.deepcopy(frames[cut_idx[0] : cut_idx[1]])
        cut_idx = cut_list(raws, start_ts, end_ts)
        new_raws = copy.deepcopy(raws[cut_idx[0] : cut_idx[1]])

        new_meta['frames'] = new_frames
        new_meta['raws'] = new_raws

        with open(new_seg_meta_file, "w") as fp:
            ss = json.dumps(new_meta, indent=4)
            fp.write(ss)
        


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("[ERROR] usage: python cut_cross_traj_seg.py origin_seg_path dst_root")
    #     exit(1)
    logger.add("./log/cut_cross_traj_seg.log", level="INFO")
    seg_path = '/data_cold2/origin_data/sihao_y7862/custom_seg/city_scene_test/20240408/sihao_y7862_20240408-10-28-49_seg0'
    main_gen_segs(seg_path, "/data_autodrive/users/brli/dev_raw_data/re_segment_clips")