from multiprocessing import Pool
from tqdm import tqdm
from loguru import logger

import os, sys
sys.path.append("../")
import json
from utils import lmdb_helper, prepare_infos
import shutil
import numpy as np

ANNO_INFO_JSON = "annotation.json"
ANNO_INFO_CALIB_KEY = "calib"
ANNO_INFO_INFO_KEY = "clip_info"
ANNO_INFO_LANE_KEY = "lane"
ANNO_INFO_OBSTACLE_KEY = "obstacle"
ANNO_INFO_OBSTACLE_STATIC_KEY = "obstacle_static"
ANNO_INFO_PAIR_KEY = "pair_list"
ANNO_INFO_RAW_PAIR_KEY = "raw_pair_list"
ANNO_INFO_POSE_KEY = "pose_list"
TEST_ROADS_GNSS = "test_roads_gnss_info.json"
DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

def read_file(pcd_img_path):
    with open(pcd_img_path, "rb") as f:
        file_bytes = f.read()
        return file_bytes

def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def gen_seg_lmdb(
    meta: dict,
    enable_cams: list,
    enable_bpearls: list,
    submit_path: str,
    max_frame=1200
):
    def gen_lmdb_bytes(meta, frame_cnt, start_idx, end_idx):
        frames = meta["frames"]
        total_cnt = 0
        key_list = []
        seg_frame_path = meta["frames_path"]
        lmdb_bytes = {}

        if 'inno_lidar' not in frame_cnt:
            frame_cnt['inno_lidar'] = 0
        inno_lidar_path = os.path.join(seg_frame_path, "inno_lidar")
        inno_lidars = []
        inno_lidar_tss = []
        if os.path.exists(inno_lidar_path):
            for f in os.listdir(inno_lidar_path):
                if f.endswith(".pcd"):
                    ts = int(f[:-4])
                    inno_lidars.append(os.path.join(inno_lidar_path, f))
                    inno_lidar_tss.append(ts)
        inno_ts_arr = np.array(inno_lidar_tss)

        for i, f in enumerate(frames):
            # 拷贝雷达数据
            if i < start_idx or i >= end_idx:
                continue
            
            innos = {}
            if len(enable_bpearls) > 0:
                bpearl_keys = list()
                if "innos" in f:
                    innos = f["innos"]
                    if len(innos) > 0:
                        bpearl_keys.extend(list(innos.keys()))

            b = 'inno_lidar'
            if b not in bpearl_keys:
                lidar_ts = int(f['lidar']['timestamp']  )                  
                match_idx = abs(inno_ts_arr - lidar_ts - 25).argmin()
                inno_lidar_ts = inno_lidar_tss[match_idx]
                inno_lidar_pcd = inno_lidars[match_idx]
                if not os.path.exists(inno_lidar_pcd):
                    logger.error(f"{inno_lidar_pcd} not exist!")
                    continue
                b_bytes = read_file(inno_lidar_pcd)
                b_key = f"inno_lidar_{inno_lidar_ts}"
                lmdb_bytes[b_key] = b_bytes
                key_list.append(b_key)
                frame_cnt[b] += 1
                total_cnt += 1
            else:
                b_path = innos[b]["path"]
                b_ts = innos[b]['timestamp']
                if b_ts == 0:
                    continue
                b_key = f"{b}_{innos[b]['timestamp']}"                
                b_src = os.path.join(seg_frame_path, b_path)
                if not os.path.exists(b_src):
                    logger.error(f"{b_src} not exist!")
                    continue
                b_bytes = read_file(b_src)                
                lmdb_bytes[b_key] = b_bytes
                key_list.append(b_key)
                frame_cnt[b] += 1
                total_cnt += 1

        return lmdb_bytes, total_cnt, key_list, frame_cnt
    
    logger.info(f"Prepare {meta['seg_uid']} to Pack {submit_path}")    
    def write_lmdb(submit_path, lmdb_bytes, frame_cnt, key_list, total_cnt, lmdb_name ):
        cache_size = 50 * 1024 * 1024 * 1024
        _lmdb_data_path = submit_path + f"/{lmdb_name}"
        if os.path.exists(_lmdb_data_path):
            os.system(f"rm -rf {_lmdb_data_path}")
        lmdb_handle = lmdb_helper.LmdbHelper(_lmdb_data_path, tmp_size=cache_size)
        lmdb_info = {}    

        lmdb_handle.write_datas(lmdb_bytes)
        lmdb_size, lmdb_hash = lmdb_handle.cacl_hash(submit_path +  f"/{lmdb_name}")
        lmdb_info["lmdb_size"] = lmdb_size
        lmdb_info["lmdb_hash"] = lmdb_hash
        lmdb_info["frame_cnt"] = frame_cnt
        lmdb_info["total_cnt"] = total_cnt
        lmdb_info["key_list"] = key_list
        with open(submit_path + f"/{lmdb_name}_info.json", "w") as f:
            json.dump(lmdb_info, f)
    
    frame_total_cnt = len(meta['frames'])
    if frame_total_cnt < (max_frame + 1):
        frame_cnt = {}
        lmdb_bytes, total_cnt, key_list, frame_cnt = gen_lmdb_bytes(meta, frame_cnt, 0, frame_total_cnt)
        write_lmdb(submit_path, lmdb_bytes, frame_cnt, key_list, total_cnt, "lmdb")
    else:        
        total_lmdb_clip_cnt = int(frame_total_cnt / max_frame) + 1
        logger.info(f"\tcut {meta['seg_uid']} to {total_lmdb_clip_cnt} LMDB_PACKS.")
        for idx in range(total_lmdb_clip_cnt):
            _lmdb_name = f"lmdb_{idx}"
            logger.info(f"\t{_lmdb_name} start......")
            start_idx = idx * max_frame
            end_idx = (idx + 1) * max_frame if (idx + 1) * max_frame < frame_total_cnt else frame_total_cnt
            frame_cnt = {}
            lmdb_bytes, total_cnt, key_list, frame_cnt = gen_lmdb_bytes(meta, frame_cnt, start_idx, end_idx)
            write_lmdb(submit_path, lmdb_bytes, frame_cnt, key_list, total_cnt, _lmdb_name) 

if __name__ == "__main__":
    inno_file = "/data_autodrive/users/brli/dev_raw_data/inno_segs_2023.json"
    data_root = "/data_cold2/origin_data/sihao_37xu2/common_seg/"
    anno_root = "/data_autodrive/auto/custom/sihao_37xu2/inno_lidar_data/"
    cnt = 0
    fp = open(inno_file, "r")
    inno_list = json.load(fp)
    for k, v in inno_list.items():
        if k in [
            "yingyan.sihao_37xu2.20231021",
            "yingyan.sihao_37xu2.20231022",
            "yingyan.sihao_37xu2.20231023",
            "yingyan.sihao_37xu2.20231025",
            "yingyan.sihao_37xu2.20231026",
            "yingyan.sihao_37xu2.20231027",
            "yingyan.sihao_37xu2.20231028",
            "yingyan.sihao_37xu2.20231029",
        ]:
            continue
        task, car, subfix = k.split(".")
        segs = v
        for seg in segs:
            print(f"{k} <-> {seg}")
            cnt += 1
            seg_path = os.path.join(data_root, subfix, seg)
            anno_seg_path = os.path.join(anno_root, subfix, seg)
            os.makedirs(anno_seg_path, exist_ok=True)
            meta_json = os.path.join(seg_path, "meta.json")
            if not os.path.exists(meta_json):
                logger.error(f"{seg} not exist")
                continue
            fp = open(meta_json, "r")
            meta = json.load(fp)
            gen_seg_lmdb(meta, [], ["inno_lidar"], anno_seg_path, 3600)
            fp.close()

    # print(cnt)
