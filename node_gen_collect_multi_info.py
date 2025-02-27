# -*- coding: utf-8 -*-
from datetime import datetime
import os, sys
import json
import numpy as np
import string
from loguru import logger
import shutil

curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_path)
sys.path.append(f"{curr_dir}/lib/python3.8/site_packages")
import odometry

# sample multi_info file
"""{
	"20240410-17-41-22-9CYjaosB": {
		"clips_path": [

        ],
		"main_clip_path": [
			"/data_cold2/origin_data/sihao_27en6/custom_seg/frwang_chadaohuichu/night/20240314/sihao_27en6_20240314-23-12-47_seg0"
		],
		"mode": 1,
		"reconstruct_path": "/data_cold2/origin_data/sihao_27en6/custom_coll/frwang_chadaohuichu/20240410/20240410-17-41-22-9CYjaosB",
		"version": "2.0",
		"reconstruct_status": "failed",
        "increment_enable": true,
        "increment_status": false,
        "multi_odometry_status": false
	}
}"""

def generate_random_string(length=8):
    alphabet = string.ascii_letters + string.digits
    return ''.join(np.random.choice(list(alphabet), length))

def parse_multi_infos(multi_info_root):
    if not os.path.exists(multi_info_root):
        logger.error("multi_info root not exist: {}".format(multi_info_root))
        return
    multi_colls = os.listdir(multi_info_root)
    logger.info(f"Parsing {multi_info_root}, total {len(multi_colls)} colls")
    ret = {}
    for multi_coll in multi_colls:
        multi_coll_path = os.path.join(multi_info_root, multi_coll)
        coll_id = multi_coll
        coll_info = os.path.join(multi_coll_path, "multi_info.json")
        if not os.path.exists(coll_info):
            logger.error("multi_info not exist: {}".format(coll_info))
            continue
        coll_reconstruct_path = os.path.join(multi_coll_path, "multi_reconstruct")
        if not os.path.exists(coll_reconstruct_path):
            logger.error("multi_reconstruct not exist: {}".format(coll_reconstruct_path))
            continue

        with open(coll_info, "r") as f:
            coll_info = json.load(f)
            if coll_id not in coll_info:
                logger.error("coll_id not in multi_info: {}".format(coll_id))
                continue
            info = coll_info[coll_id]
            if "reconstruct_status" not in info or info["reconstruct_status"] != "success":
                logger.error("reconstruct status not success: {}".format(coll_id))
                continue
            clip_paths = info["main_clip_path"]
            for clip_path in clip_paths:
                clip_id = os.path.basename(clip_path)
                ret[clip_id] = [clip_path, coll_id, coll_reconstruct_path]
    return ret

def node_main(run_config, clip_multi_info:dict):
    seg_config = run_config["preprocess"]
    tgt_seg_path = seg_config["segment_path"]
    dst_coll_root = tgt_seg_path.replace("custom_seg", "custom_coll")
    dst_coll_root = dst_coll_root.replace("common_seg", "common_coll")    
    logger.info(f"start to generate multi_info file {dst_coll_root}")
    car_name = seg_config["car"]

    if not os.path.exists(tgt_seg_path):
        logger.error("segment path not exist: {}".format(tgt_seg_path))
        return

    night_segs = os.listdir(tgt_seg_path)
    for night_seg in night_segs:
        night_seg_path = os.path.join(tgt_seg_path, night_seg)
        if not os.path.isdir(night_seg_path):
            continue

        night_seg_meta_json = os.path.join(night_seg_path, "meta.json") 
        if not os.path.exists(night_seg_meta_json):
            logger.error("night seg meta json not exist: {}".format(night_seg_meta_json))
            continue
        night_meta = json.load(open(night_seg_meta_json, "r"))
        if 'related_seg' not in night_meta:
            continue
        logger.info(f"generate multi_info for {night_seg}")
        related_day_seg = night_meta["related_seg"]

        # generate 8 rand string
        rand_str = generate_random_string()
        now = datetime.now()
        formated_time = now.strftime("%Y%m%d-%H-%M-%S")
        multi_info_id = "{}-{}".format(formated_time, rand_str)
        coll_path = os.path.join(dst_coll_root, multi_info_id)
        if not os.path.exists(coll_path):
            os.makedirs(coll_path, mode=0o777, exist_ok=True)
        multi_info_file = os.path.join(coll_path, "multi_info.json")
        logger.info(f"generate multi_info file {multi_info_file}")

        multi_info = {
            multi_info_id: {
                "clips_path": [related_day_seg],
                "main_clip_path": [night_seg_path],
                "mode": 1,
                "reconstruct_path": coll_path,
                "version": "2.0",
                "increment_enable": True,
                "increment_status": False,
                "multi_odometry_status": False
            }
        }        
        
        if clip_multi_info is None:
            with open(multi_info_file, "w") as fp:
                json.dump(multi_info, fp, indent=4)
            odometry.MultiIncrementConstruct(coll_path)
            related_day_seg_recon = os.path.join(related_day_seg, "reconstruct")
            night_seg_recon = os.path.join(night_seg_path, "reconstruct")
            shutil.copytree(related_day_seg_recon, night_seg_recon, dirs_exist_ok=True)
        else:   
            related_day_seg_id = os.path.basename(related_day_seg)         
            if related_day_seg_id in clip_multi_info:
                related_day_clip_path = clip_multi_info[related_day_seg_id][0]
                coll_id = clip_multi_info[related_day_seg_id][1]
                multi_meta_file = os.path.join(related_day_clip_path, f"{coll_id}_multi_meta.json")
                multi_info[multi_info_id]["clips_path_meta"] = [multi_meta_file]
                with open(multi_info_file, "w") as fp:
                    json.dump(multi_info, fp, indent=4)
                odometry.MultiIncrementConstruct(coll_path)
                related_day_seg_recon = clip_multi_info[related_day_seg_id][2]
                night_seg_recon = os.path.join(night_seg_path, "reconstruct")
                shutil.copytree(related_day_seg_recon, night_seg_recon, dirs_exist_ok=True)
            else:
                logger.error(f"related_day_seg {related_day_seg_id} not in clip_multi_info")
                continue

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)
    
    clip_multi_info = None
    if len(sys.argv) > 2:
        day_multi_info_root = sys.argv[2]
        clip_multi_info = parse_multi_infos(day_multi_info_root)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "node_gen_multi_info.log"))

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    node_main(run_config, clip_multi_info)

