from node_create_config import node_main, parse_luce_xls, handle_luce_clip

import argparse
import os, sys
import json
from multiprocessing import pool
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Automatically generate run configurations from template configurations")

    parser.add_argument('--template', '-t', type=str, required=True)
    parser.add_argument('--excel', '-i', type=str, required=True )
    parser.add_argument('--subfix', '-d', type=str, required=True)
    parser.add_argument('--car', '-c', type=str,required=True)
    parser.add_argument('--calib_date', type=str, required=True)
    parser.add_argument('--config_file', '-o', type=str, default="./utils/luce_config.json")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    luce_xls = args.excel
    car_name = args.car
    subfix = args.subfix
    frame_path = os.path.join(f"/data_cold2/origin_data/{car_name}/luce_frame/{subfix}")
    calib_date = args.calib_date
    calib_path = os.path.join("/data_autodrive/auto/calibration/", car_name, calib_date)
    template_cfg_file = args.template
    config_file = args.config_file
    if not os.path.exists(template_cfg_file):
        print(f"Template Config File lost.")
        sys.exit(1)
    
    run_cfg = json.load(open(template_cfg_file, "r"))
    clips = parse_luce_xls(luce_xls)
    clip_ids = list(clips.keys())
    origin_luce_frame_root = None
    for clip_id, clip_info in clips.items():
        # handle_luce_clip(clip_id, clip_info, frame_path)
        # handle_clip(clip_id, clip_info, frame_path)
        # pool.apply_async(handle_luce_clip, args=(clip_id, clip_info, frame_path,))
        print(f"clip_id: {clip_id}")
        clip_path = clip_info["clip_path"]
        if origin_luce_frame_root is None:
            origin_luce_frame_root = os.path.dirname(clip_path)
        tag_info = dict()
        clip_tag_json = os.path.join(clip_path, "tag_info.json")
        if os.path.exists(clip_tag_json):
            with open(clip_tag_json, "r") as fp:
                tag_info = json.load(fp)

        tag_info["luce_info"] = clip_info
        with open(os.path.join(clip_path, "tag_info.json"), "w") as fp:
            ss = json.dumps(tag_info)
            fp.write(ss)

    init_cfg = {
        "frames_path": origin_luce_frame_root,
        "calibration_path": calib_path,
        "target_anno_output": "/data_autodrive/auto/luce/",
        "car_name": car_name,
        # "seg_mode": "luce",
        'method': 'cli',
        "spec_clips": clip_ids,
        "luce_excel_path": luce_xls,
    }

    config = node_main(init_cfg, run_cfg)
    with open(config_file, "w") as fp:
        ss = json.dumps(config, indent=4)
        fp.write(ss)