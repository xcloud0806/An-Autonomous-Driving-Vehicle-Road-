from node_create_config import node_main
from version import fetch_latest_tool

import argparse
import os, sys
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Automatically generate run configurations from template configurations")

    parser.add_argument('--template', '-t', type=str, required=True)
    parser.add_argument('--frames', '-i', type=str, required=True )
    parser.add_argument('--car', '-c', type=str,required=True)
    parser.add_argument('--calib_date', type=str, required=True)
    parser.add_argument('--config_file', '-o', type=str, default="run.cfg")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    try:
        fetch_latest_tool()
    except:
        print("Failed to fetch latest tool.")

    frame_path = args.frames
    car_name = args.car
    calib_date = args.calib_date
    calib_path = os.path.join("/data_autodrive/auto/calibration/", car_name, calib_date)
    template_cfg_file = args.template
    config_file = args.config_file
    if not os.path.exists(template_cfg_file):
        print(f"Template Config File lost.")
        sys.exit(1)
    
    try:
        run_cfg = json.load(open(template_cfg_file, "r"))

        init_cfg = {
            "frames_path": frame_path,
            "calibration_path": calib_path,
            "target_anno_output": "/data_autodrive/auto/custom/",
            "car_name": car_name,
            "method": "cli"
        }

        config = node_main(init_cfg, run_cfg)
        with open(config_file, "w") as fp:
            ss = json.dumps(config)
            fp.write(ss)
    except Exception as e:
        print(f"Caught an exception of type {type(e).__name__}: {e}")
        sys.exit(1)