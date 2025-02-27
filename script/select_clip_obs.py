import json
import os
import pandas as pd
import shutil

def main(obs_path, dst_obs_path, xls_file):
    df = pd.read_excel(xls_file, engine='openpyxl')
    seg_avi_list = df.iloc[:, 1].tolist()
    status = df.iloc[:, 3].tolist()

    seg_list = [item for item in seg_avi_list]
    st_list = [item == "true" for item in status]

    for i, seg in enumerate(seg_list):
        # handle error seg
        status = st_list[i]
        if status: 
            continue

        print(f"Handle {seg} <-> {st_list[i]}")

        src_seg_obs = os.path.join(obs_path, seg)
        dst_seg_obs = os.path.join(dst_obs_path, seg)
        if os.path.exists(src_seg_obs) :
            os.makedirs(dst_seg_obs, mode=0o775, exist_ok=True)
            os.system(f"mv {src_seg_obs}/* {dst_seg_obs}")

if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='Generate obstacle for anno')
        parser.add_argument('--input', '-i', type=str, help='Input directory containing obstacles')
        parser.add_argument('--date', '-d', type=str, required=True)
        # parser.add_argument("--output", "-d", type=str, required=True)
        parser.add_argument("--xls", "-x", type=str, required=True)

        return parser.parse_args()
    
    args = parse_args()
    obs_path = os.path.join(args.input, "clip_obstacle", args.date)
    dst_obs_path = os.path.join(args.input, "clip_obstacle_sl", args.date)
    os.makedirs(dst_obs_path, mode=0o775, exist_ok=True)
    main(obs_path, dst_obs_path, args.xls)