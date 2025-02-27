from pathlib import Path
import os
import sys
import json
import numpy as np

root_path = "/data_cold2/origin_data"
type_pairs = [
    ("common_frame", "common_seg"),
    ("custom_frame", "custom_seg")
]

cars = [
    "chery_04228",
    "sihao_19cp2",
    "sihao_2xx71",
    "chery_10034",
    "chery_53054",
    "chery_b8615",
    "aion_d77052",
    "aion_d77360",
    "chery_13484",
    "chery_06826",
    "sihao_1482",
    "sihao_27en6",
    "sihao_0fx60",
    "sihao_8j998",
    "sihao_47465",
    "chery_32694",
    "sihao_36gl1",
    "sihao_21pt6",
    "sihao_7xx65",
    "sihao_y7862",
    "sihao_47466",
    "sihao_37xu2",
    "sihao_35kw2",
    "sihao_23gc9",
    "sihao_72kx6",
]

DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

total_segs = 0
unused_cnt = 0
unused_segs = {}

for car in cars:
    car_seg_root = Path(os.path.join(root_path, car, "common_seg"))
    print(car_seg_root)
    if not car_seg_root.exists():
        continue
    dates = [x for x in car_seg_root.iterdir() if x.is_dir()]
    for date in dates:
        print(f"{date}")
        # date_path = car_seg_root / date
        # if not date_path.exists():
            # continue
        segs = [x for x in date.iterdir() if x.is_dir() ]
        seg_total_num = len(segs)
        total_segs += seg_total_num
        for seg in segs:
            undist_pcd_path = seg / "correct_pointclouds"
            if not undist_pcd_path.exists():
                unused_cnt += 1
                seg_id = seg.name
                unused_segs[seg_id] = seg
                print(f"\t [{unused_cnt}/{total_segs}] - {seg_id}")
                continue

            # meta_path = seg / "meta.json"
            # if not meta_path.exists():
            #     continue
            # with open(meta_path, "r") as f:
            #     seg_info = json.load(f)
            #     seg_uid = seg_info['seg_uid']
            #     first_lidar_pose = np.array(seg_info["frames"][0]["lidar"]["pose"]).astype(
            #         np.float32
            #     )
            #     dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
            #     if (first_lidar_pose == dft_pose_matrix).all():
            #         unused_segs[seg_uid] = seg

print(total_segs)
print(len(unused_segs))
with open("unused_segs.json", "w") as f:
    json.dump(unused_segs, f)
        # for seg in segs:

