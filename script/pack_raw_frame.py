import os, sys
import numpy as np
import json

curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(os.path.dirname(curr_path))
print(curr_dir)
sys.path.append(curr_dir)
from utils import lmdb_helper, CarMeta


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

class RawFrameDataset:
    def __init__(self, frame_path, calib_path) -> None:
        self.__frame_path = frame_path
        self.__car_meta = CarMeta()
        car_meta_json = os.path.join(calib_path, "car_meta.json")
        self.__car_meta.from_json(car_meta_json)

    def pack(self, dst_path):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path, mode=0o777, exist_ok=True)
        enable_cameras =  self.__car_meta.cameras
        enable_cameras.append('lidar')

        source = {}
        key_list = []
        frame_cnt = {}
        for sensor in enable_cameras:
            sensor_frame_path = os.path.join(self.__frame_path, sensor)
            files = os.listdir(sensor_frame_path)
            files.sort()
            frame_cnt[sensor] = 0
            for f in files:
                b = read_file(os.path.join(sensor_frame_path, f))
                ts = int(f.split('.')[0])
                k = f"{sensor}_{ts}"
                source[k] = b
                key_list.append(k)
                frame_cnt[sensor] += 1
        total_cnt = len(source)
        self.write_lmdb(dst_path, source, frame_cnt, key_list, total_cnt, "lmdb")

    def write_lmdb(self, submit_path, lmdb_bytes, frame_cnt, key_list, total_cnt, lmdb_name ):
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

if __name__ == "__main__":
    frame_path = "/data_cold2/origin_data/sihao_21pt6/luce_frame/20240407"
    calib_path = "/data_autodrive/auto/calibration/sihao_21pt6/20240315"
    clips = os.listdir(frame_path)
    clips.sort()
    for seg in clips:
        seg_frame_path =  os.path.join(frame_path, seg)
        handle = RawFrameDataset(seg_frame_path, calib_path)
        handle.pack(f"/data_autodrive/auto/custom/sihao_21pt6/clip_submit/data/20240407/{seg}")
