import lmdb
import cv2
import numpy as np
from .pcd_io import load_pcd_from_bytes
import json
import time
import glob
from pathlib import Path
import os
import argparse
import tqdm
import shutil
class LmdbHelper:
    def __init__(self, lmdb_path, read_only=False, tmp_size=1099511627776):
        if read_only:            
            self.env = lmdb.open(lmdb_path, readonly=True, map_size=tmp_size)
        else:
            self.env = lmdb.open(lmdb_path, map_size=tmp_size)
        self.read_only = read_only

    def __del__(self):
        self.env.close()

    def write_data(self, key: str, value):
        txn =  self.env.begin(write=True)
        key_bytes = key.encode()
        txn.put(key_bytes, value)
        txn.commit()

    def write_datas(self,  datas: dict):
        txn =   self.env.begin(write=True)
        for key, value in datas.items():
            key_bytes = key.encode()
            txn.put(key_bytes, value)
        txn.commit()

    def read_img(self, key: str):
        content = self.read_data(key)
        if content=="None":
            return "None"
        # img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if content is not None:
            image = np.frombuffer(content, dtype=np.uint8)
            arr = cv2.imdecode(image, cv2.IMREAD_COLOR)
        else:
            arr = None
        return arr

    def read_data(self, key: str):
        txn = self.env.begin(write=False)
        value = txn.get(str(key).encode(),"None")
        return value

    def read_pcd(self, key: str):
        content = self.read_data(key)
        if content=="None":
            return "None"
        pcd, fields, sizes, data_types = load_pcd_from_bytes(content)
        # print(len(pcd), pcd.keys())
        def convert(data:dict):
            array_lst = []
            for k, v in data.items():
                v_array = np.array(v, dtype=np.float64).reshape((-1, 1)) 
                array_lst.append(v_array)
            return np.concatenate(array_lst, axis=1)
        lidar_data = convert(pcd)
        return lidar_data

def read_json(path):
    with open(path,encoding="utf-8") as f:
        data = json.load(f)
    return data
def read_pcd(lidar_path):
    import pypcd.pypcd as pypcd
    pc = pypcd.PointCloud.from_path(lidar_path)
    np_x = (np.array(pc.pc_data['x'], dtype=np.float64)).astype(np.float64)
    np_y = (np.array(pc.pc_data['y'], dtype=np.float64)).astype(np.float64)
    np_z = (np.array(pc.pc_data['z'], dtype=np.float64)).astype(np.float64)
    np_i = (np.array(pc.pc_data['intensity'], dtype=np.float64)).astype(np.float64) 
    lidar_points = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
    lidar_points = lidar_points[(np.isnan(lidar_points)==False)].reshape(-1,4)
    return lidar_points

def read_single_clips(json_path,sample_ratio):
    path = os.path.dirname(json_path)
    lmdb_path = os.path.join(path,"lmdb")
    lmdb_dir_split = os.path.join(path,"lmdb_0")
    if os.path.exists(lmdb_path):
        data_loader = LmdbHelper(lmdb_path,read_only=False)
    elif os.path.exists(lmdb_dir_split):
        data_loader = []
        num = len(glob.glob(os.path.join(path,"lmdb_*_info.json")))
        for m in range(num):
            data_loader.append(LmdbHelper(path + "/lmdb_{}".format(m),read_only=False))
    else:
        raise FileExistsError
    clip_info = read_json(json_path)
    #data_loader.read_pcd(clip_info["pair_list"][1]["lidar"])
    tokens_all = [i['lidar'] for i in clip_info['pair_list']]
    tokens = tokens_all[::sample_ratio]
    tokens.extend(tokens_all[::-1][::sample_ratio])
    cam_names = list(clip_info["pose_list"][0].keys())
    cam_names = [i for i in cam_names if i !="lidar"]
    # lidar_dir = "/train30/cv2/permanent/brli/ftp/sihao_19cp2/20231223/sihao_19cp2_20231223-09-26-04_seg9/lidar/"
    # path2 = glob.glob(lidar_dir+"/*.pcd")
    # tokens2 = [Path(i).stem for i in path2]
    for i,t in enumerate(tokens):
        pcd_key = f"lidar_{t}"
        start = time.time()
        if isinstance(data_loader,list):
            for m in data_loader:
                points_sweep = m.read_pcd(pcd_key)
                # if points_sweep is "None":
                if not isinstance(points_sweep,np.ndarray):
                    continue
                else:
                    break
        else:
            points_sweep = data_loader.read_pcd(pcd_key)
        if isinstance(points_sweep,np.ndarray):
            points_sweep = points_sweep[:,:4]
        else:
            raise ValueError


def delete_non_empty_folder(folder_path):
    try:
        shutil.rmtree(folder_path)  # 使用 shutil 模块的 rmtree 函数删除文件夹及其内容
        print(f"已成功删除文件夹: {folder_path}")
    except Exception as e:
        print(f"删除文件夹 {folder_path} 时出现错误: {e}")


def save_txt(out_list,path):
    with open(path,"w") as f:
        for i in out_list:
            f.writelines(i+"\n")
def verfiy_error_clips(data_root,sample_ratio):
    path_list = glob.glob(os.path.join(data_root,"*","clip_info.json"))
    errors_clips = []
    for path in tqdm.tqdm(path_list):
        try:
            read_single_clips(path,sample_ratio)
        except:
            dirname = os.path.dirname(path)
            lmdb_dir = os.path.join(dirname,"lmdb")
            lmdb_dir0 = os.path.join(dirname,"lmdb_0")
            if os.path.exists(lmdb_dir):
                delete_non_empty_folder(lmdb_dir)
            elif os.path.exists(lmdb_dir0):
                num = len(glob.glob(os.path.join(path,"lmdb_*_info.json")))
                for m in range(num):
                    delete_non_empty_folder(os.path.join(dirname,"lmdb_{}".format(m)))
            errors_clips.append(path)
    return errors_clips
if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--data_root', type=str, default=None, help='specify the config for training')
    # args = parser.parse_args()
    # data_root = args.data_root
    # data_root = "/yfw-b3-mix01/cv2/permanent/taoguo/ripples_platform/haixu_4Dhaomibopucai/chengshidaolu/chery_01829/data/20241231_n/"
    data_root = "/yfw-b3-mix01/cv2/permanent/taoguo/ripples_platform/meixing_shanghai_data_11v/chery_24029/data/20241212_d"
    sample_ratio = 1000 #采样倍数隔帧校验，采样倍数越大，校验越快，
    errors_clips = verfiy_error_clips(data_root,sample_ratio)
    save_txt(errors_clips,"/train30/cv2/permanent/liangzhao11/release/error_logs/6768de008af5ae4cacd57fd7_v1.txt")
    print(errors_clips)
    

    # eval $(ssh-agent -s)
    # ssh-add /train30/cv2/permanent/liangzhao11/ssh_key

    