import lmdb
import cv2
import os
import numpy as np
try:
    from .pcd_io import load_pcd_from_bytes
except:
    from utils.pcd_io import load_pcd_from_bytes
from hashlib import sha256
from loguru import logger
import json
DFT_CHUNK_SIZE=(8*1024*1024)

class LmdbHelper:
    def __init__(self, lmdb_path, tmp_size=(25*1024*1024*1024)):
        self.env = lmdb.open(lmdb_path, map_size=tmp_size)
        self.lmdb_path = lmdb_path

    def __del__(self):
        # print("close lmdb")
        logger.info(f"{self.lmdb_path} close lmdb")
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
        # img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        image = np.frombuffer(content, dtype=np.uint8)
        arr = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return arr

    def read_data(self, key: str):
        txn = self.env.begin(write=False)
        value = txn.get(str(key).encode())
        return value
    
    def get_all_keys(self):
        txn =  self.env.begin(write=False)
        cursor = txn.cursor()
        keys = [key.decode() for key, _ in cursor]
        return keys

    def read_pcd(self, key: str):
        content = self.read_data(key)
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
    
    @staticmethod
    def cacl_hash(lmdb_path):
        """
        计算给定文件的SHA-1哈希值。该函数接受一个文件路径作为参数，读取文件内容并计算其SHA-1哈希值。
        
        Parameters:
        lmdb_path (str): 待计算哈希值的文件路径。
        
        Returns:
        tuple: 返回一个元组，第一个元素是文件大小（单位：字节），第二个元素是文件内容的SHA-1哈希值（十六进制字符串表示）。
        
        注意：
        这个函数只计算文件的头尾和中间8M数据的hash值。如果文件大小小于8M，则计算整个文件的hash值。
        """
        
        def hash_file(filename):
            """This function returns the SHA-1 hash
            of the file passed into it"""

            # make a hash object
            h = sha256()
            file_size = os.path.getsize(filename)

            # open file for reading in binary mode
            with open(filename,'rb') as file:
                # loop till the end of the file
                # 只计算文件的头尾和中间8M数据的hash值
                chunk = file.read(min(DFT_CHUNK_SIZE, file_size))
                h.update(chunk)
                if  (file_size - DFT_CHUNK_SIZE) > DFT_CHUNK_SIZE:
                    file.seek(max(0, file_size-DFT_CHUNK_SIZE), 0)
                    chunk = file.read(min(DFT_CHUNK_SIZE, file_size - DFT_CHUNK_SIZE))
                    h.update(chunk)    

            # return the hex representation of digest
            return h.hexdigest(), file_size
        bin_file = os.path.join(lmdb_path, "data.mdb")
        hash_str, file_size = hash_file(bin_file)
        return file_size, hash_str

class SegLmdbReader:
    def __init__(self, seg_path) -> None:
        self.contain_lmdb = False

def read_json(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data
if  __name__ == "__main__":
    clip_info_path = "/data_autodrive/users/liangzhao11/work_data/6768de008af5ae4cacd57fd7/data/20241212_d/chery_24029_20241212-14-16-33_seg46/clip_info.json"
    lmdb_dir = os.path.join(os.path.dirname(clip_info_path),"lmdb")
    clip_info = read_json(clip_info_path)
    data = LmdbHelper(lmdb_dir)
    tokens = [m["lidar"]for m in clip_info["pair_list"]]
    for ts in tokens:
        pcd_key = "lidar_{}".format(ts)
        pts = data.read_pcd(pcd_key)[:,:4]
    print("done")