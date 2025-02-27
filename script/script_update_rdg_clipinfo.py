# coding:utf-8
import json
from math import floor
import os, sys
import os.path
from ftplib import FTP, error_perm
import time
import sys
import traceback as tb
from loguru import logger
import numpy as np

sys.path.append("../utils")
from prepare_clip_infos import prepare_infos

DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# 如需要支持中文名文件传输，需要将ftplib.py文件中的  encoding = "latin-1"  改为   encoding = "utf-8"
class FTP1(FTP):     # 继承FTP类，修改解码参数，解决不识别文件名中文的问题
    encoding = "utf-8"

class FtpUploadTracker:
    sizeWritten = 0
    totalSize = 0
    lastShownPercent = 0

    def __init__(self, totalSize):
        self.totalSize = totalSize

    def handle(self, block):
        self.sizeWritten += 8192
        percentComplete = floor((self.sizeWritten / self.totalSize) * 100)

        if (self.lastShownPercent != percentComplete):
            self.lastShownPercent = percentComplete
            # print(str(percentComplete) +"% percent complete")
            print("{:0>2d}%... ".format(percentComplete), end="\r")
        
        if percentComplete == 100:
            print("Upload Complete.")

class FtpUploadHelper:
    def __init__(self) -> None:
        self.flag = False
        self.handle = FTP1()
        self.handle.set_debuglevel(0)
        try:
            self.handle.connect('10.1.165.27', 21)
            # self.handle.login('brli', 'lerinq1w2E#R$')
            self.handle.login('taoguo', 'Dltt1991191527///')
            self.flag = True
        except error_perm:
            logger.error("FTP Login Failed.")
            sys.exit(1)
        except Exception as e:
            tb.print_exc()
    
    def ftp_mkd_cwd(self, path, first_call=True):
        if not self.flag:
            return
        try:
            self.handle.cwd(path)
        except error_perm:
            self.ftp_mkd_cwd(os.path.dirname(path), False)
            self.handle.mkd(path)
            if first_call:
                self.handle.cwd(path)
    
    def upload_file(self, filepath, filename, dst_path):
        if not self.flag:
            return
        self.ftp_mkd_cwd(dst_path)
        file_size = os.path.getsize(filepath)
        _tracker = FtpUploadTracker(file_size)
        bufsize = 8192
        fp = open(filepath, "rb")    
        
        tic = time.time()
        self.handle.storbinary('STOR %s' % filename, fp, bufsize, callback=_tracker.handle)
        toc = time.time()
        print("upload %s, size:%dMB, cost:%.2f" % (filename, round(file_size / 1024 / 1024 ,1), toc - tic))
        fp.close()

    def upload_segment_path(self, src_dir, config_file, dst_path):
        if not self.flag:
            return

        items = os.listdir(src_dir)
        for item in  items:        
            filepath = os.path.join(src_dir, item)
            if os.path.isdir(filepath):
                if item.startswith("lmdb"):
                    dbs = os.listdir(filepath)
                    for db in dbs:
                        _filepath =  os.path.join(filepath, db)
                        if os.path.isdir(_filepath):
                            continue
                        self.upload_file(_filepath, db, os.path.join(dst_path, item))
            else:
                self.upload_file(filepath, item, dst_path)
        _config = os.path.basename(config_file)
        self.upload_file(config_file, _config, dst_path)
        # ftp.close()
    
    def __del__(self):
        if self.flag:
            self.handle.close()

def node_main(run_config):
    seg_config = run_config["preprocess"]
    deploy_cfg = run_config["deploy"]
    src_deploy_root = deploy_cfg["clip_submit_data"]
    seg_mode = seg_config["seg_mode"]
    tgt_seg_path = seg_config["segment_path"]
    tgt_deploy_root = deploy_cfg["tgt_rdg_path"]
    tgt_frame_root = deploy_cfg['tgt_rdg_deploy_path']
    subfix = deploy_cfg['data_subfix']
    src_segs = os.path.join(src_deploy_root, subfix)
    
    ftp_handle = FtpUploadHelper()
    seg_root_path = tgt_seg_path
    if not os.path.exists(seg_root_path):
        logger.error(f"{seg_root_path} NOT Exist...")
        sys.exit(1)
    seg_names = os.listdir(seg_root_path)
    seg_names.sort()
    for segid in seg_names:
        submit_data_path = os.path.join(src_segs, segid)
        if not os.path.exists(submit_data_path):
            continue
        seg_path = os.path.join(seg_root_path, segid)
        meta_file = os.path.join(seg_root_path, segid, "updated_meta.json")
        if not os.path.exists(meta_file):
            meta_file = os.path.join(seg_root_path, segid, "meta.json")
        if not os.path.exists(meta_file):
            continue

        meta_json = open(meta_file, "r")
        meta = json.load(meta_json)
        seg_frame_path = meta['frames_path']
        first_lidar_pose = np.array(meta['frames'][0]['lidar']['pose']).astype(np.float32)
        dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
        if (first_lidar_pose==dft_pose_matrix).all():
            logger.warning(f"{segid} not selected .")
            continue

        enable_cams = meta["cameras"]
        enable_bpearls = []
        if "other_sensors_info" in meta:
            _info = meta["other_sensors_info"]
            if "bpearl_lidar_info" in _info:
                if _info["bpearl_lidar_info"]["enable"] == "true":
                    enable_bpearls = _info["bpearl_lidar_info"]["positions"]
            if "inno_lidar_info" in _info:
                if _info["inno_lidar_info"]["enable"] == "true":
                    enable_bpearls.extend(_info["inno_lidar_info"]["positions"])

        logger.info("Commit segment {}.".format(segid))
        clip_info = prepare_infos(meta, enable_cams, enable_bpearls, seg_path)
        if seg_mode == 'test' or seg_mode == 'luce':
            clip_info['datasets'] = 'test'

        clip_submit_info_file = os.path.join(submit_data_path, "clip_info.json")
        with open(clip_submit_info_file, "w") as fp:
            ss = json.dumps(clip_info, ensure_ascii=False, default=dump_numpy)
            fp.write(ss)
        dst_path = os.path.join(tgt_deploy_root, subfix, segid)
        dst_deploy_path = os.path.join(tgt_frame_root, subfix, segid)
        ftp_handle.upload_file(clip_submit_info_file, "clip_info.json", dst_path)
        ftp_handle.upload_file(clip_submit_info_file, "clip_info.json", dst_deploy_path)
    logger.info(f"......\t Upload {subfix} to FTP Done...")

if __name__ == '__main__':
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    node_main(run_config)