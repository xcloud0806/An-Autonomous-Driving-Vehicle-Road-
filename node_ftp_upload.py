# coding:utf-8
from genericpath import isdir
import json
from math import floor
import os
import os.path
from ftplib import FTP, error_perm
import time
import sys
import traceback as tb
from loguru import logger

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
            # print("{:0>2d}%... ".format(percentComplete), end="\r")
        
        # if percentComplete == 100:
        #     print("Upload Complete.")

def ftp_upload(filename, src_path, dst_path):
    user = "xbchang"
    passwd = "iflytek@2022cxb"

    ftp = FTP1()
    ftp.set_debuglevel(0)
    ftp.connect('10.1.184.211', 21)
    ftp.login(user, passwd)

    def ftp_mkd_cwd(path, first_call=True):
        try:
            ftp.cwd(path)
        except error_perm:
            ftp_mkd_cwd(os.path.dirname(path), False)
            ftp.mkd(path)
            if first_call:
                ftp.cwd(path)                 

    ftp_mkd_cwd(dst_path)

    file_size = os.path.getsize(os.path.join(src_path, filename))
    _tracker = FtpUploadTracker(file_size)

    bufsize = 8192
    fp = open(os.path.join(src_path, filename), "rb")    
    logger.info(f"Upload {filename} with FTP, Total file size : {str(round(file_size / 1024 / 1024 ,1))} MB")
    tic = time.time()
    ftp.storbinary('STOR %s' % filename, fp, bufsize, callback=_tracker.handle)
    toc = time.time()
    # file_size = fp.tell()
    logger.info("upload %s, size:%d, cost:%.2f" % (filename, file_size, toc - tic))
    fp.close()
    ftp.quit()


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
        logger.info("upload %s, size:%dMB, cost:%.2f" % (filename, round(file_size / 1024 / 1024 ,1), toc - tic))
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

def node_main(run_config, config_file, specs:list):
    pre_anno_cfg = run_config['annotation']
    clip_lane = pre_anno_cfg['clip_lane']
    clip_lane_check = clip_lane.replace("clip_lane", "clip_lane_check")
    seg_config = run_config["preprocess"]
    seg_mode = seg_config['seg_mode']
    if "hpp" in seg_mode and os.path.exists(clip_lane_check):
        specs = list()
        for rgbs in os.listdir(clip_lane_check):
            seg_id, _ = os.path.splitext(rgbs)
            specs.append(seg_id)
    
    deploy_cfg = run_config["deploy"]
    src_deploy_root = deploy_cfg["clip_submit_data"]
    tgt_deploy_root = deploy_cfg["tgt_rdg_path"]
    subfix = deploy_cfg['data_subfix']
    spec_clips = seg_config.get("spec_clips", None)
    src_segs = os.path.join(src_deploy_root, subfix)
    if not os.path.exists(src_segs):
        logger.error(f"{src_segs} not exists!")
        sys.exit(1)
    ftp_handle = FtpUploadHelper()
    for i, seg in enumerate(os.listdir(src_segs)):
        if spec_clips is not None:
            go_on = False
            for clip in spec_clips:
                if clip in seg:
                    go_on = True
                    break
            if not go_on:
                continue
        logger.info(f"[{i+1}/{len(os.listdir(src_segs))}] [{seg}] Start Uploading ......")
        if len(specs) > 0 and seg not in specs:
            continue
        seg_path = os.path.join(src_segs, seg)
        dst_path = os.path.join(tgt_deploy_root, subfix, seg)
        try:
            ftp_handle.upload_segment_path(seg_path, config_file, dst_path)
        except Exception as e:
            logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
            sys.exit(1)
        # ftp_upload_segment(seg_path, config_file, dst_path)
    logger.info(f"......\t Upload {subfix} to FTP Done...")

if __name__ == '__main__':
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    spec_segs = []
    if len(sys.argv) > 2:
        spec = sys.argv[2]

        if os.path.isfile(spec) and os.path.exists(spec):
            _spec_segs = json.load(open(spec, "r"))
            if type(_spec_segs) is list:
                spec_segs = _spec_segs
        else:
            if type(spec) is str:
                spec_segs = spec.split(",")

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    node_main(run_config, config_file, spec_segs)