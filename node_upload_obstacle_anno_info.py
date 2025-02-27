import os, sys
from utils import prepare_infos
import json
import numpy as np
import shutil
from node_ftp_upload import FTP1, FtpUploadTracker, error_perm
import time

ANNO_INFO_JSON = "annotation.json"
ANNO_INFO_CALIB_KEY = "calib"
ANNO_INFO_INFO_KEY = "clip_info"
ANNO_INFO_LANE_KEY = "lane"
ANNO_INFO_OBSTACLE_KEY = "obstacle"
ANNO_INFO_OBSTACLE_STATIC_KEY = "obstacle_static"
ANNO_INFO_PAIR_KEY = "pair_list"
ANNO_INFO_RAW_PAIR_KEY = "raw_pair_list"
ANNO_INFO_POSE_KEY = "pose_list"
TEST_ROADS_GNSS = "test_roads_gnss_info.json"
DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
def node_main(run_config):
    seg_config = run_config["preprocess"]
    seg_mode =  seg_config["seg_mode"]
    tgt_seg_path = seg_config["segment_path"]
    car_name = seg_config['car']
    pre_anno_cfg = run_config['annotation']
    clip_obstacle =  pre_anno_cfg['clip_obstacle']
    clip_obstacle_test =  pre_anno_cfg['clip_obstacle_test']
    test_road_gnss_file = f"{pre_anno_cfg['test_gnss_info']}"

    clip_lane = pre_anno_cfg['clip_lane']
    clip_lane_check = clip_lane.replace("clip_lane", "clip_lane_check")
    specs = list()
    if seg_mode == "hpp" and os.path.exists(clip_lane_check):
        for rgbs in os.listdir(clip_lane_check):
            seg_id, _ = os.path.splitext(rgbs)
            specs.append(seg_id)

    deploy_cfg = run_config["deploy"]
    src_deploy_root = deploy_cfg["clip_submit_data"]
    subfix = deploy_cfg['data_subfix']
    tgt_deploy_root = deploy_cfg["tgt_rdg_path"]
    anno_path = os.path.join(src_deploy_root, subfix)
    spec_clips = seg_config.get("spec_clips", None)
    seg_root_path = tgt_seg_path
    if not os.path.exists(seg_root_path):
        print(f"{seg_root_path} NOT Exist...")
        sys.exit(1)
    seg_names = os.listdir(seg_root_path)
    seg_names.sort()

    user = "taoguo"
    passwd = 'Dltt1991191527///'
    ftp = FTP1()
    ftp.set_debuglevel(0)
    ftp.connect('10.1.165.27', 21)
    ftp.login(user, passwd)

    def ftp_mkd_cwd(path, first_call=True):
        try:
            ftp.cwd(path)
        except error_perm:
            ftp_mkd_cwd(os.path.dirname(path), False)
            ftp.mkd(path)
            if first_call:
                ftp.cwd(path)       
    def func_upload_file(filepath, filename, dst_path):
        ftp_mkd_cwd(dst_path)
        file_size = os.path.getsize(filepath)
        _tracker = FtpUploadTracker(file_size)
        bufsize = 8192
        fp = open(filepath, "rb")    
        
        tic = time.time()
        ftp.storbinary('STOR %s' % filename, fp, bufsize, callback=_tracker.handle)
        toc = time.time()
        print("upload %s, size:%dMB, cost:%.2f" % (filename, round(file_size / 1024 / 1024 ,1), toc - tic))
        fp.close()         

    for segid in seg_names:
        if len(specs) > 0 and segid not in specs:
            continue
        if spec_clips is not None:
            go_on = False
            for clip in spec_clips:
                if clip in segid:
                    go_on = True
                    break
            if not go_on:
                continue        
        seg_path = os.path.join(seg_root_path, segid)
        meta_file = os.path.join(seg_root_path, segid, "meta.json")
        for f in os.listdir(seg_path):
            _f = os.path.join(seg_path, f)
            if os.path.isfile(_f) and f.endswith('multi_meta.json'):
                meta_file = _f
                break
        if not os.path.exists(meta_file):
            continue

        meta_json = open(meta_file, "r")
        meta = json.load(meta_json)
        first_lidar_pose = np.array(meta['frames'][0]['lidar']['pose']).astype(np.float32)
        dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
        if (first_lidar_pose==dft_pose_matrix).all():
            print(f"{segid} not selected .")
            continue
        
        print("Commit segment {}.".format(segid))
        meta_json.close()
        obs_dst_path = os.path.join(clip_obstacle, segid)
        if not os.path.exists(obs_dst_path):
            obs_dst_path = os.path.join(clip_obstacle_test, segid)
        if not os.path.exists(obs_dst_path):
            continue
        
        filepath = os.path.join(obs_dst_path, "{}_infos.json".format(segid))
        dst_path = os.path.join(tgt_deploy_root, subfix, segid)
        item = "{}_infos.json".format(segid)
        if os.path.isfile(filepath):
            func_upload_file(filepath, item, dst_path)            
    ftp.quit()

if  __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    node_main(run_config)

    sys.exit(0)  