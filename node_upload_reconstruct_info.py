import os, sys
from utils import prepare_infos
import json
import numpy as np
import shutil
from node_ftp_upload import FTP1, FtpUploadTracker, error_perm
import time
from loguru import logger

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
    rec_cfg = run_config["reconstruction"]
    skip_reconstruct = False
    if rec_cfg['enable'] != "True":
        skip_reconstruct = True
    pre_anno_cfg = run_config['annotation']
    clip_lane = pre_anno_cfg['clip_lane']
    clip_lane_check = clip_lane.replace("clip_lane", "clip_lane_check")
    specs = list()
    if seg_mode == "hpp" and os.path.exists(clip_lane_check):
        for rgbs in os.listdir(clip_lane_check):
            seg_id, _ = os.path.splitext(rgbs)
            specs.append(seg_id)
    test_road_gnss_file = f"{pre_anno_cfg['test_gnss_info']}"
    spec_clips = seg_config.get("spec_clips", None)
    deploy_cfg = run_config["deploy"]
    src_deploy_root = deploy_cfg["clip_submit_data"]
    subfix = deploy_cfg['data_subfix']
    tgt_data_root = deploy_cfg["tgt_rdg_path"]
    tgt_deploy_root = deploy_cfg["tgt_rdg_deploy_path"]
    anno_path = os.path.join(src_deploy_root, subfix)

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
        try:
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
        except Exception as e:
            logger.error(f"func_upload_file({filepath} {filename} {dst_path}) Caught an exception {e}")
            raise ValueError(f"func_upload_file({filepath} {filename} {dst_path}) Caught an exception {e}")
    
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

        meta_fp = open(meta_file, "r")
        meta = json.load(meta_fp)
        if seg_mode == "hpp":
            if 'key_frames' not in meta:
                logger.warning(f"{segid} skip. Because no [key_frame] field.")
                continue
            sig_frames = meta['key_frames']
            if len(sig_frames) <= 10:
                logger.warning(f"{segid} skip. Because too few key frame.")
                continue
            sig_frames_lost = meta.get('key_frames_lost', 0)
            if sig_frames_lost > 2:
                logger.warning(f"{segid} skip. Because too many key frame lost. [{sig_frames_lost}]")
                continue

        reconstruct_path = os.path.join(seg_root_path, segid, "multi_reconstruct")
        if not os.path.exists(reconstruct_path):
            reconstruct_path = os.path.join(seg_root_path, segid, "reconstruct")
        if not skip_reconstruct and not os.path.exists(reconstruct_path):
            logger.warning(f"Reconstruct path {reconstruct_path} does not exist. Skipping...")
            continue
            
        rgb_file = []
        if os.path.exists(reconstruct_path):
            files = os.listdir(reconstruct_path)
            for f in files:
                if f.endswith("jpg") or f.endswith("jpeg") or f.endswith("png"):
                    r_file = os.path.join(reconstruct_path, f)
                    rgb_file.append(r_file)
                if f.endswith("npy") or f.endswith("npz"):
                    npy_file = os.path.join(reconstruct_path, f)
                    rgb_file.append(npy_file)
        else:
            if not skip_reconstruct:
                continue

        meta_json = open(meta_file, "r")
        meta = json.load(meta_json)
        first_lidar_pose = np.array(meta['frames'][0]['lidar']['pose']).astype(np.float32)
        dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
        if (first_lidar_pose==dft_pose_matrix).all():
            logger.warning(f"{segid} not selected .")
            continue
        
        logger.info("Commit segment {}.".format(segid))
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

        meta_json.close()
        submit_data_path = os.path.join(anno_path, segid)
        if not os.path.exists(submit_data_path):
            os.makedirs(submit_data_path, mode=0o775, exist_ok=True)
        def prepare_copy(file_name, rgb=False):
            file_src = os.path.join(seg_path, file_name)
            file_dst = os.path.join(submit_data_path, file_name)
            if rgb:
                file_src = file_name
                _f = os.path.basename(file_name)
                file_dst = os.path.join(submit_data_path, _f)

            shutil.copy(file_src, file_dst)

        for r in rgb_file:
            prepare_copy(r, True)

        clip_info = prepare_infos(meta, enable_cams, enable_bpearls, seg_path, test_road_gnss_file)
        if seg_mode == 'test' or seg_mode == 'luce' or seg_mode == 'hpp_luce' or seg_mode == 'aeb':
            clip_info['datasets'] = 'test'
        
        if 'hpp' in seg_mode and 'key_frames' in meta:
            clip_info['key_frames'] = meta['key_frames']
            
        clip_info_file = os.path.join(submit_data_path, "clip_info.json")
        with open(clip_info_file, "w") as fp:
            ss = json.dumps(clip_info, ensure_ascii=False, default=dump_numpy)
            fp.write(ss)

        logger.info("Upload segment {}.".format(segid))
        items = os.listdir(submit_data_path)
        dst_data_path = os.path.join(tgt_data_root, subfix, segid)
        dst_path = os.path.join(tgt_deploy_root, subfix, segid)
        for item in  items:        
            filepath = os.path.join(submit_data_path, item)
            if os.path.isfile(filepath):
                func_upload_file(filepath, item, dst_path)    
                func_upload_file(filepath, item, dst_data_path)        
    ftp.quit()

if  __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        logger.error(f"{config_file} Not Exists.")
        sys.exit(1)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "upload_reconstruct_info.log"))
    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    try:
        node_main(run_config)
    except Exception as e:
        logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
        sys.exit(1)
    sys.exit(0)  