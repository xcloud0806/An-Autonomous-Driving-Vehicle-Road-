import os, sys
from utils import prepare_infos, prepare_coll_seg_infos
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

paths = [
    # segment, clip_submit, subfix, deploy
    (
        "/data_cold2/origin_data/sihao_1482/custom_seg/city_scene_test/20240326",
        "/data_autodrive/auto/custom/sihao_1482/city_scene_test",
        "20240326",
        "/train30/cv2/permanent/brli/unpack_lmdb_ftp/demand/city_scene_test/sihao_1482/20240326",
    ),
    (
        "/data_cold2/origin_data/sihao_19cp2/custom_seg/city_scene_test/20240323",
        "/data_autodrive/auto/custom/sihao_19cp2/city_scene_test",
        "20240323",
        "/train30/cv2/permanent/brli/unpack_lmdb_ftp/demand/city_scene_test/sihao_19cp2/20240323"
    ),
    (
        "/data_cold2/origin_data/sihao_27en6/custom_seg/city_scene_test/20240325",
        "/data_autodrive/auto/custom/sihao_27en6/city_scene_test",
        "20240325",
        "/train30/cv2/permanent/brli/unpack_lmdb_ftp/demand/city_scene_test/sihao_27en6/20240325"
    ),
    (
        "/data_cold2/origin_data/sihao_27en6/custom_seg/city_scene_test/20240326",
        "/data_autodrive/auto/custom/sihao_27en6/city_scene_test",
        "20240326",
        "/train30/cv2/permanent/brli/unpack_lmdb_ftp/demand/city_scene_test/sihao_27en6/20240326"
    ),
]

def node_main(seg_root, clip_submt, subfix, tgt_deploy_root):
    seg_mode = "distance"
    tgt_seg_path = seg_root
    clip_lane = os.path.join(clip_submit, "clip_lane", subfix)
    specs = list()
    test_road_gnss_file = "test_roads_gnss_info.json"

    src_deploy_root = os.path.join(clip_submit, "data")
    anno_path = os.path.join(src_deploy_root, subfix)

    seg_root_path = tgt_seg_path
    if not os.path.exists(seg_root_path):
        print(f"{seg_root_path} NOT Exist...")
        sys.exit(1)
    seg_names = os.listdir(seg_root_path)
    seg_names.sort()

    # user = "brli"
    # passwd = "lerinq1w2E#R$"
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
        seg_path = os.path.join(seg_root_path, segid)
        meta_file = os.path.join(seg_root_path, segid, "meta.json")
        for f in os.listdir(seg_path):
            _f = os.path.join(seg_path, f)
            if os.path.isfile(_f) and f.endswith('multi_meta.json'):
                meta_file = _f
                break
        if not os.path.exists(meta_file):
            continue

        reconstruct_path = os.path.join(clip_lane, segid)
        if not os.path.exists(reconstruct_path):
            print(f"Reconstruct path {reconstruct_path} does not exist. Skipping...")
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

        meta_json = open(meta_file, "r")
        meta = json.load(meta_json)
        first_lidar_pose = np.array(meta['frames'][0]['lidar']['pose']).astype(np.float32)
        dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
        if (first_lidar_pose==dft_pose_matrix).all():
            print(f"{segid} not selected .")
            continue
        
        print("Commit segment {}.".format(segid))
        enable_cams = meta["cameras"]
        enable_bpearls = []

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

        # clip_info = prepare_infos(meta, enable_cams, enable_bpearls, seg_path, test_road_gnss_file)
        clip_info = prepare_coll_seg_infos(reconstruct_path, meta, enable_cams, enable_bpearls, seg_path, test_road_gnss_file)
        if seg_mode == 'test' or seg_mode == 'luce' or seg_mode == 'hpp_luce':
            clip_info['datasets'] = 'test'
            
        clip_info_file = os.path.join(submit_data_path, "clip_info.json")
        with open(clip_info_file, "w") as fp:
            ss = json.dumps(clip_info, ensure_ascii=False, default=dump_numpy)
            fp.write(ss)

        print("Upload segment {}.".format(segid))
        items = os.listdir(submit_data_path)
        dst_path = os.path.join(tgt_deploy_root, subfix, segid)
        for item in  items:        
            filepath = os.path.join(submit_data_path, item)
            if os.path.isfile(filepath):
                func_upload_file(filepath, item, dst_path)          
    ftp.quit()

if  __name__ == "__main__":
    for p in paths:
        print(f"handle {p[0]}")
        segment, clip_submit, subfix, deploy = p 
        node_main(segment, clip_submit, subfix, deploy)

    sys.exit(0)  
