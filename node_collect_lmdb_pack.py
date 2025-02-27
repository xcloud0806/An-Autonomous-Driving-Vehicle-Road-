from multiprocessing import Pool
from tqdm import tqdm
from utils import lmdb_helper, prepare_infos, prepare_coll_seg_infos
from node_lmdb_pack import gen_seg_lmdb
import os, sys
import json
import shutil
import numpy as np
from loguru import logger
from node_lmdb_pack import check_cnt_result, check_with_lmdb_info, multi_process_error_callback, dump_numpy

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

def node_main(run_config):
    if 'multi_seg' not in run_config:
        logger.error(" multi_seg not in config file")
        sys.exit(1)
    if not run_config['multi_seg']['enable']:
        logger.error(" multi_seg not enable")
        sys.exit(1)

    collect_root = run_config['multi_seg']['multi_info_path']
    pre_anno_cfg = run_config['annotation']
    test_road_gnss_file = f"{pre_anno_cfg['test_gnss_info']}"
    deploy_cfg = run_config["deploy"]
    src_deploy_root = deploy_cfg["clip_submit_data"]
    subfix = deploy_cfg['data_subfix']
    anno_path = os.path.join(src_deploy_root, subfix)
    colls = os.listdir(collect_root)
    pool = Pool(processes=8)
    for coll in colls:
        coll_path = os.path.join(collect_root, coll)
        multi_info_json = os.path.join(coll_path, "multi_info.json")
        if not os.path.exists(multi_info_json):
            logger.error(f"{multi_info_json} Not Exists.")
            continue

        multi_info = json.load(open(multi_info_json, "r"))
        multi_info_id = coll
        multi_info_status = True
        if not multi_info[multi_info_id]["multi_odometry_status"] :
            multi_info_status = False
        if not multi_info_status:
            logger.warning(f"{multi_info_id} multi odometry status is False")
            continue
        night_seg_path = multi_info[multi_info_id]["main_clip_path"][0]
        night_seg_id = os.path.basename(night_seg_path)
        night_meta_file = os.path.join(night_seg_path, f"{coll}_multi_meta.json")
        if not os.path.exists(night_meta_file):
            logger.error(f"{night_meta_file} Not Exists.")
            continue
        meta = json.load(open(night_meta_file, "r"))
        segid = meta['seg_uid']

        reconstruct_path = os.path.join(night_seg_path, "multi_reconstruct")
        if not os.path.exists(reconstruct_path):
            reconstruct_path = os.path.join(night_seg_path, "reconstruct")

        rgb_file = []
        if os.path.exists(reconstruct_path):
            files = os.listdir(reconstruct_path)
            for f in files:
                if f.endswith("jpg") or f.endswith("jpeg"):
                    r_file = os.path.join(reconstruct_path, f)
                    rgb_file.append(r_file)
                if f.endswith("npy"):
                    npy_file = os.path.join(reconstruct_path, f)
                    rgb_file.append(npy_file)
        
        seg_frame_path = meta['frames_path']
        first_lidar_pose = np.array(meta['frames'][0]['lidar']['pose']).astype(np.float32)
        dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
        if (first_lidar_pose==dft_pose_matrix).all():
            print(f"{segid} not selected .")
            continue
        
        print("Commit segment {}.".format(segid))
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

        submit_data_path = os.path.join(anno_path, segid)
        def prepare_copy(file_name, rgb=False):
            file_src = os.path.join(night_seg_path, file_name)
            file_dst = os.path.join(submit_data_path, file_name)
            if rgb:
                file_src = file_name
                _f = os.path.basename(file_name)
                file_dst = os.path.join(submit_data_path, _f)

            shutil.copy(file_src, file_dst)

        if not os.path.exists(submit_data_path):
            os.makedirs(submit_data_path, mode=0o775, exist_ok=True)
        clip_info = prepare_infos(meta, enable_cams, enable_bpearls, night_seg_path, test_road_gnss_file)
        with open(os.path.join(submit_data_path, "clip_info.json"), "w") as fp:
            ss = json.dumps(clip_info, ensure_ascii=False, default=dump_numpy)
            fp.write(ss)
        prepare_copy("gnss.json")
        prepare_copy("vehicle.json")
        for r in rgb_file:
            prepare_copy(r, True)
        pool.apply_async(
            gen_seg_lmdb,
            args=(
                meta,
                enable_cams,
                enable_bpearls,
                submit_data_path
            ),
            error_callback=multi_process_error_callback,
        )        
    pool.close()
    pool.join()
    print(f">>>> {collect_root} Prepare LMDB Done.")
    commit_segs = os.listdir(anno_path)
    for seg in commit_segs:
        seg_lmdb_path = os.path.join(anno_path, seg)
        submit_data_path = os.path.join(anno_path, seg)
        ret = check_with_lmdb_info(seg_lmdb_path)
        if not ret:
            print(f"{seg} check with lmdb info failed.")
            continue

if __name__ == '__main__':
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "pack_collect_lmdb.log"))
    node_main(run_config)

    sys.exit(0)  