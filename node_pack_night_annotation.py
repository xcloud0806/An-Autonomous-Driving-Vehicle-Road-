import os
import numpy as np
import sys
import json
import shutil
from multiprocessing import Pool
from copy import deepcopy
from utils import gen_label_obstacle, gen_label_obstacle_hpp, gen_label_obstacle_static, prepare_coll_seg_infos, prepare_infos

ANNO_INFO_JSON = "annotation.json"
ANNO_INFO_CALIB_KEY = "calib"
ANNO_INFO_INFO_KEY = "clip_info"
ANNO_INFO_LANE_KEY = "lane"
ANNO_INFO_OBSTACLE_KEY = "obstacle"
ANNO_INFO_OBSTACLE_STATIC_KEY = "obstacle_static"
ANNO_INFO_PAIR_KEY = "pair_list"
ANNO_INFO_RAW_PAIR_KEY = "raw_pair_list"
ANNO_INFO_POSE_KEY = "pose_list"
ANNO_INFO_OBSTACLE_HPP_KEY = "obstacle_hpp"
TEST_ROADS_GNSS = "test_roads_gnss_info.json"


def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def get_coll_annotation(
    dst_coll_path,
    coll_id,
    coll_date,
    seg_meta,
    seg_root_path,
    lane_res_path,
    lane_data_path,
    obs_res_path,
    obs_data_path,
    test_gnss_json,
    enable_cams,
    enable_bpearls
):
    annotation = {}

    calibs = seg_meta["calibration"]
    segid = seg_meta['seg_uid']
    seg_subfix = seg_meta['date']
    annotation[ANNO_INFO_CALIB_KEY] = calibs
    testing_sets = False
    pack_lane = False
    pack_obstacle = False

    annotation[ANNO_INFO_LANE_KEY] = {}
    if not os.path.exists(lane_res_path) or not os.path.exists(lane_data_path):
        print("skip {} lane anno data submit in {}".format(seg_meta['seg_uid'], coll_id))
    else:
        pack_lane = True
        lane_anno_res = {}
        lane_anno_res_json = open(os.path.join(lane_res_path, os.listdir(lane_res_path)[0]))
        lane_anno_res = json.load(lane_anno_res_json)
        annotation[ANNO_INFO_LANE_KEY] = lane_anno_res
        clip_info = prepare_coll_seg_infos(
            lane_data_path,
            seg_meta,
            enable_cams,
            enable_bpearls,
            seg_root_path,
            test_gnss_json,
        )
        # annotation[ANNO_INFO_INFO_KEY] = clip_info
        annotation[ANNO_INFO_PAIR_KEY] = clip_info[ANNO_INFO_PAIR_KEY]
        annotation[ANNO_INFO_RAW_PAIR_KEY] = clip_info[ANNO_INFO_RAW_PAIR_KEY]
        annotation[ANNO_INFO_POSE_KEY] = clip_info[ANNO_INFO_POSE_KEY]
        anno_clip_info = {}

        anno_clip_info["seg_uid"] = clip_info["seg_uid"]
        anno_clip_info["segment_path"] = clip_info["segment_path"]
        anno_clip_info["datasets"] = clip_info["datasets"]
        if obs_data_path is not None and 'clip_obstacle_test' in obs_data_path:
            anno_clip_info['datasets'] = 'test'
        anno_clip_info["road_name"] = clip_info["road_name"]
        anno_clip_info["road_list"] = clip_info["road_list"]
        anno_clip_info["world_to_img"] = clip_info["world_to_img"]
        anno_clip_info["region"] = clip_info["region"]
        anno_clip_info["lane_version"] = clip_info["lane_version"]
        annotation[ANNO_INFO_INFO_KEY] = anno_clip_info
        if anno_clip_info['datasets'] == 'test':
            testing_sets = True

    annotation[ANNO_INFO_OBSTACLE_KEY] = {}
    annotation[ANNO_INFO_OBSTACLE_STATIC_KEY] = {}
    annotation[ANNO_INFO_OBSTACLE_HPP_KEY] = {}

    if not pack_obstacle and not pack_lane:
        print(f"No lane res or obs res in [{segid}], clip info use default info.")
        clip_info = prepare_infos(seg_meta, enable_cams, [], seg_root_path, test_gnss_json)
        annotation[ANNO_INFO_PAIR_KEY] = clip_info[ANNO_INFO_PAIR_KEY]
        annotation[ANNO_INFO_RAW_PAIR_KEY] = clip_info[ANNO_INFO_RAW_PAIR_KEY]
        annotation[ANNO_INFO_POSE_KEY] = clip_info[ANNO_INFO_POSE_KEY]
        anno_clip_info = {}

        anno_clip_info["seg_uid"] = clip_info["seg_uid"]
        anno_clip_info["segment_path"] = clip_info["segment_path"]
        anno_clip_info["datasets"] = clip_info["datasets"]
        anno_clip_info["road_name"] = clip_info["road_name"]
        anno_clip_info["road_list"] = clip_info["road_list"]
        annotation[ANNO_INFO_INFO_KEY] = anno_clip_info
        if anno_clip_info['datasets'] == 'test':
            testing_sets = True

    seg_submit_path = os.path.join(dst_coll_path, "annotation_train", coll_date, segid)
    if testing_sets:
        seg_submit_path = os.path.join(dst_coll_path, "annotation_test", coll_date, segid)
    if not os.path.exists(seg_submit_path):
        os.makedirs(seg_submit_path, mode=0o777, exist_ok=True)
    print(f"...Submit {seg_submit_path}")
    submit_anno_json = os.path.join(seg_submit_path, ANNO_INFO_JSON)
    with open(submit_anno_json, "w") as wfp:
        anno_json_str = json.dumps(annotation, ensure_ascii=False, default=dump_numpy)
        wfp.write(anno_json_str)
    return 0, testing_sets


def node_main(run_config: dict):
    pre_anno_cfg = run_config["annotation"]
    clip_lane = pre_anno_cfg["clip_lane"]
    clip_obstacle = None
    clip_obstacle_test = None
    test_road_gnss_file = pre_anno_cfg["test_gnss_info"]

    anno_res_cfg = run_config["ripples_platform"]["abk_mark_result"]
    clip_lane_anno_path = (
        anno_res_cfg["clip_lane_annotation"]
        if "clip_lane_annotation" in anno_res_cfg
        else None
    )
    clip_obs_anno_path = None

    if clip_lane_anno_path is None:
        print("!!! Lane Anno is NECESSARY in collect.")
        sys.exit(0)

    deploy_cfg = run_config["deploy"]
    anno_root = deploy_cfg["clip_submit"]
    subfix = deploy_cfg["data_subfix"]

    if 'multi_seg' in run_config:
        multi_cfg = run_config['multi_seg']
        enable_multi_seg = (multi_cfg['enable'] == "True")

    if not enable_multi_seg:
        print("Wrong node selected.")
        sys.exit(0)

    coll_root = multi_cfg["multi_info_path"]
    collects = os.listdir(coll_root)
    for coll in collects:
        coll_info_json = os.path.join(coll_root, coll, "multi_info.json")
        if not os.path.exists(coll_info_json):
            continue
        coll_info = json.load(open(coll_info_json, "r"))
        coll_id = coll
        segs = coll_info[coll]["main_clip_path"]
        day_seg_path = coll_info[coll]["clips_path"][0]
        day_seg_id = os.path.basename(day_seg_path)
        reconstruct_path = coll_info[coll]['reconstruct_path']
        for seg_path in segs:            
            meta_json = os.path.join(seg_path, f"{coll_id}_multi_meta.json")
            if not os.path.exists(meta_json):
                print(f"{seg_path} CANNOT READ IN THIS COLLECT {coll_id}")
                continue
            meta = json.load(open(meta_json, "r"))
            segid = meta["seg_uid"]
            seg_subfix = meta['date']
            enable_cams = meta["cameras"]
            print(f"Commit Seg[{segid}] in Collect[{coll_id}]")
            lane_anno_path = None
            lane_path = os.path.join(clip_lane, segid)
            if clip_lane_anno_path is not None:
                lane_anno_path = os.path.join(clip_lane_anno_path, day_seg_id)            
            obstacle_path = ""
            obstacle_anno_path = ""

            status, is_test = get_coll_annotation(
                anno_root,
                coll_id,
                subfix,
                meta,
                seg_path,
                lane_anno_path,
                lane_path,                
                obstacle_anno_path,
                obstacle_path,                
                test_road_gnss_file,
                enable_cams,
                []
            )

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    lane_anno_path = None
    if len(sys.argv) == 3:
        lane_anno_path = sys.argv[2]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    
    if lane_anno_path is not None and lane_anno_path != "":
        run_config["ripples_platform"]["abk_mark_result"]["clip_lane_annotation"] = lane_anno_path

    node_main(run_config)
