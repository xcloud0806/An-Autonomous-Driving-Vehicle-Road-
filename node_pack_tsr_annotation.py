import os
import numpy as np
import sys
import json
import shutil
from multiprocessing import Pool
from copy import deepcopy
from scipy.interpolate import interp1d
from script.tool_obtain_tag import obtain_seg_tag
from utils import (
    gen_label_obstacle,
    gen_label_obstacle_static,
    gen_label_obstacle_hpp,
    prepare_infos,
)

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
ANNO_INFO_TSR_KEY = "tsr"
TEST_ROADS_GNSS = "test_roads_gnss_info.json"


def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def gen_annotation(
    seg_anno_path,
    seg_subfix,
    lane_anno_res_path,
    lane_anno_data_path,
    seg_root_path,
    obstacle_res_path,
    obstacle_data_path,
    tsr_anno,
    enable_cams,
    test_gnss_json,
):
    annotation = {}

    prepare_lane = False
    prepare_obstacle = False
    prepare_hpp = False

    seg_meta_json = os.path.join(seg_root_path, "multi_meta.json")
    if not os.path.exists(seg_meta_json):
        seg_meta_json = os.path.join(seg_root_path, "meta.json")

    seg_meta_fp = open(seg_meta_json, "r")
    meta = json.load(seg_meta_fp)
    calibs = meta["calibration"]
    segid = meta["seg_uid"]
    if 'record' in meta and meta['record'] is not None and 'luce_info' in meta['record']:
        annotation['luce_info'] = meta['record']['luce_info']

    annotation[ANNO_INFO_CALIB_KEY] = calibs
    testing_sets = False
    seg_meta_fp.close()

    clip_info = prepare_infos(meta, enable_cams, [], seg_root_path, test_gnss_json)
    annotation[ANNO_INFO_PAIR_KEY] = clip_info[ANNO_INFO_PAIR_KEY]
    annotation[ANNO_INFO_RAW_PAIR_KEY] = clip_info[ANNO_INFO_RAW_PAIR_KEY]
    annotation[ANNO_INFO_POSE_KEY] = clip_info[ANNO_INFO_POSE_KEY]
    anno_clip_info = {}

    anno_clip_info["seg_uid"] = clip_info["seg_uid"]
    anno_clip_info["segment_path"] = clip_info["segment_path"]
    anno_clip_info["datasets"] = clip_info["datasets"]
    if obstacle_data_path is not None and 'clip_obstacle_test' in obstacle_data_path:
        anno_clip_info['datasets'] = 'test'
    anno_clip_info["road_name"] = clip_info["road_name"]
    anno_clip_info["road_list"] = clip_info["road_list"]
    annotation[ANNO_INFO_INFO_KEY] = anno_clip_info
    if anno_clip_info["datasets"] == "test":
        testing_sets = True

    annotation[ANNO_INFO_LANE_KEY] = {}
    if (
        lane_anno_data_path is None
        or lane_anno_res_path is None
        or not os.path.exists(lane_anno_res_path)
        or not os.path.exists(lane_anno_data_path)
    ):
        print("skip {} lane anno data submit".format(meta["seg_uid"]))
    else:
        lane_anno_res = {}
        lane_anno_res_json = open(
            os.path.join(lane_anno_res_path, os.listdir(lane_anno_res_path)[0])
        )
        lane_anno_res = json.load(lane_anno_res_json)
        annotation[ANNO_INFO_LANE_KEY] = lane_anno_res
        annotation[ANNO_INFO_LANE_KEY]["anno_source"] = "aibiaoke"
        anno_clip_info["world_to_img"] = clip_info["world_to_img"]
        anno_clip_info["region"] = clip_info["region"]
        anno_clip_info["lane_version"] = clip_info["lane_version"]
        prepare_lane = True

    annotation[ANNO_INFO_INFO_KEY] = anno_clip_info
    annotation[ANNO_INFO_OBSTACLE_KEY] = {}
    annotation[ANNO_INFO_OBSTACLE_STATIC_KEY] = {}
    annotation[ANNO_INFO_OBSTACLE_HPP_KEY] = {}

    if (
        obstacle_res_path is None
        or obstacle_data_path is None
        or not os.path.exists(obstacle_res_path)
        or not os.path.exists(obstacle_data_path)
    ):
        print("skip {} obstacle anno data submit".format(meta["seg_uid"]))
    else:
        annotation[ANNO_INFO_OBSTACLE_KEY] = gen_label_obstacle(
            segid, obstacle_res_path, obstacle_data_path, meta
        )
        annotation[ANNO_INFO_OBSTACLE_KEY]["anno_source"] = "aibiaoke"
        annotation[ANNO_INFO_OBSTACLE_STATIC_KEY] = gen_label_obstacle_static(
            segid, obstacle_res_path, obstacle_data_path, meta
        )
        annotation[ANNO_INFO_OBSTACLE_HPP_KEY] = gen_label_obstacle_hpp(
            segid, obstacle_res_path, obstacle_data_path, meta
        )
        annotation[ANNO_INFO_OBSTACLE_STATIC_KEY]["anno_source"] = "aibiaoke"
        annotation[ANNO_INFO_OBSTACLE_HPP_KEY]["anno_source"] = "aibiaoke"
        prepare_obstacle = True

    if tsr_anno is not None:
        tsr_annos = {}
        for ts, _tsr_anno in tsr_anno.items():
            tsr_anno_file = _tsr_anno['anno_path']
            with open(tsr_anno_file, "r") as fp:
                tsr_anno_data = json.load(fp)
                tsr_annos[ts] = tsr_anno_data
        annotation[ANNO_INFO_TSR_KEY] = tsr_annos
        prepare_hpp = True

    if prepare_obstacle or prepare_lane or prepare_hpp:
        seg_submit_path = os.path.join(
            seg_anno_path, "annotation_tsr_train", seg_subfix, segid
        )
        if testing_sets:
            seg_submit_path = os.path.join(
                seg_anno_path, "annotation_tsr_test", seg_subfix, segid
            )
        if not os.path.exists(seg_submit_path):
            os.makedirs(seg_submit_path, mode=0o777, exist_ok=True)
        print(f"...Submit {seg_submit_path}")
        submit_anno_json = os.path.join(seg_submit_path, ANNO_INFO_JSON)
        with open(submit_anno_json, "w") as wfp:
            anno_json_str = json.dumps(
                annotation, ensure_ascii=False, default=dump_numpy, indent=4
            )
            wfp.write(anno_json_str)
        return 0, testing_sets
    else:
        print("skip {} commit.".format(meta["seg_uid"]))
        return 1, testing_sets

def parse_tsr_anno(tsr_anno_root):
    ret = {}
    if not os.path.exists(tsr_anno_root):
        return ret

    if not os.path.isdir(tsr_anno_root):
        return ret

    anno_files = os.listdir(tsr_anno_root)
    anno_files.sort()

    for anno_file in anno_files:
        anno_name = os.path.splitext(anno_file)[0]
        segid, ts, index = anno_name.split("+")
        if segid not in ret:
            ret[segid] = {}
        ret[segid][ts] = {}
        ret[segid][ts]["index"] = index
        ret[segid][ts]["anno_path"] = os.path.join(tsr_anno_root, anno_file)
    return ret

def parse_tsr_anno_by_directory(tsr_anno_root):
    ret = {}
    if not os.path.exists(tsr_anno_root):
        return ret

    if not os.path.isdir(tsr_anno_root):
        return ret

    segs = os.listdir(tsr_anno_root)
    segs.sort()

    for segid in segs:
        ret[segid] = {}
        tsr_seg_anno_path = os.path.join(tsr_anno_root, segid)
        anno_files = os.listdir(tsr_seg_anno_path)
        anno_files.sort()

        for anno_file in anno_files:
            anno_name = os.path.splitext(anno_file)[0]
            ts, index = anno_name.split("+")
            if segid not in ret:
                ret[segid] = {}
            ret[segid][ts] = {}
            ret[segid][ts]["index"] = index
            ret[segid][ts]["anno_path"] = os.path.join(tsr_seg_anno_path, anno_file)
    return ret

def node_main(run_config: dict, tsr_anno_root:str):
    if tsr_anno_root is None:
        print("tsr_anno_root is None, skip submit")
        return 
    tsr_annos = parse_tsr_anno_by_directory(tsr_anno_root)
    if len(tsr_annos) == 0:
        print("tsr_anno_root is empty, skip submit")
        return
    
    seg_config = run_config["preprocess"]
    seg_mode =  seg_config["seg_mode"]
    tgt_seg_path = seg_config["segment_path"]
    car_name = seg_config["car"]
    rec_cfg = run_config["reconstruction"]
    skip_reconstruct = False
    if rec_cfg["enable"] != "True":
        skip_reconstruct = True
    spec_clips = seg_config.get("spec_clips", None)        
    pre_anno_cfg = run_config["annotation"]
    clip_lane = pre_anno_cfg["clip_lane"]
    clip_obstacle = pre_anno_cfg["clip_obstacle"]
    clip_obstacle_test = pre_anno_cfg["clip_obstacle_test"]
    test_road_gnss_file = pre_anno_cfg["test_gnss_info"]

    anno_res_cfg = run_config["ripples_platform"]["abk_mark_result"]
    clip_lane_anno_path = (
        anno_res_cfg["clip_lane_annotation"]
        if "clip_lane_annotation" in anno_res_cfg
        else None
    )
    clip_obs_anno_path = (
        anno_res_cfg["clip_obstacle_annotation"]
        if "clip_obstacle_annotation" in anno_res_cfg
        else None
    )

    deploy_cfg = run_config["deploy"]
    anno_root = deploy_cfg["clip_submit"]
    subfix = deploy_cfg["data_subfix"]

    seg_names = os.listdir(tgt_seg_path)
    segroot = tgt_seg_path
    seg_names.sort()
    for segid in seg_names:
        if spec_clips is not None:
            go_on = False
            for clip in spec_clips:
                if clip in segid:
                    go_on = True
                    break
            if not go_on:
                continue        
        seg_meta_json = os.path.join(segroot, segid, "multi_meta.json")
        seg_path = os.path.join(segroot, segid)
        if not os.path.exists(seg_meta_json):
            seg_meta_json = os.path.join(segroot, segid, "meta.json")
        reconstruct_path = os.path.join(segroot, segid, "multi_reconstruct")
        if not os.path.exists(reconstruct_path):
            reconstruct_path = os.path.join(segroot, segid, "reconstruct")
        if not skip_reconstruct and (
            not os.path.exists(reconstruct_path)
            or not os.path.exists(seg_meta_json)
            or not os.path.getsize(seg_meta_json)
            or not os.path.getsize(os.path.join(segroot, segid, "gnss.json"))
        ):
            continue

        if skip_reconstruct and (
            not os.path.exists(seg_meta_json)
            or not os.path.getsize(seg_meta_json)
            or not os.path.getsize(os.path.join(segroot, segid, "gnss.json"))
        ):
            continue

        if segid not in tsr_annos:
            continue

        print("Commit segment {}.".format(segid))
        meta_json = open(seg_meta_json)
        meta = json.load(meta_json)
        enable_cams = meta["cameras"]
        meta_json.close()

        lane_anno_path = None
        lane_path = os.path.join(clip_lane, segid)
        if clip_lane_anno_path is not None:
            lane_anno_path = os.path.join(clip_lane_anno_path, segid)

        obstacle_path = os.path.join(clip_obstacle, segid)
        if not os.path.exists(obstacle_path):
            obstacle_path = os.path.join(clip_obstacle_test, segid)

        obstacle_anno_path = None
        if clip_obs_anno_path is not None:
            obstacle_anno_path = os.path.join(clip_obs_anno_path, segid)

        if obstacle_anno_path is None and lane_anno_path is None:
            if tsr_anno_root is None or len(tsr_anno_root) == 0:
                return

        tsr_seg_annos = tsr_annos[segid]

        status, is_test = gen_annotation(
            anno_root,
            subfix,
            lane_anno_path,
            lane_path,
            seg_path,
            obstacle_anno_path,
            obstacle_path,
            tsr_seg_annos,
            enable_cams,
            test_road_gnss_file,
        )

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    lane_anno_path = None
    obs_anno_path = None
    tsr_anno_path = None
    if len(sys.argv) == 5:
        lane_anno_path = sys.argv[2]
        obs_anno_path = sys.argv[3]
        tsr_anno_path = sys.argv[4]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)

    if "abk_mark_result" not in run_config['ripples_platform']:
        run_config["ripples_platform"]["abk_mark_result"] = {}
    
    if lane_anno_path is not None and lane_anno_path != "null":
        run_config["ripples_platform"]["abk_mark_result"]["clip_lane_annotation"] = lane_anno_path
    
    if lane_anno_path == 'null':
        run_config["reconstruction"]['enable'] = 'False'
    
    if obs_anno_path is not None and obs_anno_path != "null":
        run_config["ripples_platform"]["abk_mark_result"]["clip_obstacle_annotation"] = obs_anno_path

    node_main(run_config, tsr_anno_path)

