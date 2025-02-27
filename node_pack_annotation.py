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
from pyquaternion import Quaternion

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
ANNO_INFO_DRIVER_OBSTACLE_STATIC_KEY = "driver_obstacle_static"

def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def transform_points3d_to_Rt_size(points_3d):
    # points_3d, array [8, 3]
    #        | z                  3_______7
    #        |                  2/______6/|
    #        |_______ y         | 0_ _ _|_4
    #       /                   1/______5/
    #      / x    
    points_3d = [[pt['x'], pt['y'], pt['z']] for pt in points_3d]
    points_3d = np.array(points_3d, dtype=np.float32)
    vector = points_3d[1] - points_3d[0]
    rotation_z = np.arctan2(vector[1], vector[0])
    rotation = np.array([0, 0, rotation_z], dtype=np.float32)
    location = np.mean(points_3d, axis=0)
    size_x = np.linalg.norm(points_3d[1] - points_3d[0])
    size_y = np.linalg.norm(points_3d[4] - points_3d[0])
    size_z = np.linalg.norm(points_3d[3] - points_3d[0])
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(axis=[0, 0, 1], angle=rotation_z).rotation_matrix # 3Dbbox只有偏移角不为 0
    transform[:3, 3] = location
    return transform, np.array([size_x, size_y, size_z], dtype=np.float32), rotation, location

def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]

def gen_driver_static_obstacles_labels(platform_label_path, transform_matrix_path):
    platform_label = json.load(open(platform_label_path))["polygon"] # list of instance label
    transform_matrix = json.load(open(transform_matrix_path))
    recon_img_to_world = transform_matrix['img_to_world']
    recon_img_to_world = np.array(recon_img_to_world)
    labels_list = []
    for instance in platform_label:
        label = dict()
        uvz = np.array(instance['points'], dtype=np.float32) # [4, 3], bottom rectangle
        xyz = transform_points(uvz, recon_img_to_world)
        short_edge = np.linalg.norm(xyz[0, :2] - xyz[1, :2]) <= np.linalg.norm(xyz[2, :2] - xyz[1, :2])
        if not short_edge:
            # 不符合短边优先, 修改顺序
            xyz = xyz[[0, 3, 2, 1], :]
            uvz = uvz[[0, 3, 2, 1], :]
        xyz[:, 2] = uvz[:, 2]
        high_xyz = xyz.copy()
        high_xyz[:, 2] += instance['altitude']
        xyz = np.concatenate([xyz[[0, 1], :], high_xyz[[1, 0], :], xyz[[3, 2], :], high_xyz[[2, 3], :]], axis=0)
        label['points_3d'] = xyz.tolist()
        points3d = [{'x': point[0], 'y': point[1], 'z': point[2]} for point in xyz]
        pose, size, rotation, location = transform_points3d_to_Rt_size(points3d)
        label['pose'] = pose.tolist()
        label['box_3d'] = dict()
        label['box_3d']['size'] = size.tolist()
        label['box_3d']['rotation'] = rotation.tolist()
        label['box_3d']['location'] = location.tolist()
        label['static'] = instance['static']
        label['velocity'] = [None, None, None]
        label['class_name'] = instance['type']['jingtaizhangaiwushuxing']
        label['class-name'] = instance['type']['jingtaizhangaiwushuxing']
        label['id'] = instance["Id"]
        labels_list.append(label)
    return labels_list


# bbox_list_transform = transform_static_obstacles_labels(anno_path, join(recon_path, 'transform_matrix.json'))

def gen_annotation(
    seg_anno_path,
    seg_subfix,
    lane_anno_res_path,
    lane_anno_data_path,
    seg_root_path,
    obstacle_res_path,
    obstacle_data_path,
    enable_cams,
    test_gnss_json,
    driver_transform_matrix_path,
    driver_clip_lane_annotation    
):
    annotation = {}

    prepare_lane = False
    prepare_obstacle = False

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

        rgb_json_path = os.path.join(driver_clip_lane_annotation, segid)
        file_list = os.listdir(rgb_json_path)
        rgb_json_file = os.path.join(rgb_json_path, file_list[0])
        annotation[ANNO_INFO_DRIVER_OBSTACLE_STATIC_KEY] = gen_driver_static_obstacles_labels(
                                                                rgb_json_file,
                                                                driver_transform_matrix_path)

    if prepare_obstacle or prepare_lane:
        seg_submit_path = os.path.join(
            seg_anno_path, "annotation_train", seg_subfix, segid
        )
        if testing_sets:
            seg_submit_path = os.path.join(
                seg_anno_path, "annotation_test", seg_subfix, segid
            )
        if not os.path.exists(seg_submit_path):
            os.makedirs(seg_submit_path, mode=0o777, exist_ok=True)
        print(f"...Submit {seg_submit_path}")
        submit_anno_json = os.path.join(seg_submit_path, ANNO_INFO_JSON)
        with open(submit_anno_json, "w") as wfp:
            anno_json_str = json.dumps(
                annotation, ensure_ascii=False, default=dump_numpy
            )
            wfp.write(anno_json_str)
        return 0, testing_sets
    else:
        print("skip {} commit.".format(meta["seg_uid"]))
        return 1, testing_sets


def node_main(run_config: dict):
    seg_config = run_config["preprocess"]
    seg_mode =  seg_config["seg_mode"]
    tgt_seg_path = seg_config["segment_path"]
    car_name = seg_config["car"]
    rec_cfg = run_config["reconstruction"]
    skip_reconstruct = False
    if rec_cfg["enable"] != "True":
        skip_reconstruct = True
        
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
    spec_clips = seg_config.get("spec_clips", None)
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
            return

        driver_transform_matrix_path = os.path.join(lane_path, "transform_matrix.json")
        driver_clip_lane_annotation = clip_obstacle.replace("clip_obstacle", "clip_lane_annotation")
        status, is_test = gen_annotation(
            anno_root,
            subfix,
            lane_anno_path,
            lane_path,
            seg_path,
            obstacle_anno_path,
            obstacle_path,
            enable_cams,
            test_road_gnss_file,
            driver_transform_matrix_path,
            driver_clip_lane_annotation
        )
    try:
        obtain_seg_tag(run_config)
    except Exception as e:
        print(f"Caught an exception of type {type(e).__name__}: {e}")

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    lane_anno_path = None
    obs_anno_path = None
    if len(sys.argv) == 4:
        lane_anno_path = sys.argv[2]
        obs_anno_path = sys.argv[3]

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

    node_main(run_config)

