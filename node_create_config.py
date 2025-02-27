import copy
import json
import os, sys
import pandas as pd
from multiprocessing import pool
import shutil
import hashlib
import traceback as tb
import string
import numpy as np
from loguru import logger

reconstruct_cameras = {
    "sihao_0fx60": ["surround_rear_120_8M", "surround_front_120_8M"],
    "sihao_8j998": ["surround_rear_120_8M", "surround_front_120_8M"],
    "sihao_d77052": ["surround_rear_120_8M", "surround_front_120_8M"],
}
default_reconstruct_cameras = [
    "ofilm_surround_front_120_8M",
    "ofilm_surround_rear_100_2M",
]

def generate_random_string(length=8):
    alphabet = string.ascii_letters + string.digits
    return ''.join(np.random.choice(list(alphabet), length))

def parse_luce_xls(xls_file):
    df = pd.read_excel(xls_file, skiprows=1)
    col_lists = [df[col].tolist() for col in df.columns]
    ids = col_lists[0]
    weathers = col_lists[1]
    day_night = col_lists[2]
    road_types = col_lists[3]
    curve_types = col_lists[4]
    lane_clarity = col_lists[5]
    citys = col_lists[6]
    problems = col_lists[8]
    problem_links = col_lists[9]
    clip_paths = col_lists[10]

    ret = {}
    origin_frame_root = None
    for idx, ids in enumerate(ids):
        clip_path = clip_paths[idx]
        if not isinstance(clip_path, str) :
            continue
        clip_path = clip_path.rstrip("/")
        if not os.path.exists(clip_path):
            continue
        clip_id = os.path.basename(clip_path)
        clip_tag = {
            "weather": weathers[idx],
            "day_night": day_night[idx],
            "road_type": road_types[idx],
            "curve_type": curve_types[idx],
            "lane_clarity": lane_clarity[idx],
            "city": citys[idx],
            "problem": problems[idx],
            "problem_link": problem_links[idx],
            "clip_path": clip_paths[idx],
            "clip_id": clip_id,
        }
        if origin_frame_root is None:
            origin_frame_root = os.path.dirname(clip_path)
        else:
            if origin_frame_root != os.path.dirname(clip_path):
                logger.error(f"clip_path {origin_frame_root} <-> {clip_path} is not consistent")
                continue
        ret[clip_id] = clip_tag
    return ret

def multi_process_error_callback(error):
    # get the current process
    process = os.getpid()
    # report the details of the current process
    print(f"Callback Process: {process}, Exeption {error}", flush=True)

def copy_file_with_structure(src, dst_root):
    """拷贝单个文件，并保持目录结构"""
    # relative_path = os.path.relpath(os.path.dirname(src), start=os.path.dirname(dst_root))
    # dst_dir = os.path.join(dst_root, relative_path)
    if not os.path.exists(dst_root):
        os.makedirs(dst_root, exist_ok=True)
    dst = os.path.join(dst_root, os.path.basename(src))        
    shutil.copy2(src, dst)
    return 

def parallel_copy_with_structure(src_root, dst_root, num_processes=16):
    """并行拷贝带有二级目录结构的文件"""
    files_to_copy = []
    level1 = os.listdir(src_root)
    for item1 in level1:
        src = os.path.join(src_root, item1)
        if os.path.isdir(src):
            level2 = os.listdir(src)
            for item2 in level2:
                src2 = os.path.join(src, item2)
                files_to_copy.append((src2, os.path.join(dst_root, item1)))
        else:
            files_to_copy.append((src, dst_root))
    print(f"{len(files_to_copy)} files to copy")        
    p = pool.Pool(processes=num_processes)
    for _src, _dst in files_to_copy:
        p.apply_async(copy_file_with_structure, args=(_src, _dst), error_callback=multi_process_error_callback)
        # copy_file_with_structure(_src, _dst)
    p.close()
    p.join()

def handle_luce_clip(clip_id, clip_info, dst_path):
    clip_path = clip_info["clip_path"]
    if not os.path.exists(clip_path):
        print(f"{clip_id} path not exists")
        return

    print(f"{clip_id} handling...")
    tag_info = dict()
    clip_tag_json = os.path.join(clip_path, "tag_info.json")
    if os.path.exists(clip_tag_json):
        with open(clip_tag_json, "r") as fp:
            tag_info = json.load(fp)

    tag_info["luce_info"] = clip_info
    frame_path = os.path.join(dst_path, clip_id)
    if os.path.exists(frame_path):
        return 
    os.makedirs(frame_path, exist_ok=True)
    # os.system(f"cp -rf {clip_path} {dst_path}/")
    with open(os.path.join(frame_path, "tag_info.json"), "w") as fp:
        ss = json.dumps(tag_info)
        fp.write(ss)

    parallel_copy_with_structure(clip_path, frame_path)


def node_main(init_config: dict, run_config: dict):
    frames_path = init_config["frames_path"]
    car = init_config["car_name"]
    target_anno_root = init_config["target_anno_output"]
    calib_path = init_config["calibration_path"]
    if "seg_mode" in init_config:
        seg_mode = init_config["seg_mode"]
    else:
        seg_mode = run_config["preprocess"]["seg_mode"]
    multi_seg = False
    if "multi_seg" in run_config and run_config['multi_seg']['enable'] == "True":
        multi_seg = True
    # rdg_root = "/train30/cv2/permanent/brli/ripples_platform"
    # 切换目录到涛哥目录
    rdg_root_train30 = "/train30/cv2/permanent/taoguo/ripples_platform"
    rdg_root_mix01 = "/yfw-b3-mix01/cv2/permanent/taoguo/ripples_platform"

    if seg_mode == "luce" or seg_mode == "hpp_luce" or seg_mode == 'aeb':
        task = ""
        pattern = "luce_frame"
        root_path = f"/data_cold2/origin_data/{car}"
        subfix = os.path.basename(frames_path)
        if "luce_excel_path" in init_config:
            luce_xls = init_config["luce_excel_path"]
            hash_string = hashlib.shake_128(luce_xls.encode("utf-8")).hexdigest(4)
            subfix = f"{subfix}_{hash_string}"
    else:
        tt = frames_path.split("/")
        root_path = f"/{tt[1]}/{tt[2]}/{car}"
        task = ""
        subfix = tt[-1]
        pattern = tt[-2]
        if len(tt) == 7:
            task = tt[-2]
            pattern = tt[-3]
        elif len(tt) == 8:
            tasks = [tt[-3], tt[-2]]
            task = "/".join(tasks)
            pattern = tt[-4]
        elif len(tt) == 9:
            tasks = [tt[-4], tt[-3], tt[-2]]
            task = "/".join(tasks)
            pattern = tt[-5]
        else:
            if len(tt) != 6:
                print("frames path max support 3 task name.")
                sys.exit(1)

    seg_pattern = pattern.replace("frame", "seg")
    if len(task) > 0:
        segment_path = os.path.join(root_path, seg_pattern, task, subfix)
        clip_lane = os.path.join(target_anno_root, car, task, "clip_lane", subfix)
        clip_obstacle = os.path.join(
            target_anno_root, car, task, "clip_obstacle", subfix
        )
        clip_obstacle_test = os.path.join(
            target_anno_root, car, task, "clip_obstacle_test", subfix
        )
        clip_check = os.path.join(target_anno_root, car, task, "clip_check", subfix)
        clip_submit = os.path.join(target_anno_root, car, task, "clip_submit")
        clip_submit_data = os.path.join(
            target_anno_root, car, task, "clip_submit", "data"
        )
        # 数采数据下使用mix01 存储
        rdg_root = rdg_root_mix01
        tgt_rdg_path = os.path.join(rdg_root, task, car, "data")
        tgt_rdg_anno_path = os.path.join(rdg_root, task, car, "annos")
        tgt_rdg_anno_abk_tmp_path = os.path.join(rdg_root, task, car, "annos_abk")
        tgt_rdg_anno_autolabel_tmp = os.path.join(
            rdg_root, task, car, "annos_autolabel"
        )
        tgt_rdg_deploy_path = os.path.join(rdg_root, task, car, "deploy")
        tgt_rdg_anno_abk_tmp_path = os.path.join(rdg_root, task, car, "annos_abk")
        tgt_rdg_anno_autolabel_tmp = os.path.join(
            rdg_root, task, car, "annos_autolabel"
        )
    else:
        segment_path = os.path.join(root_path, seg_pattern, subfix)
        clip_lane = os.path.join(target_anno_root, car, "clip_lane", subfix)
        clip_obstacle = os.path.join(target_anno_root, car, "clip_obstacle", subfix)
        clip_obstacle_test = os.path.join(
            target_anno_root, car, "clip_obstacle_test", subfix
        )
        clip_check = os.path.join(target_anno_root, car, "clip_check", subfix)
        clip_submit = os.path.join(target_anno_root, car, "clip_submit")
        clip_submit_data = os.path.join(target_anno_root, car, "clip_submit", "data")
        # 数采数据下使用mix01 存储，路测模式下的数据使用train30 存储
        if "luce" not in seg_mode and 'aeb' not in seg_mode:
            rdg_root = rdg_root_mix01
        else:
            rdg_root = rdg_root_train30
        tgt_rdg_path = (
            os.path.join(rdg_root, "common", car, "data")
            if "luce" not in seg_mode and 'aeb' not in seg_mode
            else os.path.join(rdg_root, "luce", car, "data")
        )
        tgt_rdg_anno_path = (
            os.path.join(rdg_root, "common", car, "annos")
            if "luce" not in seg_mode  and 'aeb' not in seg_mode
            else os.path.join(rdg_root, "luce", car, "annos")
        )
        tgt_rdg_anno_abk_tmp_path = (
            os.path.join(rdg_root, "common", car, "annos_abk")
            if "luce" not in seg_mode  and 'aeb' not in seg_mode
            else os.path.join(rdg_root, "luce", car, "annos_abk")
        )
        tgt_rdg_anno_autolabel_tmp = (
            os.path.join(rdg_root, "common", car, "annos_autolabel")
            if "luce" not in seg_mode  and 'aeb' not in seg_mode
            else os.path.join(rdg_root, "luce", car, "annos_autolabel")
        )
        tgt_rdg_deploy_path = (
            os.path.join(rdg_root, "common", car, "deploy")
            if "luce" not in seg_mode  and 'aeb' not in seg_mode
            else os.path.join(rdg_root, "luce", car, "deploy")
        )

    if car not in reconstruct_cameras:
        map_camera_name = default_reconstruct_cameras[0]
        map_camera_name_add = default_reconstruct_cameras[1]
    else:
        map_camera_name = reconstruct_cameras[car][0]
        map_camera_name_add = reconstruct_cameras[car][1]

    config = copy.deepcopy(run_config)
    if multi_seg:
        multi_info_path = segment_path.replace("_seg", "_coll")
        multi_info_path = multi_info_path.rstrip("/")
        rand_id = generate_random_string()
        config["multi_seg"]["multi_info_path"] = multi_info_path + f"_{rand_id}"
        config['multi_seg']['segment_path'] = segment_path

    if "method" in init_config:
        config["method"] = init_config["method"]
    else:
        config["method"] = "ripple"
    config["preprocess"]["frames_path"] = frames_path
    config["preprocess"]["segment_path"] = segment_path
    config["preprocess"]["calibration_path"] = calib_path
    config["preprocess"]["target_anno_output"] = target_anno_root
    config["preprocess"]["car"] = car
    if "seg_mode" in init_config:
        config["preprocess"]["seg_mode"] = init_config["seg_mode"]
    if seg_mode == 'luce' or seg_mode == 'hpp_luce' or seg_mode == 'aeb':
        spec_clips = init_config.get("spec_clips", None)
        config["preprocess"]["spec_clips"] = spec_clips

    car_meta_json = os.path.join(calib_path, "car_meta.json")
    with open(car_meta_json, "r") as f:
        car_meta = json.load(f)
    
    _map_camera_name = car_meta.get("map_campera_name", map_camera_name)
    config["reconstruction"]["map_camera_name"] = _map_camera_name
    _map_camera_name_add = car_meta.get("map_camera_name_add", "")
    config["reconstruction"]["map_camera_name_add"] = _map_camera_name_add

    config["annotation"]["clip_lane"] = clip_lane
    config["annotation"]["clip_obstacle"] = clip_obstacle
    config["annotation"]["clip_obstacle_test"] = clip_obstacle_test
    config["annotation"]["clip_check"] = clip_check

    config["deploy"]["clip_submit"] = clip_submit
    config["deploy"]["clip_submit_data"] = clip_submit_data
    config["deploy"]["data_subfix"] = subfix
    config["deploy"]["tgt_rdg_path"] = tgt_rdg_path
    config["deploy"]["tgt_rdg_anno_path"] = tgt_rdg_anno_path
    config["deploy"]["tgt_rdg_anno_abk_tmp"] = tgt_rdg_anno_abk_tmp_path
    config["deploy"]["tgt_rdg_anno_autolabel_tmp"] = tgt_rdg_anno_autolabel_tmp
    config["deploy"]["tgt_rdg_deploy_path"] = tgt_rdg_deploy_path

    if "4DAutoLabelParams" not in run_config:
        config["4DAutoLabelParams"] = dict()
        config["4DAutoLabelParams"]["target_anno_output"] = os.path.join(
            tgt_rdg_path, subfix, "work_data"
        )

    config["4DAutoLabelParams"]["frames_path"] = os.path.join(tgt_rdg_path, subfix)
    config["4DAutoLabelParams"]["car_name"] = car
    config["4DAutoLabelParams"]["date"] = subfix

    if "RoadAutoLabelParams" not in run_config:
        config["RoadAutoLabelParams"] = dict()
        config["RoadAutoLabelParams"][
            "output_dir"
        ] = "/train30/cv2/permanent/shuaixiong/lianyi_platform/auto_label_model/auto_mark_output_annos"
        config["RoadAutoLabelParams"][
            "result_dir"
        ] = "/train30/cv2/permanent/shuaixiong/lianyi_platform/auto_label_model/auto_mark_output"

    config["RoadAutoLabelParams"]["data_path"] = os.path.join(tgt_rdg_path, subfix)
    config["RoadAutoLabelParams"]["car_name"] = car
    config["RoadAutoLabelParams"]["date"] = subfix

    return config


if __name__ == "__main__":
    config_file = "./utils/sample_init.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "node_create_config.log"))

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
        demand_cfg = run_config["ripples_platform_demand"]
    init_config = dict()
    if "luce_excel_path" in demand_cfg:
        car_name = demand_cfg["car_name"]
        target_anno_root = demand_cfg["target_anno_output"]
        calib_path = demand_cfg["calibration_path"]

        luce_xls = demand_cfg["luce_excel_path"]
        hash_string = hashlib.shake_128(luce_xls.encode("utf-8")).hexdigest(4)
        # luce_origin_path = os.path.dirname(luce_xls)
        # subfix = os.path.basename(luce_origin_path)
        # frame_path = os.path.join(
        #     f"/data_cold2/origin_data/{car_name}/luce_frame/{subfix}_{hash_string}"
        # )
        # logger.info(f"luce_origin_path: {frame_path}")
        try:
            clips = parse_luce_xls(luce_xls)
        except Exception as e:
            tb.print_exc()
            logger.error(f"parse luce excel failed as {e}")
            sys.exit(1)
        clip_ids = list(clips.keys())
        origin_luce_frame_root = None
        for clip_id, clip_info in clips.items():
            logger.info(f"clip_id: {clip_id}")
            clip_path = clip_info["clip_path"]
            if origin_luce_frame_root is None:
                origin_luce_frame_root = os.path.dirname(clip_path)
            tag_info = dict()
            clip_tag_json = os.path.join(clip_path, "tag_info.json")
            if os.path.exists(clip_tag_json):
                with open(clip_tag_json, "r") as fp:
                    tag_info = json.load(fp)

            tag_info["luce_info"] = clip_info
            with open(os.path.join(clip_path, "tag_info.json"), "w") as fp:
                ss = json.dumps(tag_info)
                fp.write(ss)
            # handle_luce_clip(clip_id, clip_info, frame_path)
            
        init_config = {
            "frames_path": origin_luce_frame_root,
            "calibration_path": calib_path,
            "target_anno_output": "/data_autodrive/auto/luce/",
            "car_name": car_name,
            # "seg_mode": "luce",
            "spec_clips": clip_ids,
            "luce_excel_path": luce_xls,
        }
    else:
        init_config = demand_cfg
    updated_config = node_main(init_config, run_config)
    with open(config_file, "w") as fp:
        ss = json.dumps(updated_config)
        fp.write(ss)
