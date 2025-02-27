from python import handle_ifly_frame, call_gen_night_clips
from utils import combine_calib, CarMeta

import numpy as np
import json
from datetime import datetime
import pandas as pd
import traceback as tb
import os, sys
from loguru import logger

RECORD_FILE = "record.json"
META_FILE = "car_meta.json"


def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def gen_clip_segs(
    source, segment, car_name, calib_path, car_meta: CarMeta, spec_time_list: list
):
    calib_info = combine_calib(calib_path)
    lidar_info = {
        "sensor_name": car_meta.lidar_name,
        "sensor_position": car_meta.lidar_name,
        "sensor_type": "MainLidar",
        "lidar_mode": car_meta.lidar_type,
    }
    bpearl_info = []
    if len(car_meta.bpearl_lidars) > 0:
        for bpearl in car_meta.bpearl_lidars:
            _info = {
                "sensor_name": bpearl,
                "sensor_position": bpearl,
                "sensor_type": "BpearlLidar",
                "lidar_mode": car_meta.bpearl_lidar_type,
            }
            bpearl_info.append(_info)
    inno_info = []
    if len(car_meta.inno_lidars) > 0:
        for inno in car_meta.inno_lidars:
            _info = {
                "sensor_name": inno,
                "sensor_position": inno,
                "sensor_type": "InnoLidar",
                "lidar_mode": car_meta.inno_lidar_type,
            }
            inno_info.append(_info)
    calib_ints = calib_info["intrinsics"]
    calib_ints.append(lidar_info)
    calib_ints.extend(bpearl_info)
    calib_ints.extend(inno_info)

    seg_infos = []
    logger.info(f"......\t{source} cut seg start... {str(datetime.now())}")
    try:

        if "iflytek" in car_meta.dc_system_version:
            seg_infos = handle_ifly_frame(
                source,
                segment,
                car_name,
                distance=0,
                time_interval=0,
                spec_time_list=spec_time_list,
            )
    except Exception as e:
        logger.error(f"Try cut {source} segment failed. as exeption {e}")
        tb.print_exc()
        return []

    logger.info(f"......\t{source} cut seg end... {str(datetime.now())}")
    logger.info("\tcut to {} segs.".format(len(seg_infos)))
    if len(seg_infos) == 0:
        return []

    record_file = os.path.join(source, RECORD_FILE)
    record = None
    if os.path.exists(record_file):
        fp = open(record_file, "r")
        record = json.load(fp)

    if not os.path.exists(segment):
        os.umask(0o002)
        os.makedirs(segment, mode=0o775, exist_ok=True)

    clip_segs = []
    logger.info(f"......\t{source} gen seg info start... {str(datetime.now())}")
    for seg_idx, seg_info in enumerate(seg_infos):
        seg, gnss, vech = seg_info
        seg["calibration"] = calib_info
        seg["record"] = record
        seg["data_tags"] = []
        seg["data_system"] = car_meta.dc_system_version
        seg["car"] = car_meta.car_name
        seg["other_sensors_info"] = car_meta.other_sensors_info
        seg["bpearl_lidars"] = car_meta.bpearl_lidars
        seg["inno_lidars"] = car_meta.inno_lidars
        seg["vision_slot"] = car_meta.vision_slot
        seg["vision_slot_interval"] = car_meta.vision_slot_timeinterval

        seg_id = seg["seg_uid"]
        os.makedirs(os.path.join(segment, seg_id), exist_ok=True)

        # curr_seg_dir = os.path.join(segment, date, seg_id)
        # meta_json = os.path.join(os.path.join(segment, date, seg_id, "meta.json"))

        vech_json = os.path.join(os.path.join(segment, seg_id, "vehicle.json"))
        with open(vech_json, "w") as fp:
            ss = json.dumps(vech, ensure_ascii=False, default=dump_numpy)
            fp.write(ss)

        gnss_json = os.path.join(os.path.join(segment, seg_id, "gnss.json"))
        with open(gnss_json, "w") as fp:
            ss = json.dumps(gnss, ensure_ascii=False, default=dump_numpy)
            fp.write(ss)

        pre_anno_json = os.path.join(os.path.join(segment, seg_id, "pre_anno.json"))
        with open(pre_anno_json, "w") as fp:
            pre_anno = {"point_cloud": []}
            ss = json.dumps(pre_anno, ensure_ascii=False, default=dump_numpy)
            fp.write(ss)
        clip_segs.append(seg)
    logger.info(f"......\t{segment} gen seg info end... {str(datetime.now())}")
    return clip_segs


def main_gen_segs(config: dict, clip_spec_time_list: dict):
    seg_config = config["preprocess"]
    frames_path = seg_config["frames_path"]
    tgt_seg_path = seg_config["segment_path"]
    calib_path = seg_config["calibration_path"]
    seg_mode = seg_config["seg_mode"]
    seg_value = seg_config["seg_value"]
    car_name = seg_config["car"]

    car_meta = CarMeta()
    car_meta_file = os.path.join(calib_path, "car_meta.json")
    with open(car_meta_file, "r") as fp:
        car_meta_dict = json.load(fp)
        car_meta.from_json_iflytek(car_meta_dict)

    clip_segs = []
    clips = os.listdir(frames_path)
    clips.sort()
    for clip in clips:
        if not clip.startswith("202"):
            continue
        if clip not in clip_spec_time_list:
            continue
        spec_time_list = clip_spec_time_list[clip]
        logger.info("Start cut seg {}......".format(clip))
        clip_frame = os.path.join(frames_path, clip)
        curr_clip_segs = gen_clip_segs(
            clip_frame,
            tgt_seg_path,
            car_name,
            calib_path,
            car_meta,
            spec_time_list
        )
        clip_segs.extend(curr_clip_segs)
    logger.info(
        f"......\t{frames_path} total {len(clips)} clips cut segs to {len(clip_segs)} segments end..."
    )
    avail_seg_distance = 0.0
    avail_seg_time = 0.0
    avail_seg_frame = 0
    segs = []
    for seg in clip_segs:
        seg_id = seg["seg_uid"]
        segs.append(seg_id)
        avail_seg_distance += seg["distance"]
        avail_seg_time += seg["time_interval"]
        avail_seg_frame += len(seg["frames"])
        meta_json = os.path.join(os.path.join(tgt_seg_path, seg_id, "meta.json"))
        meta_ss = json.dumps(seg, ensure_ascii=False, default=dump_numpy)
        with open(meta_json, "w") as fp:
            fp.write(meta_ss)
    logger.info(f"\t...total clips' time is {avail_seg_time}")
    logger.info(f"\t...total clips' distance is {avail_seg_distance}")
    logger.info(f"\t...total clips' frame is {avail_seg_frame}")
    logger.info(f"\t...total segments id is {segs}")
    return clip_segs


if __name__ == "__main__":
    # config_file = "./utils/sample_config.json"
    if len(sys.argv) != 3:
        logger.error("Usage: python node_gen_segments.py $config_file $related_day_seg_root")
        sys.exit(1)
    config_file = sys.argv[1]
    related_day_seg_root = sys.argv[2]

    if not os.path.exists(config_file):
        logger.info(f"{config_file} Not Exists.")
        sys.exit(1)

    if not os.path.exists(related_day_seg_root):
        logger.info(f"{related_day_seg_root} Not Exists.")
        sys.exit(1)

    if related_day_seg_root.endswith('xlsx'):
        df = pd.read_excel(related_day_seg_root, skiprows=1)
        night_clips = {}
        for idx, row in df.iterrows():
            _, car, subfix, clip_id, related_day_subfix = row
            if not isinstance(related_day_subfix, str) and not isinstance(
                related_day_subfix, int
            ):
                continue
            night_clips[clip_id] = [car, str(subfix), str(related_day_subfix)]
        clip_cut_segs = call_gen_night_clips(config_file, night_clips)
    else:
        clip_cut_segs = call_gen_night_clips(config_file, related_day_seg_root)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "node_gen_segments.log"))

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    main_gen_segs(run_config, clip_cut_segs)
