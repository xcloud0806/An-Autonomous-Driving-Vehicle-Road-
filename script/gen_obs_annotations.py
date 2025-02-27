import argparse
import os, sys
sys.path.append("../")
import json
import numpy as np
from loguru import logger
import pandas as pd
import traceback

from utils import (
    gen_label_obstacle,
    gen_label_obstacle_static,
    gen_label_obstacle_hpp,
    prepare_infos,
    db_utils
)

logger.add("gen_obs_annotations.log", level="INFO", rotation="10 MB")

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
TEST_ROADS_GNSS = "hefei_wuhu_test_roads.json"

xlsx_files = [
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_1st_2nd.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0516.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_3rd.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0522.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0527.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0603.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0612.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0614.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0615.xlsx"
]

ref_segments_path = [
    "/data_cold2/ripples/chery_13484/custom_seg/lane_change_crimping/20240525_n",
    "/data_cold2/ripples/chery_13484/custom_seg/lane_change_crimping/20240515_n"
]

def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def parse_infos_with_excels(xlsx_files: list):
    ret = {}
    total_cnt = 0
    for xlsx_file in xlsx_files:
        df = pd.read_excel(xlsx_file, skiprows=1)
        _total = df.shape[0]
        total_cnt += _total
        # pool = Pool(processes=16)
        for idx, row in df.iterrows():
            segid, objcnt, speed, daynight, task, car, deploy_subfix = row
            # (objcnt, speed, daynight, task, car, deploy_subfix)
            ret[segid] = {
                "objcnt": objcnt,
                "speed": speed,
                "daynight": daynight,
                "task": task,
                "car": car,
                "deploy_subfix": str(deploy_subfix),
                "segid": segid,
            }
    logger.info(f"EXCEL total_count: {total_cnt}")
    return ret

def gen_annotation(
    seg_anno_path,
    deploy_subfix,
    seg_root_path,
    obstacle_res_path,
    obstacle_data_path,
    test_gnss_json,
):
    annotation = {}
    prepare_obstacle = False

    seg_meta_json = os.path.join(seg_root_path, "multi_meta.json")
    if not os.path.exists(seg_meta_json):
        seg_meta_json = os.path.join(seg_root_path, "meta.json")
    if not os.path.exists(seg_meta_json):
        logger.error(f"{seg_root_path} not exists")
        return 1, None

    seg_meta_fp = open(seg_meta_json, "r")
    meta = json.load(seg_meta_fp)
    calibs = meta["calibration"]
    segid = meta["seg_uid"]
    enable_cams = meta["cameras"]

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
        logger.warning("skip {} obstacle anno data submit".format(meta["seg_uid"]))
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

    if prepare_obstacle:
        seg_submit_path = os.path.join(
            seg_anno_path, "annotation_train", deploy_subfix, segid
        )
        if testing_sets:
            seg_submit_path = os.path.join(
                seg_anno_path, "annotation_test", deploy_subfix, segid
            )
        if not os.path.exists(seg_submit_path):
            os.makedirs(seg_submit_path, mode=0o777, exist_ok=True)
        logger.info(f"...Submit {seg_submit_path}")
        submit_anno_json = os.path.join(seg_submit_path, ANNO_INFO_JSON)
        with open(submit_anno_json, "w") as wfp:
            anno_json_str = json.dumps(
                annotation, ensure_ascii=False, default=dump_numpy
            )
            wfp.write(anno_json_str)
        return 0, testing_sets
    else:
        logger.warning("skip {} commit.".format(meta["seg_uid"]))
        return 1, testing_sets

def query_segs_info(seg_list):
    seg_infos = db_utils.query_seg(seg_list)
    ret = {}
    if seg_infos[0] > 0:
        infos = seg_infos[1]
        for info in infos:
            ret[info["id"]] = info
        
    return ret

def main(obs_res_root, obs_data_root, deploy_seg_infos, dst_root):
    if not os.path.exists(obs_res_root):
        logger.error("{} not exists".format(obs_res_root))
        return 1

    if not os.path.exists(obs_data_root):
        logger.error("{} not exists".format(obs_data_root))
        return 1

    # if not os.path.exists(deploy_excel):
    #     logger.error("{} not exists".format(deploy_excel))
    #     return 1

    segs = os.listdir(obs_data_root)
    segs.sort()
    total_segs = len(segs)
    seg_infos = query_segs_info(segs)
    
    if len(seg_infos) == 0:
        logger.warning("{} not in db".format(obs_res_root))
        # return 1

    if len(segs) != len(seg_infos):
        logger.warning("{} short in db".format(obs_res_root))

    for i, seg in enumerate(segs):
        logger.info("{}/{} {}".format(i, total_segs, seg))
        seg_res_path = os.path.join(obs_res_root, seg)
        seg_data_path = os.path.join(obs_data_root, seg)
        if (
            not os.path.exists(seg_res_path)
            or not os.path.exists(seg_data_path)
        ):
            logger.warning("{} not exists".format(seg))
            continue

        if seg in seg_infos:
            seg_info = seg_infos[seg]
            if not os.path.exists(os.path.join(seg_data_path, f"{seg}_infos.json")):
                logger.warning("{} not exists".format(seg))
                seg_data_path = seg_info['pathMap']['obstacle3dAnnoDataPath']

            
            if seg not in deploy_seg_infos:
                logger.warning("{} not in deploy excel".format(seg))
                task = ""
                deploy_subfix = seg_info["collectionDataDate"]
            else:
                deploy_seg_info = deploy_seg_infos[seg]
                task = deploy_seg_info["task"]
                deploy_subfix = deploy_seg_info["deploy_subfix"]
            seg_path = seg_info["segPath"]        
            dst_root_task = os.path.join(dst_root, task)
            # subfix = seg_info["collectionDataDate"]
        else:
            logger.warning("{} not in db".format(seg))
            if seg not in deploy_seg_infos:
                logger.warning("{} not in deploy excel".format(seg))
                task = ""
                deploy_subfix = seg_info["collectionDataDate"]
            else:
                deploy_seg_info = deploy_seg_infos[seg]
                task = deploy_seg_info["task"]
                deploy_subfix = deploy_seg_info["deploy_subfix"]
            dst_root_task = os.path.join(dst_root, task)

            ref_seg_paths = {}
            for ref in ref_segments_path:
                segs = os.listdir(ref)
                segs.sort()
                for _seg in segs:
                    seg_path = os.path.join(ref, _seg)
                    if _seg in ref_seg_paths:
                        logger.warning("{} already in {}".format(_seg, ref))
                        continue
                    ref_seg_paths[_seg] = seg_path
            seg_path = ref_seg_paths[seg]

        ret, testing_sets = gen_annotation(
            dst_root_task,
            deploy_subfix,
            seg_path,
            seg_res_path,
            seg_data_path,
            TEST_ROADS_GNSS
        )

def run_root_annotation():
    root_paths = [
        # ("/data_autodrive/auto/label_4d/post_delete_annotations/20240512-20240520", "/data_autodrive/auto/label_4d/post_delete/20240512", "/data_autodrive/auto/label_4d/post_delete_submit/20240512"),
        # ("/data_autodrive/auto/label_4d/post_delete_annotations/20240519-20240528", "/data_autodrive/auto/label_4d/post_delete/20240519", "/data_autodrive/auto/label_4d/post_delete_submit/20240519"),
        # ("/data_autodrive/auto/label_4d/post_delete_annotations/first-20240527", "/data_autodrive/auto/label_4d/post_delete/first", "/data_autodrive/auto/label_4d/post_delete_submit/first"),
        # ("/data_autodrive/auto/label_4d/post_delete_annotations/20240522-20240613", "/data_autodrive/auto/label_4d/post_delete/20240522", "/data_autodrive/auto/label_4d/post_delete_submit/20240522_0614"),
        # ("/data_autodrive/auto/label_4d/post_delete_annotations/20240523-20240624", "/data_autodrive/auto/label_4d/post_delete/20240523", "/data_autodrive/auto/label_4d/post_delete_submit/20240523_0624"),
        # ("/data_autodrive/auto/label_4d/post_delete_annotations/20240527-20240613", "/data_autodrive/auto/label_4d/post_delete/20240527", "/data_autodrive/auto/label_4d/post_delete_submit/20240527_0614"),
        # ("/data_autodrive/auto/label_4d/post_delete_annotations/20240603-20240626", "/data_autodrive/auto/label_4d/post_delete/20240603", "/data_autodrive/auto/label_4d/post_delete_submit/20240603_0626"),
        # ("/data_autodrive/auto/label_4d/post_delete_annotations/20240527-20240626", "/data_autodrive/auto/label_4d/post_delete/20240527", "/data_autodrive/auto/label_4d/post_delete_submit/20240527_0626"),
        # ("/data_autodrive/auto/label_4d/post_delete_annotations/20240527-20240701", "/data_autodrive/auto/label_4d/post_delete/20240527", "/data_autodrive/auto/label_4d/post_delete_submit/20240527_0701"),
        # ("/data_autodrive/auto/label_4d/post_delete_annotations/20240614-20240703", "/data_autodrive/auto/label_4d/post_delete/20240614", "/data_autodrive/auto/label_4d/post_delete_submit/20240614_0703"),
        ("/data_autodrive/auto/label_4d/refined_annotations/20240522-28881001", "/data_autodrive/auto/label_4d/refined/20240522", "/data_autodrive/auto/label_4d/refined_submit/20240522_28881001"),
        ("/data_autodrive/auto/label_4d/refined_annotations/20240523-28881001", "/data_autodrive/auto/label_4d/refined/20240523", "/data_autodrive/auto/label_4d/refined_submit/20240523_28881001"),
        # ("/data_autodrive/auto/label_4d/refined_annotations/20240529-20240628", "/data_autodrive/auto/label_4d/refined/20240529", "/data_autodrive/auto/label_4d/refined_submit/20240529_0628"),
        # ("/data_autodrive/auto/label_4d/refined_annotations/20240607-20240628", "/data_autodrive/auto/label_4d/refined/20240607", "/data_autodrive/auto/label_4d/refined_submit/20240607_0628"),
        # ("/data_autodrive/auto/label_4d/refined_annotations/20240611-20240702", "/data_autodrive/auto/label_4d/refined/20240611", "/data_autodrive/auto/label_4d/refined_submit/20240611_0702"),
        # ("/data_autodrive/auto/label_4d/refined_annotations/20240611-20240703", "/data_autodrive/auto/label_4d/refined/20240611", "/data_autodrive/auto/label_4d/refined_submit/20240611_0703"),
        # ("/data_autodrive/auto/label_4d/refined_annotations/20240527-20240607", "/data_autodrive/auto/label_4d/refined/20240527", "/data_autodrive/auto/label_4d/refined_submit/20240527_0607"),
        # ("/data_autodrive/auto/label_4d/refined_annotations/20240527-20240613", "/data_autodrive/auto/label_4d/refined/20240527", "/data_autodrive/auto/label_4d/refined_submit/20240527_0613"),
        # ("/data_autodrive/auto/label_4d/refined_annotations/20240527-20240619", "/data_autodrive/auto/label_4d/refined/20240527", "/data_autodrive/auto/label_4d/refined_submit/20240527_0619"),
        # ("/data_autodrive/auto/label_4d/refined_annotations/20240607-20240614", "/data_autodrive/auto/label_4d/refined/20240607", "/data_autodrive/auto/label_4d/refined_submit/20240607_0614"),
        # ("/data_autodrive/auto/label_4d/refined_annotations/20240607-20240619", "/data_autodrive/auto/label_4d/refined/20240607", "/data_autodrive/auto/label_4d/refined_submit/20240607_0619"),
    ]

    deploy_seg_infos = parse_infos_with_excels(xlsx_files)
    
    for root_anno, root_data, root_dst in root_paths:
        types = os.listdir(root_data)
        for t in types:
            type_root_data = os.path.join(root_data, t)
            if not os.path.isdir(type_root_data):
                continue
            cars = os.listdir(type_root_data)
            for car in cars:
                car_root_data = os.path.join(type_root_data, car)
                dates = os.listdir(car_root_data)
                for date in dates:
                    car_date_root_data = os.path.join(car_root_data, date)
                    car_date_root_anno = os.path.join(root_anno, t, car)
                    # car_date_root_data = os.path.join(root_data, car, date)
                    car_date_root_dst = os.path.join(root_dst, car)
                    if not os.path.exists(car_date_root_anno):
                        logger.warning("{} not exists".format(car_date_root_anno))
                        continue
                    if not os.path.exists(car_date_root_data):
                        logger.warning("{} not exists".format(car_date_root_data))
                        continue
                    if not os.path.exists(car_date_root_dst):
                        os.makedirs(car_date_root_dst, mode=0o777, exist_ok=True)
                    logger.info("...Submit {}.{} -> {}".format(car, date, car_date_root_dst))
                    try:
                        main(car_date_root_anno, car_date_root_data, deploy_seg_infos, car_date_root_dst)
                    except Exception as e:
                        logger.error(e)
                        traceback.print_exc()

def run_root_annotation_v2():
    root_paths = [
        # ("/data_autodrive/auto/label_4d/first_select_annotations/20240527-20240701", "/data_autodrive/auto/label_4d/post_delete/20240527", "/data_autodrive/auto/label_4d/post_delete_submit/20240527_0701"),
        (
            "/data_autodrive/auto/label_4d/second_no_annotation_annotations",
            "/data_autodrive/auto/label_4d/second_no_annotation",
            "/data_autodrive/auto/label_4d/post_delete_submit/second_no_annotation_submit",
        ),
    ]

    deploy_seg_infos = parse_infos_with_excels(xlsx_files)

    for root_anno, root_data, root_dst in root_paths:        
        cars = os.listdir(root_data)
        for car in cars:
            car_root_data = os.path.join(root_data, car)
            dates = os.listdir(car_root_data)
            for date in dates:
                car_date_root_data = os.path.join(car_root_data, date)
                car_date_root_anno = os.path.join(root_anno, car, date)
                # car_date_root_data = os.path.join(root_data, car, date)
                car_date_root_dst = os.path.join(root_dst, car)
                if not os.path.exists(car_date_root_anno):
                    logger.warning("{} not exists".format(car_date_root_anno))
                    continue
                if not os.path.exists(car_date_root_data):
                    logger.warning("{} not exists".format(car_date_root_data))
                    continue
                if not os.path.exists(car_date_root_dst):
                    os.makedirs(car_date_root_dst, mode=0o777, exist_ok=True)
                logger.info("...Submit {}.{} -> {}".format(car, date, car_date_root_dst))
                try:
                    main(car_date_root_anno, car_date_root_data, deploy_seg_infos, car_date_root_dst)
                except Exception as e:
                    logger.error(e)
                    traceback.print_exc()

def run_args():
    def parse_args():
        parser = argparse.ArgumentParser(description="gen_obs_annotations")
        parser.add_argument(
            "--obs_res_root", '-a',
            type=str,
            default="",
            help="obs_res_root",
        )
        parser.add_argument(
            "--obs_data_root", '-i',
            type=str,
            default="",
            help="obs_data_root",
        )
        parser.add_argument(
            "--dst_root", '-o',
            type=str,
            default="",
            help="dst_root",
        )
        
        args = parser.parse_args()
        return args
    
    args = parse_args()
    if not os.path.exists(args.dst_root):
        os.makedirs(args.dst_root, mode=0o777, exist_ok=True)
    main(args.obs_res_root, args.obs_data_root, {}, args.dst_root)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_args()
    else:
        run_root_annotation_v2()
