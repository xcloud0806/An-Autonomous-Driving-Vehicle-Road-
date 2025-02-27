import os
import pandas as pd
import sys
import shutil
import numpy as np
import json
from datetime import datetime
import time
import cv2

sys.path.append("../utils")
sys.path.append("../lib/python3.8/site_packages")
import pcd_iter as pcl
from calib_utils import load_calibration, load_bpearls, undistort
from db_utils import query_seg
from loguru import logger
from multiprocessing.pool import Pool
from overpy import Overpass

xlsx_files = [
    # "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_1st_2nd.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0516.xlsx",
    # "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_3rd.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0522.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0527.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0603.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0612.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0614.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/refined/total_objcnt_speed_0615.xlsx",
]
loss_json = (
    "/data_autodrive/users/brli/dev_raw_data/refined/loss_list_0617.json"
)
logger.add("logs/search_segs_complete.log", rotation="10 MB", level="INFO")

DATE = "20240617"
SRC_ROOT = f"/data_autodrive/auto/label_4d/post_delete/{DATE}"
DST_REFINED_ROOT = f"/data_autodrive/auto/label_4d/refined/{DATE}"
DST_POSTDEL_ROOT = f"/data_autodrive/auto/label_4d/post_delete/{DATE}"
# PREANNO_ROOT = f"/data_autodrive/auto/label_4d/result/post_delete/{DATE}"
PREANNO_ROOT = "/data_autodrive/auto/label_4d/result/post_delete/"
DST_INFO_ROOT = f"/data_autodrive/auto/label_4d/post_delete/{DATE}/info"

MAX_LOST_LIMIT = 2
INFO_FILE = "infos.json"
MAX_FRAMES = 80
PICK_INTERVAL = 5  # 10 * 0.5
DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]


def prepare_obstacle(idx, total, segid, segment_path, dst_path, mode="common"):
    def convert_frame(frame_idx, frame, meta, dst_path, cameras: list):
        frame_dir = meta["frames_path"]
        bpearl_lidars = meta["bpearl_lidars"]
        inno_lidars = meta["inno_lidars"]
        bpearl_exts = load_bpearls(meta)

        frame_info = {}

        lidar = frame["lidar"]
        pose = lidar["pose"]

        images = frame["images"]
        img_info = {}
        for cam in cameras:
            cam_calib = load_calibration(meta, cam)
            if cam in images:
                cam_img = images[cam]
                img_src = os.path.join(frame_dir, cam_img["path"])
                img_src_arr = cv2.imread(img_src)  # type: ignore
                img_dst = os.path.join(
                    dst_path, "image_frames", cam, "{}.jpg".format(frame_idx)
                )
                img_dst_arr = undistort(cam_calib, img_src_arr)
                cv2.imwrite(img_dst, img_dst_arr)  # type: ignore

                img_info[cam] = {}
                lidar_to_camera = np.concatenate(
                    (cam_calib["extrinsic"], np.array([[0, 0, 0, 1]]))
                )
                camera_to_world = np.matmul(
                    np.array(pose), np.linalg.pinv(lidar_to_camera)
                )
                img_info[cam]["pose"] = camera_to_world.tolist()
                img_info[cam]["timestamp"] = str(cam_img["timestamp"])
                img_info[cam]["frame_id"] = frame_idx
        frame_info["image_frames"] = img_info

        pc_info = {}
        pc_info["pose"] = pose
        pc_info["timestamp"] = str(lidar["timestamp"])
        pc_info["frame_id"] = "{}.pcd".format(frame_idx)

        pc_src = os.path.join(frame_dir, lidar["path"])
        pc_dst = os.path.join(dst_path, "pc_frames", "{}.pcd".format(frame_idx))
        pcd = pcl.LoadXYZI(pc_src)

        bpearls = frame["bpearls"]
        innos = frame["innos"]
        if bpearl_lidars is not None and len(bpearls) > 0 and len(bpearl_lidars) > 0:
            pcd_lst = [pcd]
            for bpearl in bpearl_lidars:
                bpearl_frame = bpearls[bpearl]
                bpearl_pcd_src = os.path.join(frame_dir, bpearl_frame["path"])
                if not os.path.exists(bpearl_pcd_src):
                    print(f"\t{pc_info['timestamp']}loss {bpearl} pcd")
                    continue
                bpearl_pcd = pcl.LoadXYZI(bpearl_pcd_src)
                itensity_mat = bpearl_pcd[:, 3:]
                bpearl_ext = bpearl_exts[bpearl]
                r = bpearl_ext[:3, :3]
                t = bpearl_ext[:3, 3:].reshape((1, 3))
                bpearl_trans_pcd = np.matmul(bpearl_pcd[:, :3], r.T) + t
                bpearl_trans_pcd = np.concatenate(
                    [bpearl_trans_pcd, itensity_mat], axis=-1
                )
                pcd_lst.append(bpearl_trans_pcd)
            if len(pcd_lst) > 1:
                merge_pcd = np.concatenate(pcd_lst, axis=0)
                pcl.SavePcdZ(merge_pcd, pc_dst)
            else:
                pcl.SavePcdZ(pcd, pc_dst)
        elif inno_lidars is not None and len(innos) > 0 and len(inno_lidars) > 0:
            pcd_lst = [pcd]
            for inno in inno_lidars:
                inno_frame = innos[inno]
                inno_pcd_src = os.path.join(frame_dir, inno_frame["path"])
                if not os.path.exists(inno_pcd_src):
                    print(f"\t{pc_info['timestamp']}loss {inno} pcd")
                    continue
                inno_pcd = pcl.LoadXYZI(inno_pcd_src)
                itensity_mat = inno_pcd[:, 3:]
                inno_ext = bpearl_exts[inno]
                r = inno_ext[:3, :3]
                t = inno_ext[:3, 3:].reshape((1, 3))
                inno_trans_pcd = np.matmul(inno_pcd[:, :3], r.T) + t
                inno_trans_pcd = np.concatenate([inno_trans_pcd, itensity_mat], axis=-1)
                pcd_lst.append(inno_trans_pcd)
            if len(pcd_lst) > 1:
                merge_pcd = np.concatenate(pcd_lst, axis=0)
                pcl.SavePcdZ(merge_pcd, pc_dst)
            else:
                pcl.SavePcdZ(pcd, pc_dst)
        else:
            pcl.SavePcdZ(pcd, pc_dst)

        frame_info["pc_frame"] = pc_info
        return frame_info

    meta_json = os.path.join(segment_path, "meta.json")
    meta_fp = open(meta_json, "r")
    meta = json.load(meta_fp)
    enable_cameras = meta["cameras"]
    segid = meta["seg_uid"]
    calib = meta["calibration"]
    calib_sensors = calib["sensors"]
    cameras = [item for item in enable_cameras if item in calib_sensors]
    max_lost_limit = len(cameras) * MAX_LOST_LIMIT
    print(f">>> [{idx}/{total}] {segid} prepare_obstacle...")
    if meta["lost_image_num"] > (max_lost_limit):
        print(
            f"{segid} skip. Because lost too much frame. {meta['lost_image_num']} > {max_lost_limit}"
        )
        return

    first_lidar_pose = np.array(meta["frames"][0]["lidar"]["pose"]).astype(np.float32)
    dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
    if (first_lidar_pose == dft_pose_matrix).all():
        print(f"{segid} not selected .")
        return

    gnss_json = os.path.join(segment_path, "gnss.json")
    gnss_fp = open(gnss_json, "r")
    gnss = json.load(gnss_fp)

    veh_json = os.path.join(segment_path, "vehicle.json")
    veh_fp = open(veh_json, "r")
    vehicle = json.load(veh_fp)

    os.makedirs(os.path.join(dst_path, "image_frames"), mode=0o775, exist_ok=True)
    os.makedirs(os.path.join(dst_path, "pc_frames"), mode=0o775, exist_ok=True)
    for cam in cameras:
        os.makedirs(
            os.path.join(dst_path, "image_frames", cam), mode=0o775, exist_ok=True
        )

    info = {}
    info["calib_params"] = calib
    info["frames"] = []

    frames = meta["frames"]
    frame_cnt = len(frames)
    # only use middle 400M for obstacle anno
    start_idx = int(frame_cnt * 0.25)
    end_idx = int(frame_cnt * 0.75)
    if mode == "luce" or mode == "test":
        start_idx = int(frame_cnt * 0.05)
        end_idx = int(frame_cnt * 0.95)

    cnt = 0
    pre_speed = []
    for f_idx in range(start_idx, end_idx):
        # use frame every 0.5s
        if f_idx % PICK_INTERVAL != 0:
            continue
        # total frame not bigger than 80
        if cnt > MAX_FRAMES:
            break
        frame = frames[f_idx]
        frame_gnss = gnss[str(frame["gnss"])]
        speed = float(frame_gnss["speed"])
        # skip static frame
        if speed == 0:
            frame_veh = vehicle[str(frame["vehicle"])]
            speed = float(frame_veh["vehicle_spd"])
        if speed < 0.01 and sum(pre_speed) < 0.1:
            continue

        if len(pre_speed) > 10:
            pre_speed.pop(0)
        pre_speed.append(speed)

        frame_info = convert_frame(cnt, frame, meta, dst_path, cameras)
        info["frames"].append(frame_info)
        cnt += 1

    with open(os.path.join(dst_path, "{}_infos.json".format(segid)), "w") as fp:
        ss = json.dumps(info)
        fp.write(ss)

    meta_fp.close()
    gnss_fp.close()


def handle_seg(idx, total, segid, src, dst, dst_info):
    print(f">>> [{idx}/{total}] {segid} going...")
    # src_info =  os.path.join(src, f"{segid}_infos.json")
    # shutil.copy(src_info, dst_info)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def judge_China(lon, lat):
    if lon < 70 or lon > 140:
        return True
    if lat < 0 or lat > 55:
        return True
    return False


def get_ways_name(wgs84_lon_lst: list, wgs84_lat_lst: list):
    wgs84_lon_arr = np.array(wgs84_lon_lst)
    wgs84_lat_arr = np.array(wgs84_lat_lst)

    wgs84_lst = [
        np.min(wgs84_lat_lst),
        np.min(wgs84_lon_lst),
        np.max(wgs84_lat_lst),
        np.max(wgs84_lon_lst),
    ]
    lat_0 = wgs84_lst[0]
    lon_0 = wgs84_lst[1]
    lat_1 = wgs84_lst[2]
    lon_1 = wgs84_lst[3]
    query_str = "[out:json][timeout:30];way[highway]({},{},{},{});(._;>;);out;".format(
        lat_0, lon_0, lat_1, lon_1
    )

    api = Overpass()
    result = api.query(query_str)
    if len(result.ways) == 0:
        return {}
    ret = {}
    for way in result.ways:
        way_id = way.id
        if way.tags.get("highway"):
            name = way.tags.get("name:en")
            highway = way.tags.get("highway")
            if name not in ret:
                ret[way_id] = {
                    "highway": highway,
                    "name:en": name,
                    "name:zh": way.tags.get("name"),
                    "infos": way.tags,
                    "count": 1,
                }
            else:
                ret[name]["count"] += 1
    return ret


def get_main_way_by_gnss_json(gnss_json: str):
    gnss_fp = open(gnss_json, "r")
    gnsses = json.load(gnss_fp)
    tss = list(gnsses.keys())
    # indexs = [0, len(tss)/4, len(tss)/2, len(tss)*3/4, len(tss)-1]
    indexs = [
        0,
        len(tss) / 8,
        len(tss) / 4,
        len(tss) * 3 / 8,
        len(tss) / 2,
        len(tss) * 5 / 8,
        len(tss) * 3 / 4,
        len(tss) * 7 / 8,
        len(tss) - 1,
    ]
    pre_lon = float(gnsses[tss[0]]["longitude"])
    pre_lat = float(gnsses[tss[0]]["latitude"])
    ways = {}
    for idx in indexs[1:]:
        ts = tss[int(idx)]
        loc = gnsses[ts]
        lon = float(loc["longitude"])
        lat = float(loc["latitude"])
        if judge_China(lon, lat):
            continue

        cur_ts = int(ts)
        way = get_ways_name([pre_lon, lon], [pre_lat, lat])
        for name, info in way.items():
            if name not in ways:
                ways[name] = info
            else:
                ways[name]["count"] += info["count"]
    # 按照count 对way 进行排序，获取way name
    way_names = sorted(ways.items(), key=lambda x: x[1]["count"], reverse=True)
    roads = {}
    for k, v in way_names:
        if v["name:en"] is None and v["name:zh"] is None:
            continue

        if v["name:zh"] not in roads:
            roads[v["name:zh"]] = v
        else:
            roads[v["name:zh"]]["count"] += v["count"]
    max_cnt = 0
    for k, v in roads.items():
        if v["count"] > max_cnt:
            max_cnt = v["count"]
    ret = []
    for k, v in roads.items():
        if v["count"] == max_cnt:
            ret.append(v)
    return ret


def multi_process_error_callback(error):
    # get the current process
    process = os.getpid()
    # report the details of the current process
    print(f"Callback Process: {process}, Exeption {error}", flush=True)


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


def update_seg_infos(seg_infos):
    ret = {}
    preanno_batches = os.listdir(PREANNO_ROOT)
    for batch in preanno_batches:
        annos_path = os.path.join(PREANNO_ROOT, batch)
        cars = os.listdir(annos_path)
        cars.sort()        
        for car in cars:
            car_annos_root = os.path.join(annos_path, car)
            subfixes = os.listdir(car_annos_root)
            subfixes.sort()
            for subfix in subfixes:
                subfix_path = os.path.join(car_annos_root, subfix)
                segs = os.listdir(subfix_path)
                segs.sort()
                for seg in segs:
                    seg_path = os.path.join(subfix_path, seg)
                    if not os.path.isdir(seg_path):
                        continue
                    if not os.path.exists(
                        os.path.join(seg_path, "annotations", "result.json")
                    ):
                        continue
                    if seg in ret:
                        logger.info(f"[{seg}] already updated")
                        continue
                    if seg not in seg_infos:
                        continue

                    seg_info = seg_infos[seg]
                    speed = float(seg_info["speed"])
                    daynight = seg_info["daynight"]
                    task = seg_info["task"]

                    # res = query_seg([seg])                    
                        
                    result_json = os.path.join(seg_path, "annotations", "result.json")
                    with open(result_json, "r") as fp:
                        result = json.load(fp)
                    frame_annos = result["researcherData"]
                    frame_cnt = len(frame_annos)
                    bbox_cnt = 0
                    for _f in frame_annos:
                        _anno = _f["frame_annotations"]
                        _bbox = len(_anno)
                        bbox_cnt += _bbox
                    ret[seg] = [
                        speed,
                        daynight,
                        task,
                        car,
                        subfix,
                        frame_cnt,
                        bbox_cnt,
                        "",
                        "",
                        "",
                        result_json
                    ]
    return ret

def get_valid_obstacle_path(xlsx_files):
    seg_infos = parse_infos_with_excels(xlsx_files)
    idx = 0
    ret = {}
    for segid, seg_info in seg_infos.items():
        speed = float(seg_info["speed"])
        daynight = seg_info["daynight"]
        task = seg_info["task"]
        deploy_subfix = seg_info["deploy_subfix"]
        car = seg_info["car"]
        idx += 1
        seg_clip_obs_path = ""
        res = query_seg([segid])
        res_cnt = res[0]
        if res_cnt > 0:
            seg_content = res[1][0]
            seg_clip_obs_path = seg_content["pathMap"]["obstacle3dAnnoDataPath"]

        if seg_clip_obs_path == "" or not os.path.exists(seg_clip_obs_path):
            # logger.info(f"{segid}.{seg_clip_obs_path} not exists, create new clip_obstacle")
            # segment_path = seg_content['segPath']
            if task == 'common' or task == 'commom':
                auto_root = os.path.join("/data_autodrive/auto/custom", car)
            else:
                auto_root = os.path.join("/data_autodrive/auto/custom", car, task)
            seg_clip_obs_path = os.path.join(auto_root, "clip_obstacle", deploy_subfix, segid)
            if not os.path.exists(seg_clip_obs_path):
                seg_clip_obs_path = os.path.join(auto_root, "clip_obstacle_test", deploy_subfix, segid)
        
        if os.path.exists(seg_clip_obs_path):
            logger.info(f"{segid}.{seg_clip_obs_path} exists")
            # use_count += 1
            ret[segid] = seg_clip_obs_path
    return ret

def gen_obs_infos():
    seg_infos = parse_infos_with_excels(xlsx_files)
    use_count = 0
    total_count = len(seg_infos)
    pool = Pool(processes=16)
    idx = 0
    for segid, seg_info in seg_infos.items():
        speed = float(seg_info["speed"])
        daynight = seg_info["daynight"]
        task = seg_info["task"]
        deploy_subfix = seg_info["deploy_subfix"]
        car = seg_info["car"]
        dst_info = os.path.join(
            DST_INFO_ROOT, car, deploy_subfix, segid, f"{segid}_infos.json"
        )
        dst_info_dir = os.path.join(DST_INFO_ROOT, car, deploy_subfix, segid)            
        os.makedirs(dst_info_dir, exist_ok=True, mode=0o777)
        idx += 1

        res = query_seg([segid])
        res_cnt = res[0]
        if res_cnt > 0:
            seg_content = res[1][0]
            seg_clip_obs_path = seg_content["pathMap"]["obstacle3dAnnoDataPath"]
            subfix = seg_content["collectionDataDate"]
            car_name = seg_content["calibrationCar"]            
            if not os.path.exists(seg_clip_obs_path):
                # logger.info(f"{segid}.{seg_clip_obs_path} not exists, create new clip_obstacle")
                segment_path = seg_content['segPath']
                if task == 'common' or task == 'commom':
                    auto_root = os.path.join("/data_autodrive/auto/custom", car)
                else:
                    auto_root = os.path.join("/data_autodrive/auto/custom", car, task)
                seg_obs_root = os.path.join(auto_root, "clip_obstacle", deploy_subfix, segid)
                if not os.path.exists(seg_obs_root):
                    seg_obs_root = os.path.join(auto_root, "clip_obstacle_test", deploy_subfix, segid)
                if not os.path.exists(seg_obs_root):
                    logger.warning(f"{seg_obs_root} not exists")
                    continue

                src_info = os.path.join(seg_obs_root, f"{segid}_infos.json")
                if not os.path.exists(src_info):
                    logger.warning(f"{src_info} not exists")
                    continue
                logger.info(f">>1<< {seg_clip_obs_path} going...")
                use_count += 1
                if not os.path.exists(dst_info):
                    shutil.copy(src_info, dst_info)
            else:
                logger.info(f">>2<< {seg_clip_obs_path} going...")
                src_info = os.path.join(seg_clip_obs_path, f"{segid}_infos.json")
                if not os.path.exists(src_info):
                    logger.warning(f"{src_info} not exists")
                    continue
                if not os.path.exists(dst_info):
                    shutil.copy(src_info, dst_info)

                use_count += 1
        else:
            print(f"{segid} not in db.")
            if task == 'common' or task == 'commom':
                auto_root = os.path.join("/data_autodrive/auto/common", car)
            else:
                auto_root = os.path.join("/data_autodrive/auto/custom", car, task)
            seg_obs_root = os.path.join(auto_root, "clip_obstacle", deploy_subfix, segid)
            if not os.path.exists(seg_obs_root):
                seg_obs_root = os.path.join(auto_root, "clip_obstacle_test", deploy_subfix, segid)
            if not os.path.exists(seg_obs_root):
                logger.warning(f"{seg_obs_root} not exists")
                continue

            src_info = os.path.join(seg_obs_root, f"{segid}_infos.json")
            if not os.path.exists(src_info):
                logger.warning(f"{src_info} not exists")
                continue
            logger.info(f">>3<< {seg_obs_root} going...")
            use_count += 1
            if not os.path.exists(dst_info):
                shutil.copy(src_info, dst_info)

    pool.close()
    pool.join()
    logger.info(f"use_count: {use_count}/total_count: {total_count}")


def gen_postdel():
    filters = set([])
    seg_infos = parse_infos_with_excels(xlsx_files)
    seg_infos = update_seg_infos(seg_infos)
    seg_obs_paths = get_valid_obstacle_path(xlsx_files)
    use_count = 0
    total_count = len(seg_infos)
    pool = Pool(processes=16)
    idx = 0
    handled_infos = []
    for segid, row in seg_infos.items():
        idx += 1
        if segid in filters:
            logger.warning(f"{segid} is filtered")
            continue
        (
            speed,
            daynight,
            task,
            car,
            deploy_subfix,
            frame_cnt,
            bbox_cnt,
            _,
            _,
            _,
            result_json
        ) = row
        seg_clip_obs_path = seg_obs_paths.get(segid, None)
        deploy_subfix = str(deploy_subfix)
        if float(speed) < 0.1 and task == "common":
            logger.warning(f"{task}.{segid} speed too low")
            continue
        # if float(speed) > 90 and task != "frwang_chengshilukou":
        #     logger.warning(f"{task}.{segid} run in motorway {speed}")
        #     continue
        # if daynight != "night":
        #     logger.warning(f"{task}.{segid} run in night")
        #     continue

        type = "hard"
        if bbox_cnt > 4000:
            logger.info(f"{task}.{segid} bbox_cnt {bbox_cnt} is hard type")
        elif bbox_cnt > 2000:
            logger.info(f"{task}.{segid} bbox_cnt {bbox_cnt} is medium type")
            type = "medium"
        elif bbox_cnt > 400:
            logger.info(f"{task}.{segid} bbox_cnt {bbox_cnt} is normal type")
            type = "normal"
        else:
            logger.info(f"{task}.{segid} bbox_cnt {bbox_cnt} is easy type")
            type = "easy"
        handled_infos.append(
            [segid, speed, daynight, task, car, deploy_subfix, frame_cnt, bbox_cnt]
        )

        dst_path = os.path.join(DST_POSTDEL_ROOT, type, car, deploy_subfix, segid)
        os.makedirs(
            os.path.join(DST_POSTDEL_ROOT, type, car, deploy_subfix),
            exist_ok=True,
            mode=0o777,
        )
        # os.system(f"mv {src_path} {dst_path}")

        if seg_clip_obs_path is not None:
            dst_info_dir = os.path.join(DST_INFO_ROOT, car, deploy_subfix, segid)
            dst_info = os.path.join(
                DST_INFO_ROOT, car, deploy_subfix, segid, f"{segid}_infos.json"
            )
            os.makedirs(dst_info_dir, exist_ok=True, mode=0o777)

            if os.path.exists(dst_path):
                logger.warning(f"{dst_path} exists")

            if not os.path.exists(seg_clip_obs_path):
                logger.info(f"{seg_clip_obs_path} not exists, create new clip_obstacle")
                # segment_path = seg_content['segPath']
                # use_count += 1
                # prepare_obstacle(segment_path, dst_path)
                # pool.apply_async(prepare_obstacle, (idx, total_count, segid, segment_path, dst_path), error_callback=multi_process_error_callback)
            else:
                logger.info(f"{seg_clip_obs_path} going...")
                src_info = os.path.join(seg_clip_obs_path, f"{segid}_infos.json")
                if not os.path.exists(src_info):
                    logger.warning(f"{src_info} not exists")
                    continue
                if not os.path.exists(dst_info):
                    shutil.copy(src_info, dst_info)

                use_count += 1
                # os.makedirs(os.path.join(DST_POSTDEL_ROOT, car_name, deploy_subfix), exist_ok=True, mode=0o777)
                # print(f">>> {segid} going...")
                # shutil.copytree(seg_clip_obs_path, dst_path)
                preanno_result_file = result_json
                dst_preanno_path = os.path.join(
                    DST_POSTDEL_ROOT, type, car, deploy_subfix, segid, "annotations"
                )
                if not os.path.exists(dst_preanno_path):
                    os.makedirs(dst_preanno_path, exist_ok=True, mode=0o777)
                dst_preanno_result_file = os.path.join(
                    DST_POSTDEL_ROOT,
                    type,
                    car,
                    deploy_subfix,
                    segid,
                    "annotations",
                    "result.json",
                )
                shutil.copy(preanno_result_file, dst_preanno_result_file)
                pool.apply_async(
                    handle_seg,
                    (idx, total_count, segid, seg_clip_obs_path, dst_path, dst_info),
                    error_callback=multi_process_error_callback,
                )

    output_xlsx = os.path.join(DST_POSTDEL_ROOT, "handled_infos.xlsx")
    df = pd.DataFrame(
        handled_infos,
        columns=[
            "segid",
            "speed",
            "daynight",
            "task",
            "car",
            "deploy_subfix",
            "frame_cnt",
            "bbox_cnt",
        ],
    )
    df.to_excel(f"{output_xlsx}.xlsx", index=False)
    pool.close()
    pool.join()
    logger.info(f"{use_count}/{total_count}")


def gen_refined_info():
    filters = set([])
    seg_infos = parse_infos_with_excels(xlsx_files)
    seg_infos = update_seg_infos(seg_infos)
    if os.path.exists(loss_json):
        with open(loss_json, "r") as fp:
            loss_dict = json.load(fp)

    segs = []
    for k, v in loss_dict.items():
        segid = os.path.basename(k)
        if segid in filters:
            logger.warning(f"{segid} is filtered")
            continue
        if segid not in seg_infos:
            logger.warning(f"{segid} not in seg_infos")
            continue

        (
            speed,
            daynight,
            task,
            car,
            deploy_subfix,
            frame_cnt,
            bbox_cnt,
            seg_obs_path,
            seg_road_name,
            seg_road_type,
            result_json
        ) = seg_infos[segid]
        loss = v
        segs.append(
            [
                segid,
                speed,
                daynight,
                task,
                car,
                deploy_subfix,
                frame_cnt,
                bbox_cnt,
                seg_road_name,
                seg_road_type,
                loss,
            ]
        )
    os.makedirs(DST_REFINED_ROOT, exist_ok=True, mode=0o777)
    dst_refined_xlsx = os.path.join(DST_REFINED_ROOT, "refined_infos")
    df = pd.DataFrame(
        segs,
        columns=[
            "segid",
            "speed",
            "daynight",
            "task",
            "car",
            "deploy_subfix",
            "frame_cnt",
            "bbox_cnt",
            "seg_road_name",
            "seg_road_type",
            "loss",
        ],
    )
    df.to_excel(f"{dst_refined_xlsx}.xlsx", index=False)


def prepare_refined_by_list():
    list_file = "/data_autodrive/users/brli/dev_raw_data/refined/refined_segs_list_0614.json"
    with open(list_file, "r") as fp:
        segs = json.load(fp)

    refined_info_xlsx = "/data_autodrive/auto/label_4d/refined/20240614/refined_infos.xlsx"
    seg_obs_paths = get_valid_obstacle_path(xlsx_files)
    seg_infos = {}
    df = pd.read_excel(refined_info_xlsx, skiprows=1)
    for idx, row in df.iterrows():
        segid, speed, daynight, task, car, deploy_subfix, f, bbox, _, _ , loss = row
        seg_infos[segid] = {
            "speed": speed,
            "daynight": daynight,
            "task": task,
            "car": car,
            "deploy_subfix": str(deploy_subfix),
            "segid": segid,
            "bbox": bbox,
            "frame_cnt": f,
            "loss": loss,
        }
    
    preanno_batches = os.listdir(PREANNO_ROOT)
    for batch in preanno_batches:
        annos_path = os.path.join(PREANNO_ROOT, batch)
        cars = os.listdir(annos_path)
        cars.sort()        
        for car in cars:
            car_annos_root = os.path.join(annos_path, car)
            subfixes = os.listdir(car_annos_root)
            subfixes.sort()
            for subfix in subfixes:
                subfix_path = os.path.join(car_annos_root, subfix)
                _segs = os.listdir(subfix_path)
                _segs.sort()
                for _seg in _segs:
                    seg_path = os.path.join(subfix_path, _seg)
                    if _seg in seg_infos:
                        seg_infos[_seg]['preanno_path'] = seg_path
    
    pool = Pool(processes=20)
    handled_infos = []
    total_count = len(segs)
    for idx, seg in enumerate(segs):
        if seg not in seg_infos:
            logger.warning(f"{seg} not in seg_infos")
            continue
        seg_info = seg_infos[seg]
        bbox_cnt = seg_info["bbox"]
        preanno_path = seg_info["preanno_path"]
        speed = seg_info["speed"]
        daynight = seg_info["daynight"]
        task = seg_info["task"]
        car = seg_info["car"]
        deploy_subfix = str(seg_info["deploy_subfix"])

        res = query_seg([seg])
        res_cnt = res[0]
        if res_cnt == 0:
            logger.warning(f"{seg} not in mongodb")
            continue
        seg_content = res[1][0]
        # seg_clip_obs_path = seg_content["pathMap"]["obstacle3dAnnoDataPath"]
        seg_clip_obs_path = seg_obs_paths.get(seg, None)
        if seg_clip_obs_path is None:
            logger.warning(f"{seg} not in seg_obs_paths")
            continue

        type = "hard"
        if bbox_cnt > 4000:
            logger.info(f"{task}.{seg} bbox_cnt {bbox_cnt} is hard type")
        elif bbox_cnt > 2000:
            logger.info(f"{task}.{seg} bbox_cnt {bbox_cnt} is medium type")
            type = "medium"
        elif bbox_cnt > 400:
            logger.info(f"{task}.{seg} bbox_cnt {bbox_cnt} is normal type")
            type = "normal"
        else:
            logger.info(f"{task}.{seg} bbox_cnt {bbox_cnt} is easy type")
            type = "easy"
        handled_infos.append(
            [seg, speed, daynight, task, car, deploy_subfix, bbox_cnt]
        )

        dst_path = os.path.join(DST_REFINED_ROOT, type, car, deploy_subfix, seg)
        os.makedirs(
            os.path.join(DST_REFINED_ROOT, type, car, deploy_subfix),
            exist_ok=True,
            mode=0o777,
        )

        preanno_result_file = os.path.join(
            preanno_path,
            "annotations",
            "result.json",
        )
        dst_preanno_path = os.path.join(
            DST_REFINED_ROOT, type, car, deploy_subfix, seg, "annotations"
        )
        if not os.path.exists(dst_preanno_path):
            os.makedirs(dst_preanno_path, exist_ok=True, mode=0o777)
        dst_preanno_result_file = os.path.join(
            DST_REFINED_ROOT,
            type,
            car,
            deploy_subfix,
            seg,
            "annotations",
            "result.json",
        )

        shutil.copy(preanno_result_file, dst_preanno_result_file)
        pool.apply_async(
            handle_seg,
            (idx, total_count, seg, seg_clip_obs_path, dst_path, ""),
            error_callback=multi_process_error_callback,
        )
    pool.close()
    pool.join()
    output_xlsx = os.path.join(DST_REFINED_ROOT, "handled_infos.xlsx")
    df = pd.DataFrame(
        handled_infos,
        columns=[
            "segid",
            "speed",
            "daynight",
            "task",
            "car",
            "deploy_subfix",
            "bbox_cnt",
        ],
    )
    df.to_excel(f"{output_xlsx}.xlsx", index=False)


if __name__ == "__main__":
    # gen_postdel()
    gen_refined_info()
    # prepare_refined_by_list()
    # gen_obs_infos()
