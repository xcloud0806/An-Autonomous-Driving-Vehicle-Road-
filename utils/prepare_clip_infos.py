import numpy as np
import json
import os
import cv2
from shapely.geometry import Polygon, MultiPoint

ANNO_INFO_JSON = "annotation.json"
ANNO_INFO_CALIB_KEY = "calib"
ANNO_INFO_INFO_KEY = "clip_info"
ANNO_INFO_LANE_KEY = "lane"
ANNO_INFO_OBSTACLE_KEY = "obstacle"
ANNO_INFO_OBSTACLE_STATIC_KEY = "obstacle_static"
ANNO_INFO_PAIR_KEY = "pair_list"
ANNO_INFO_RAW_PAIR_KEY = "raw_pair_list"
ANNO_INFO_POSE_KEY = "pose_list"
ANNO_INFO_RAW_POSE_KEY = "raw_pose_list"

def get_world_to_img(img_to_world):
    return np.linalg.inv(img_to_world)


def get_region(world_to_img, resolution, dx, dy):
    world_to_img[1, 3] = world_to_img[1, 3] - int(dy / resolution[1])
    world_to_img[1] = -world_to_img[1]
    xmin = -world_to_img[0, 3] * resolution[0]
    ymin = -world_to_img[1, 3] * resolution[1]
    # zmin = -world_to_img[2,3]*resolution[2]
    xmax = xmin + dx
    ymax = ymin + dy
    return xmin, xmax, ymin, ymax


def cal_lane_region(lane_anno_data_path):
    lane_version = "V1"
    transform_json = open(os.path.join(lane_anno_data_path, "transform_matrix.json"))
    transform_matrix = json.load(transform_json)
    if "version" in transform_matrix:
        lane_version = transform_matrix["version"]
    for file in os.listdir(lane_anno_data_path):
        if file.endswith(".npy"):
            npy_name = file
    npy_path = os.path.join(lane_anno_data_path, npy_name)

    # resolution = (0.05, 0.1, 0.5)
    img_to_world = transform_matrix["img_to_world"]
    _region = np.abs(np.array(img_to_world)[:3, :3])
    resolution = np.diagonal(_region).tolist()
    world_to_img = get_world_to_img(img_to_world)
    ret_world_to_img = world_to_img.tolist()

    height = np.load(npy_path)
    zmin = np.amin(height)
    zmax = np.amax(height)
    if lane_version.lower() == "v2":
        dx = height.shape[0] * resolution[0] * 2
        dy = height.shape[1] * resolution[1] * 2
    else:
        dx = height.shape[0] * resolution[0]
        dy = height.shape[1] * resolution[1]
    xmin, xmax, ymin, ymax = get_region(world_to_img, resolution, dx, dy)
    region_area = [xmin, xmax, ymin, ymax, zmin, zmax]
    return region_area, ret_world_to_img, lane_version


gnss_submit_keys = [
    "utc_time",
    "longitude",
    "latitude",
    "altitude",
    "speed",
    "pitch",
    "roll",
    "yaw",
    "gyrox",
    "gyroy",
    "gyroz",
]


def gen_pose_list(meta: dict, gnss: dict, pair_lst: list, pair_key='pair_list'):
    """
    从meta中获取pose,gnss信息,用于生成pose_list
    :param meta: meta json
    :param gnss: gnss json
    :param pair_lst: pair_list
    TODO:当位姿模块可以获取同步帧的位姿时，该部分需要更新
    """
    pair_ts = []
    for pair in pair_lst:
        pair_ts.append(int(pair["lidar"]))
    pose_list = []
    frames = meta["frames"]
    cameras = meta["cameras"]
    for f in frames:
        f_pose = {}
        lidar = f["lidar"]
        lidar_ts = lidar["timestamp"]
        if lidar_ts not in pair_ts:
            continue
        if pair_key == 'pair_list': # 只有pair list获取相机的pose 
            f_imgs = f["images"]
            for cam in cameras:
                if cam not in f_imgs:
                    continue
                f_pose[cam] = f_imgs[cam]["pose"]

        gnss_ts = str(f["gnss"])
        gnss_info = gnss[gnss_ts]
        gnss_submit = {}
        for k in gnss_submit_keys:
            if k == "longitude":
                long = float(gnss_info[k])
                # if gnss pt in china
                if long < 73.55 and long > 135.05:
                    continue
                gnss_submit[k] = str(long)
            elif k == "latitude":
                lat = float(gnss_info[k])
                # if gnss pt in china
                if lat < 3.86 and lat > 53.55:
                    continue
                gnss_submit[k] = str(lat)
            else:
                gnss_submit[k] = gnss_info[k]

        pose = lidar["pose"]
        f_pose.update({"lidar": pose, "gnss": gnss_submit})
        pose_list.append(f_pose)
    return pose_list


def gen_pair_list(
    meta: dict, enable_cams: list, enable_bpearls: list, pair_key="frames"
):
    if pair_key != "frames" and pair_key != "raws":
        raise ValueError("pair_key should be frames")

    frames = meta[pair_key]
    pair_list = []
    for f in frames:
        pair = {}
        lidar = str(f["lidar"]["timestamp"])
        pair["lidar"] = lidar
        images = f["images"]
        cam_keys = list(images.keys())
        for cam in enable_cams:
            if cam not in cam_keys:
                continue
            img = str(images[cam]["timestamp"])
            pair[cam] = img

        if "bpearls" in f and len(enable_bpearls) > 0:
            bpearls = f["bpearls"]
            bpearl_keys = list(bpearls.keys())
            for b in enable_bpearls:
                if b not in bpearl_keys:
                    continue
                pcd = str(bpearls[b]["timestamp"])
                pair[b] = pcd
        if "innos" in f and len(enable_bpearls) > 0:
            innos = f["innos"]
            inno_keys = list(innos.keys())
            for i in enable_bpearls:
                if i not in inno_keys:
                    continue
                pcd = str(innos[i]["timestamp"])
                pair[i] = pcd

        pair_list.append(pair)
    return pair_list


def get_road_name(meta: dict, gnss_json: str, test_roads_gnss, odometry_mode=1):
    curr_path = os.path.abspath(__file__)
    curr_dir = os.path.dirname(curr_path)
    test_road_gnss_json = os.path.join(curr_dir, test_roads_gnss)
    test_gnss_info = json.load(open(test_road_gnss_json))
    road_list = list(test_gnss_info.keys())
    if odometry_mode == 9:
        return "train", None, road_list
    clip_gnss = []
    with open(gnss_json, "r") as fp:
        gnss = json.load(fp)
        frames = meta["frames"]
        for f in frames:
            t = str(f["gnss"])
            lat = float(gnss[t]["latitude"])
            lon = float(gnss[t]["longitude"])
            clip_gnss.append([lat, lon])
    clip_gnss = np.array(clip_gnss)

    for road_name in test_gnss_info:
        road_gnss_info = test_gnss_info[road_name]
        gnss_to_img = np.array(road_gnss_info["gnss_to_img"])
        region = np.array(road_gnss_info["region"])
        image_size = np.array(road_gnss_info["image_size"])
        contours = np.array(road_gnss_info["contours"])

        img = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        cv2.fillPoly(img, [np.array(contours)], (255))  # type: ignore

        flag = (
            (clip_gnss[:, 0] > region[0, 0])
            & (clip_gnss[:, 0] < region[1, 0])
            & (clip_gnss[:, 1] > region[0, 1])
            & (clip_gnss[:, 1] < region[1, 1])
        )
        clip_gnss_in = clip_gnss[flag]
        if len(clip_gnss_in) > 0:
            clip_img = (
                np.matmul(gnss_to_img[:2, :2], clip_gnss_in.T) + gnss_to_img[:2, [2]]
            ).T
            # 与测试集轨迹有重叠, 重叠率必须大于90%
            if img[
                clip_img[:, 1].astype(np.uint32), clip_img[:, 0].astype(np.uint32)
            ].sum() > 0.9 * len(clip_gnss):
                return "test", road_name, road_list
    return "train", None, road_list


def get_city_test_region(meta: dict, gnss_json: str, test_city_region, odometry_mode=1):
    curr_path = os.path.abspath(__file__)
    curr_dir = os.path.dirname(curr_path)
    test_city_region_json = os.path.join(curr_dir, test_city_region)
    test_region_info = json.load(open(test_city_region_json))
    city_list = list(test_region_info.keys())
    if odometry_mode == 9:
        return "train", None, city_list

    regions = {}
    for city, info in test_region_info.items():
        region = np.array(info["region"])
        poly = Polygon(region)
        regions[city] = poly

    clip_gnss = []
    with open(gnss_json, "r") as fp:
        gnss = json.load(fp)
        frames = meta["frames"]
        for i, f in enumerate(frames):
            if i % 100 == 0:
                # print(f"{i}/{len(frames)}")
                t = str(f["gnss"])
                lat = float(gnss[t]["latitude"])
                lon = float(gnss[t]["longitude"])
                clip_gnss.append([lon, lat])
    total_gnss_pts = len(clip_gnss)
    # 尝试intersection接口
    clip_gnss = np.array(clip_gnss)
    for city, poly in regions.items():
        inter_pts = poly.intersection(MultiPoint(clip_gnss))
        if inter_pts is not None and inter_pts.geom_type == "MultiPoint":
            inter_pts = inter_pts.geoms
            pt_num = len(inter_pts)
        else:
            pt_num = 1
        if pt_num > 0.5 * total_gnss_pts:
            return "test", city, city_list
    return "train", None, city_list


def gen_datasets(
    meta: dict,
    gnss_json: str,
    test_roads_gnss="hefei_wuhu_test_roads.json",
    test_city_region="city_test_regions.json",
    odometry_mode=1,
):
    train_dataset, road_name, road_list = get_road_name(
        meta, gnss_json, test_roads_gnss, odometry_mode
    )
    if train_dataset == "test":
        train_dataset, city, city_list = get_city_test_region(
            meta, gnss_json, test_city_region, odometry_mode
        )
        road_list.extend(city_list)
        return "test", road_name, road_list
    else:
        train_dataset, city, city_list = get_city_test_region(
            meta, gnss_json, test_city_region, odometry_mode
        )
        road_list.extend(city_list)
        return train_dataset, city, road_list


def prepare_infos(
    meta: dict,
    enable_cams: list,
    enable_bpearls: list,
    seg_root_path: str,
    test_gnss_info="hefei_wuhu_test_roads.json",
    test_city_info="city_test_regions.json",
):
    # 从meta中生成seg 信息提交到研发网
    clip_info = {}
    calibs = meta["calibration"]
    clip_info[ANNO_INFO_CALIB_KEY] = calibs
    clip_info["data_system"] = meta["data_system"]
    pair_list = gen_pair_list(meta, enable_cams, enable_bpearls)
    raw_pair_list = gen_pair_list(meta, enable_cams, enable_bpearls, "raws")
    clip_info[ANNO_INFO_PAIR_KEY] = pair_list
    clip_info[ANNO_INFO_RAW_PAIR_KEY] = raw_pair_list

    gnss_json = os.path.join(seg_root_path, "gnss.json")
    fp = open(gnss_json, "r")
    gnss_data = json.load(fp)
    fp.close()
    clip_info[ANNO_INFO_POSE_KEY] = gen_pose_list(meta, gnss_data, pair_list)
    clip_info[ANNO_INFO_RAW_POSE_KEY] = gen_pose_list(meta, gnss_data, raw_pair_list, "raw_pair_list")

    seg_frame_path = meta["frames_path"]

    reconstruct_path = os.path.join(seg_root_path, "multi_reconstruct")
    if not os.path.exists(reconstruct_path):
        reconstruct_path = os.path.join(seg_root_path, "reconstruct")

    clip_id = meta["seg_uid"]
    clip_info["segment_path"] = seg_frame_path
    if os.path.exists(reconstruct_path) and os.path.exists(
        os.path.join(reconstruct_path, "transform_matrix.json")
    ):
        npy_name = ""
        for f in os.listdir(reconstruct_path):
            if f.endswith(".npy"):
                npy_name = f
        if npy_name != "":
            region, world_to_img, lane_ver = cal_lane_region(reconstruct_path)
            clip_info["world_to_img"] = world_to_img
            clip_info["region"] = region
            clip_info["lane_version"] = lane_ver
    clip_info["seg_uid"] = clip_id
    gnss_json = os.path.join(seg_root_path, "gnss.json")
    # (
    #     clip_info["datasets"],
    #     clip_info["road_name"],
    #     clip_info["road_list"],
    # ) = get_road_name(meta, gnss_json, test_gnss_info)
    (
        clip_info["datasets"],
        clip_info["road_name"],
        clip_info["road_list"],
    ) = gen_datasets(meta, gnss_json, test_gnss_info, test_city_info)
    return clip_info


def prepare_coll_seg_infos(
    lane_anno_data_path,
    meta: dict,
    enable_cams: list,
    enable_bpearls: list,
    seg_root_path: str,
    test_gnss_info="hefei_wuhu_test_roads.json",
    test_city_info="city_test_regions.json",
):
    seg_frame_path = meta["frames_path"]
    clip_id = meta["seg_uid"]
    region, world_to_img, lane_ver = cal_lane_region(lane_anno_data_path)

    clip_info = {}

    calibs = meta["calibration"]
    clip_info[ANNO_INFO_CALIB_KEY] = calibs
    clip_info["data_system"] = meta["data_system"]
    pair_list = gen_pair_list(meta, enable_cams, enable_bpearls)
    raw_pair_list = gen_pair_list(meta, enable_cams, enable_bpearls, "raws")
    clip_info[ANNO_INFO_PAIR_KEY] = pair_list
    clip_info[ANNO_INFO_RAW_PAIR_KEY] = raw_pair_list

    gnss_json = os.path.join(seg_root_path, "gnss.json")
    fp = open(gnss_json, "r")
    gnss_data = json.load(fp)
    fp.close()
    clip_info[ANNO_INFO_POSE_KEY] = gen_pose_list(meta, gnss_data, pair_list)

    clip_info["segment_path"] = seg_frame_path
    clip_info["world_to_img"] = world_to_img
    clip_info["region"] = region
    clip_info["seg_uid"] = clip_id
    clip_info["lane_version"] = lane_ver
    clip_info["road_list"] = ["fangxingdadao", "jichanggaosu", "yihuanlu"]
    gnss_json = os.path.join(seg_root_path, "gnss.json")
    (
        clip_info["datasets"],
        clip_info["road_name"],
        clip_info["road_list"],
    ) = gen_datasets(meta, gnss_json, test_gnss_info, test_city_info)
    return clip_info
