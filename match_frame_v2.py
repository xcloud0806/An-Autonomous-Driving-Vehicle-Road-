import sys, os
import json
import pandas
import traceback as tb

import math
import copy
import numpy as np
from utils import euler_to_rmat, rvec_to_rmat, CarMeta
from utils import PinholeCamera, FisheyeCamera
from .match_frame import correct_images_ts
from copy import deepcopy
from utils import fill_clip_match

Lidar_BPLidar_diff = {
    "e0y": 25,
    "s811": 5
}

class TransformKey:
    def __init__(self, src_key, tgt_key):
        self.src_key = src_key
        self.tgt_key = tgt_key

    def __hash__(self):
        return hash(self.src_key) + hash(self.tgt_key)

    def __eq__(self, other):
        return (self.src_key == other.src_key and self.tgt_key == other.tgt_key) or \
            (self.src_key == other.tgt_key and self.tgt_key == other.src_key)


class Transform:
    def __init__(self, transform_key, transform):
        if transform_key.src_key > transform_key.tgt_key:
            self.src_key = transform_key.src_key
            self.tgt_key = transform_key.tgt_key
            self.transform = transform
        else:
            self.src_key = transform_key.tgt_key
            self.tgt_key = transform_key.src_key
            self.transform = np.linalg.inv(transform)

    def get_transform_key(self):
        return TransformKey(self.src_key, self.tgt_key)
        
    def get_transform(self, transform_key):
        if transform_key.src_key == self.src_key and transform_key.tgt_key == self.tgt_key:
            return self.transform.copy()
        elif transform_key.src_key == self.tgt_key and transform_key.tgt_key == self.src_key:
            return np.linalg.inv(self.transform)
        else:
            raise KeyError()


def load_pose_from_json(pose_data):
    if "rmat" in pose_data:
        rmat = np.array(pose_data["rmat"], dtype=np.float64)
    elif "rvec" in pose_data:
        rvec = np.array(pose_data["rvec"], dtype=np.float64)
        rmat = rvec_to_rmat(rvec)
    elif "euler" in pose_data:
        order = pose_data["euler"]["order"]
        euler = np.array(pose_data["euler"]["angle_vec"], dtype=np.float64).reshape(-1, 1)
        if pose_data["euler"]["angle_type"] == "degree":
            euler = euler * math.pi / 180
        rmat = euler_to_rmat(euler, order)
    else:
        raise KeyError("unknown rotation key")
    
    tvec = np.array(pose_data["tvec"], dtype=np.float64)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rmat
    transform[:3, 3:] = tvec
    return transform


class TransformManager:
    def __init__(self):
        self.nodes = list()
        self.transforms = dict()

    def load_transform_from_json(self, data):
        for transform_data in data:
            source_name = transform_data["source"]
            target_name = transform_data["target"]

            if source_name not in self.nodes:
                self.nodes.append(source_name)
            
            if target_name not in self.nodes:
                self.nodes.append(target_name)
        
        for transform_data in data:
            source_name = transform_data["source"]
            target_name = transform_data["target"]

            transform_pose = load_pose_from_json(transform_data)
            transform = Transform(TransformKey(source_name, target_name), transform_pose)
            self.transforms[transform.get_transform_key()] = transform

    def add_transform(self, source, target, transform_pose):
        transform = Transform(TransformKey(source, target), transform_pose)
        assert transform.get_transform_key() not in self.transforms
        if source not in self.nodes:
            self.nodes.append(source)
        
        if target not in self.nodes:
            self.nodes.append(target)
        self.transforms[transform.get_transform_key()] = transform
    
    def get_nodes(self):
        return copy.deepcopy(self.nodes)

    def get_transform(self, source_key, target_key):
        if source_key == target_key:
            return np.eye(4, dtype=np.float64)
        
        if source_key not in self.nodes or target_key not in self.nodes:
            return None
        
        direct_transform_key = TransformKey(source_key, target_key)
        if direct_transform_key in self.transforms:
            return self.transforms[direct_transform_key].get_transform(direct_transform_key)
        else:
            reached_nodes = [source_key]
            leaf_nodes = {source_key: np.eye(4, dtype=np.float64)}
            while len(leaf_nodes) > 0:
                new_leaf_nodes = dict()
                for leaf_name, source_to_leaf_transform in leaf_nodes.items():
                    for node in self.nodes:
                        if node not in reached_nodes:
                            cur_transform_key = TransformKey(leaf_name, node)
                            if cur_transform_key in self.transforms:
                                leaf_node_to_cur_node_transform = self.transforms[cur_transform_key].get_transform(cur_transform_key)
                                new_source_to_leaf_transform = np.matmul(leaf_node_to_cur_node_transform, source_to_leaf_transform)

                                if node == target_key:
                                    return new_source_to_leaf_transform
                                else:
                                    new_leaf_nodes[node] = new_source_to_leaf_transform
                                    reached_nodes.append(node)
                leaf_nodes = new_leaf_nodes
        return None

def load_extrinsic(extrinsic_path):
    extrinsic_info = json.load(open(extrinsic_path))
    rvec = np.array(extrinsic_info["rvec"])
    tvec = np.array(extrinsic_info["tvec"])
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rvec_to_rmat(rvec)
    transform[:3, 3:] = tvec
    return transform

def load_camera(intrinsic_param):
    sensor_type = intrinsic_param["sensor_type"]
    if sensor_type != "camera":
        return None
    cam_model = intrinsic_param["cam_model"]
    if cam_model == "fisheye":
        camera = FisheyeCamera()
        camera.load_from_params(intrinsic_param)
    elif cam_model == "pinhole":
        camera = PinholeCamera()
        camera.load_from_params(intrinsic_param)
    else:
        raise TypeError("Invalid cam model: %s" % cam_model)
    return camera

def compute_camera_match_timeoffset_by_tf(cameras_name, tf, lidar_scan_start_direction):
    lidar_ref_x_axis = lidar_scan_start_direction
    lidar_ref_x_axis = lidar_ref_x_axis / np.linalg.norm(lidar_ref_x_axis)
    lidar_ref_z_axis = np.array([0, 0, 1], dtype=np.float32)

    lidar_ref_y_axis = np.cross(lidar_ref_z_axis, lidar_ref_x_axis)
    lidar_ref_y_axis = lidar_ref_y_axis / np.linalg.norm(lidar_ref_y_axis)
    
    cameras_match_timeoffset = dict()
    for camera_name in cameras_name:
        lidar_to_camera = tf.get_transform("lidar", camera_name)
        camera_to_lidar = np.linalg.inv(lidar_to_camera)
        camera_z_axis = np.array([0, 0, 1], dtype=np.float32)
        camera_z_axis_in_lidar = np.matmul(camera_to_lidar[:3, :3], camera_z_axis.reshape(-1, 1)).reshape(-1)
        theta = math.atan2(camera_z_axis_in_lidar.dot(lidar_ref_y_axis), camera_z_axis_in_lidar.dot(lidar_ref_x_axis))
        theta = theta + math.pi * 2 if theta < 0 else theta
        theta = math.pi * 2 - theta

        match_timeoffset = theta * 0.1 / (math.pi * 2)
        cameras_match_timeoffset[camera_name] = match_timeoffset
    return cameras_match_timeoffset

def load_calib_info(calib_folder):
    meta_info = json.load(open(os.path.join(calib_folder, "car_meta.json")))
    cameras_name = meta_info["camera_names"]
    # 读取相机内参
    cameras = dict()
    for camera_name in cameras_name:
        intrinsic_path = os.path.join(calib_folder, camera_name, "intrinsics.json")
        if not os.path.exists(intrinsic_path):
            continue
        cameras[camera_name] = load_camera(json.load(open(intrinsic_path)))

    # 读取传感器外参
    tf = TransformManager()
    tf_data = list()
    for camera_name in cameras_name:
        extrinsic_path = os.path.join(calib_folder, camera_name, "extrinsics.json")
        extrinsic = json.load(open(extrinsic_path))
        extrinsic["source"] = "lidar"
        extrinsic["target"] = camera_name
        tf_data.append(extrinsic)
    
    lidar_to_ego_extrisic = json.load(open(os.path.join(calib_folder, "lidar_to_ego.json")))
    lidar_to_ego_extrisic["source"] = "lidar"
    lidar_to_ego_extrisic["target"] = "ego"
    tf_data.append(lidar_to_ego_extrisic)

    lidar_to_ground_extrisic = json.load(open(os.path.join(calib_folder, "lidar_to_ground.json")))
    lidar_to_ground_extrisic["source"] = "lidar"
    lidar_to_ground_extrisic["target"] = "ground"
    tf_data.append(lidar_to_ground_extrisic)

    lidar_to_gnss_extrisic = json.load(open(os.path.join(calib_folder, "lidar_to_gnss.json")))
    lidar_to_gnss_extrisic["source"] = "lidar"
    lidar_to_gnss_extrisic["target"] = "gnss"
    tf_data.append(lidar_to_gnss_extrisic)

    tf.load_transform_from_json(tf_data)

    # 加载时间延迟
    cameras_timeoffset = {key: - val[1] * 0.05 for key, val in meta_info["offset_map"].items()}
    cameras_match_timeoffset = compute_camera_match_timeoffset_by_tf(cameras_name, tf, np.array([-1, 0, 0], dtype=np.float32))
    return tf, cameras, cameras_timeoffset, cameras_match_timeoffset

def find_nearest_item(items, timestamp):
    # 丢相机逻辑在函数执行前单独判断
    # if len(items) == 0:
    #     return None
    # if len(items) == 1:
    #     return None
    # 保留首尾丢帧的判断逻辑，首尾丢帧不应该加入到丢帧的统计数量中
    if timestamp < items[0][0]:
        return None
    if timestamp > items[-1][0]:
        return None
    
    left = 0
    right = len(items) - 1
    while True:
        if right - left == 1:
            return items[left] if timestamp - items[left][0] < items[right][0] - timestamp else items[right]
        else:
            middle = (right + left) // 2
            if items[middle][0] == timestamp:
                return items[middle]
            elif items[middle][0] > timestamp:
                right = middle
            else:
                left = middle

def find_sync_pairs(lidar_files, cameras_files, cameras_match_timeoffset):
    sync_pairs = list()
    prev_lidar_stamp = 0
    num_of_lidar_frame_lost = 0
    num_of_image_frame_lost = 0
    camera_lost = set()
    
    for idx, (lidar_stamp, lidar_file) in enumerate(lidar_files):
        if prev_lidar_stamp != 0:
            if abs(lidar_stamp - prev_lidar_stamp - 0.1) > 0.002:
                num_of_lidar_frame_lost += int(
                    (lidar_stamp - prev_lidar_stamp) * 10
                )
                continue
            else:
                prev_lidar_stamp = lidar_stamp
                
        sync_cameras_file = dict()
        
        for camera_name, camera_files in cameras_files.items():
            target_stamp = lidar_stamp + cameras_match_timeoffset[camera_name]
            
            if len(camera_files) == 1 or len(camera_files) == 0: # 判断为丢相机
                camera_lost.add(camera_name)
                num_of_image_frame_lost += 1
                sync_cameras_file[camera_name] = [0, None]
                continue
            nearest_camera_file = find_nearest_item(camera_files, target_stamp)
            if nearest_camera_file is None:  # 此时是首尾丢帧
                # num_of_image_frame_lost += 1  # 不记入统计
                sync_cameras_file[camera_name] = [0, None]  # 补零保持队列长度一致
                continue  # continue 同样是保持对列长度一致
            image_stamp, image_file = nearest_camera_file
            if abs(image_stamp - target_stamp) > 0.025:  # 此时是中间过程丢帧，不能跳过
                num_of_image_frame_lost += 1
                sync_cameras_file[camera_name] = [0, None]
                continue
            sync_cameras_file[camera_name] = [image_stamp, image_file]
        
        sync_pairs.append(dict(
            pre_lidar=lidar_files[idx-1] if idx-1 >= 0 else None,
            lidar=(lidar_stamp, lidar_file),
            next_lidar=lidar_files[idx+1] if idx+1 < len(lidar_files) else None,
            cameras=sync_cameras_file
        ))
    return sync_pairs, num_of_image_frame_lost, camera_lost, num_of_lidar_frame_lost

def match(clip_path, calib_path):
    tf, cameras, cameras_timeoffset, cameras_match_timeoffset = load_calib_info(calib_path)
    cameras_name = list(cameras.keys())

    def scan_files(folder, ext, time_offset=0, time_scale=1):
        files = list()
        for name in os.listdir(folder):
            if name.endswith(ext):
                timestamp = float(name[:name.rindex(".")]) * time_scale + time_offset
                path = os.path.join(folder, name)
                files.append((timestamp, path))
        files = sorted(files, key=lambda item: item[0])
        return files
    
    cameras_files = {
        camera_name: scan_files(
            os.path.join(clip_path, camera_name),
            (".jpg", ".jpeg", ".png"), 0, 1e-3
        ) \
            for camera_name in cameras_name
    }
    lidar_files = scan_files(os.path.join(clip_path, "lidar"), (".pcd",), 0, 1e-3)

    result = find_sync_pairs(lidar_files, cameras_files, cameras_match_timeoffset)
    return result

def match_raw(clip_path, cams:list, calib_path, lidar_bplidar_thresh):
    # 统计数据初始化，数据看板相关
    clip_name = os.path.basename(clip_path)

        
    match_type = "raw"
    num_of_lidar_frame = 0
    num_of_lidar_frame_lost = 0
    num_of_image_frame_lost = 0
    num_of_inno_lidars_frames_lost = None
    num_of_bpearl_lidars_frame_lost = None
    camera_lost = []
    inno_lidar_lost = None
    bpearl_lidar_lost = None  
    
    car_meta = CarMeta()
    car_meta_file = os.path.join(calib_path, "car_meta.json")
    with open(car_meta_file, 'r') as fp:
        car_meta_dict = json.load(fp)    
        car_meta.from_json_iflytek(car_meta_dict)
    
    lidar_name = car_meta.lidar_name
    offset_detail = car_meta.offset_detail

    match_res = {}
    cams_list = cams

    pcd_dir = os.path.join(clip_path, lidar_name)
    pcds = os.listdir(pcd_dir)
    if len(pcds) < 10:
        print("{} lidar is empty!!!".format(clip_path))
        return
    pcds.sort()
    pcd_ts = []
    pcd_ts_str = []
    for pcd in pcds:
        ts = int(pcd[:-4])
        pcd_ts.append(ts)
        pcd_ts_str.append(str(pcd[:-4]))
    num_of_lidar_frame = len(pcd_ts)  
    match_res['lidar'] = pcd_ts_str

    # lidar_ts_lst = []
    # lidar_int_ts_lst = []
    # for f in sync_pairs:
    #     lidar_ts_f = f['lidar'][0]
    #     lidar_ts = str(lidar_ts_f * 1000)
    #     lidar_ts_lst.append(lidar_ts)
    #     lidar_int_ts_lst.append(int(lidar_ts_f * 1000))
    # match_res['lidar'] = lidar_ts_lst
    # num_of_lidar_frame = len(lidar_ts_lst)

    for cam in cams_list:
        camera_name = cam
        # base, n = offset_detail[cam]
        cam_res = []
        image_target_path = os.path.join(clip_path, camera_name) 
        if cam not in cams_list or not os.path.exists(image_target_path):
            camera_lost.append(cam)
            for idx, pts in enumerate(pcd_ts):
                cam_res.append(0) 
            match_res[cam] = cam_res
            continue
        
        imgs = os.listdir(image_target_path)
        imgs.sort()
        img_ts = []
        for img in imgs:
            ts = int(os.path.splitext(img)[0])
            img_ts.append(ts)
        img_ts_arr = np.array(img_ts)
        for pts in pcd_ts:
            _res = (img_ts_arr - pts - 25)
            _res = np.abs(_res)
            # _res[_res < 0] = 10000
            match_idx = _res.argmin()
            # if _res[match_idx] > 75:
            if _res[match_idx] > 25:
                cam_res.append(0)
                num_of_image_frame_lost += 1
            else:
                cam_res.append(str(img_ts_arr[match_idx]))
            
            # _res[_res < 0] = 10000
            # match_idx = _res.argmin()
            # if _res[match_idx] > 75:
            #     cam_res.append(0)
            #     num_of_image_frame_lost += 1
            # else:
            #     cam_res.append(str(img_ts_arr[match_idx]))
            
        match_res[cam] = cam_res

    if len(car_meta.bpearl_lidars) > 0:
        num_of_bpearl_lidars_frame_lost = 0
        bpearl_lidar_lost = []
        for bpearl in car_meta.bpearl_lidars:
            bpearl_res = []
            bpearl_target_path = os.path.join(clip_path, bpearl)  
            if not os.path.exists(bpearl_target_path):
                print(f"warning car_meta.json enabled bpearl_lidar, but {bpearl_target_path} does not exist!")
                bpearl_lidar_lost.append(bpearl)
                for idx, pts in enumerate(pcd_ts):
                    bpearl_res.append(0) 
                match_res[bpearl] = bpearl_res
                continue
            bpcds = os.listdir(bpearl_target_path)
            bpcds.sort()
            bpcd_ts = []
            for p in bpcds:
                ts = int(os.path.splitext(p)[0])
                bpcd_ts.append(ts)
            bpcd_ts_arr = np.array(bpcd_ts)
            for pts in pcd_ts:
                match_idx = abs(bpcd_ts_arr - pts).argmin()
                if abs(bpcd_ts_arr[match_idx]- pts) > lidar_bplidar_thresh:
                    bpearl_res.append(0)
                    num_of_bpearl_lidars_frame_lost += 1
                else:
                    bpearl_res.append(str(bpcd_ts_arr[match_idx]))
            match_res[bpearl] = bpearl_res

    # record = {}
    # pcd_ts_str = []
    # for _ts in pcd_ts:
    #     pcd_ts_str.append("{:010f}".format(_ts))
    # record[lidar_name] = pcd_ts_str
    # for idx, cam in enumerate(cams_list):
    #     camera_name = cam
    #     record[camera_name] = match_res[cam]

    dframe = pandas.DataFrame(match_res)
    output = os.path.join(clip_path, "raw.csv")
    dframe.to_csv(output, index=False, encoding='utf8')
    
    fill_clip_match(
        clip_name = clip_name,
        match_type = match_type,
        num_of_lidar_frame = num_of_lidar_frame,
        num_of_lidar_frame_lost = num_of_lidar_frame_lost,
        num_of_image_frame_lost = num_of_image_frame_lost,
        num_of_inno_lidars_frames_lost = num_of_inno_lidars_frames_lost,
        num_of_bpearl_lidars_frame_lost = num_of_bpearl_lidars_frame_lost,
        camera_lost = camera_lost,
        inno_lidar_lost = inno_lidar_lost,
        bpearl_lidar_lost = bpearl_lidar_lost,
    )

def match_frame_fdc(clip_path, cams:list, calib_path, seg_mode): 
    # 统计数据初始化，数据看板相关
    clip_name = os.path.basename(clip_path)
    match_type = "match"
    num_of_lidar_frame = 0
    num_of_lidar_frame_lost = 0
    num_of_image_frame_lost = 0
    num_of_inno_lidars_frames_lost = None
    num_of_bpearl_lidars_frame_lost = None
    camera_lost = []
    inno_lidar_lost = None
    bpearl_lidar_lost = None   
    
    result = match(clip_path, calib_path)  
    sync_pairs, num_of_image_frame_lost, camera_lost_temp, num_of_lidar_frame_lost = result
    
    match_res = {}
    lidar_ts_lst = []   # str类型
    lidar_int_ts_lst = []   # int类型
    for f in sync_pairs:
        lidar_ts_f = f['lidar'][0]
        lidar_ts = str(lidar_ts_f * 1000)
        lidar_ts_lst.append(lidar_ts)
        lidar_int_ts_lst.append(int(lidar_ts_f * 1000))
    match_res['lidar'] = lidar_ts_lst   # str
    num_of_lidar_frame = len(lidar_ts_lst)
    
    for cam in cams:
        cam_ts_lst = []
        for f in sync_pairs:
            if cam not in f['cameras']:
                camera_lost_temp.add(cam)
                break
            cam_ts_f = f['cameras'][cam][0]
            cam_ts = str(cam_ts_f * 1000)
            cam_ts_lst.append(cam_ts)
        match_res[cam] = cam_ts_lst
    camera_lost = list(camera_lost_temp)    
    
    car_meta = CarMeta()
    car_meta_file = os.path.join(calib_path, "car_meta.json")
    with open(car_meta_file, 'r') as fp:
        car_meta_dict = json.load(fp)    
        car_meta.from_json_iflytek(car_meta_dict)

    # correct_images_ts(clip_path, clip_path, cams, car_meta)
    car_name = 'None'
    tag_info_path = os.path.join(clip_path, "tag_info.json")
    if os.path.exists(tag_info_path):
        with open(tag_info_path, 'r') as fp:
            tag_info = json.load(fp)
            car_name = tag_info["carNum"].lower().replace("-", "_")
    
    lidar_bplidar_thresh = 5
    for car_type in Lidar_BPLidar_diff:
        if car_type in car_name:
            lidar_bplidar_thresh = Lidar_BPLidar_diff[car_type]
            break

    if len(car_meta.bpearl_lidars) > 0:
        num_of_bpearl_lidars_frame_lost = 0
        bpearl_lidar_lost = []
        for bpearl in car_meta.bpearl_lidars:
            bpearl_res = []
            bpearl_target_path = os.path.join(clip_path, bpearl)  
            if not os.path.exists(bpearl_target_path):
                print(f"warning car_meta.json enabled bpearl_lidar, but {bpearl_target_path} does not exist!")
                bpearl_lidar_lost.append(bpearl)
                for idx, pts in enumerate(lidar_int_ts_lst):
                    bpearl_res.append(0) 
                match_res[bpearl] = bpearl_res
                continue
            bpcds = os.listdir(bpearl_target_path)
            bpcds.sort()
            bpcd_ts = []
            for p in bpcds:
                ts = int(os.path.splitext(p)[0])
                bpcd_ts.append(ts)
            bpcd_ts_arr = np.array(bpcd_ts)
            for pts in lidar_int_ts_lst:
                match_idx = abs(bpcd_ts_arr - pts).argmin()
                if abs(bpcd_ts_arr[match_idx]- pts) > lidar_bplidar_thresh:
                    bpearl_res.append(0)
                    num_of_bpearl_lidars_frame_lost += 1
                else:
                    bpearl_res.append(str(bpcd_ts_arr[match_idx]))
            match_res[bpearl] = bpearl_res

    if len(car_meta.inno_lidars) > 0 and seg_mode != "traffic_light":

        num_of_inno_lidars_frames_lost = 0
        inno_lidar_lost = []
        for inno in car_meta.inno_lidars:
            base = 25
            if car_meta.dc_type == "mdc":
                base = 5
            inno_res = []
            inno_target_path = os.path.join(clip_path, inno)  
            if not os.path.exists(inno_target_path):
                inno_lidar_lost.append(inno)
                for idx, pts in enumerate(lidar_int_ts_lst):
                    inno_res.append(0) 
                match_res[inno] = inno_res
                continue
            innos = os.listdir(inno_target_path)
            innos.sort()
            inno_ts = []
            for i in innos:
                ts = int(os.path.splitext(i)[0])
                inno_ts.append(ts)
            inno_ts_arr = np.array(inno_ts)
            for pts in lidar_int_ts_lst:
                match_idx = abs(inno_ts_arr - pts - base).argmin()
                if abs(inno_ts_arr[match_idx]- pts - base) > 5:
                    inno_res.append(0)
                    num_of_inno_lidars_frames_lost += 1
                else:
                    inno_res.append(str(inno_ts_arr[match_idx]))
            match_res[inno] = inno_res

    if len(car_meta.radars) > 0:
        radar_lost = []
        for radar in car_meta.radars:
            base = 25
            if car_meta.dc_type == "mdc":
                base = 5
            radar_res = []
            radar_target_path = os.path.join(clip_path, radar)
            if not os.path.exists(radar_target_path):
                radar_lost.append(radar)
                for idx, pts in enumerate(lidar_int_ts_lst):
                    radar_res.append(0) 
                match_res[radar] = radar_res
                continue
            radars = os.listdir(radar_target_path)
            radars.sort()
            radar_ts = []
            for i in radars:
                ts = int(os.path.splitext(i)[0])
                radar_ts.append(ts)
            radar_ts_arr = np.array(radar_ts)
            for pts in lidar_int_ts_lst:
                match_idx = abs(radar_ts_arr - pts - base).argmin()
                if abs(radar_ts_arr[match_idx]- pts - base) > 50:
                    radar_res.append(0)
                else:
                    radar_res.append(str(radar_ts_arr[match_idx]))
            match_res[radar] = radar_res

    dframe = pandas.DataFrame(match_res)
    output = os.path.join(clip_path, "matches.csv")
    dframe.to_csv(output, index=False, encoding='utf8')
    
    fill_clip_match(
        clip_name = clip_name,
        match_type = match_type,
        num_of_lidar_frame = num_of_lidar_frame,
        num_of_lidar_frame_lost = num_of_lidar_frame_lost,
        num_of_image_frame_lost = num_of_image_frame_lost,
        num_of_inno_lidars_frames_lost = num_of_inno_lidars_frames_lost,
        num_of_bpearl_lidars_frame_lost = num_of_bpearl_lidars_frame_lost,
        camera_lost = camera_lost,
        inno_lidar_lost = inno_lidar_lost,
        bpearl_lidar_lost = bpearl_lidar_lost,
    )
    match_raw(clip_path, cams, calib_path, lidar_bplidar_thresh)
