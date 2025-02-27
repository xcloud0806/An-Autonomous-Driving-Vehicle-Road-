#  date 12.2, add gnss data in f2c & c2f
import json
import os
import sys
import argparse
import cv2
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
import warnings
import copy


def vector_to_matrix(rvec, tvec):
    rvec = np.array(rvec, dtype=np.float32)  
    tvec = np.array(tvec, dtype=np.float32)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32) 
    T[:3, :3] = rotation_matrix
    T[:3, 3] = tvec.flatten()
    return T

def matrix_to_vector(T):
    T = np.array(T, dtype=np.float32) 
    rotation_matrix = T[:3, :3]
    rvector, _ = cv2.Rodrigues(rotation_matrix)
    tvector = T[:3, 3]
    rvec = rvector.flatten().tolist()
    tvec = tvector.tolist()
    return rvec, tvec

def bpearl_name_c2f(name):
    if name == "side_lidar_front": return "bpearl_lidar_front"
    elif name == "side_lidar_left": return "bpearl_lidar_left"
    elif name == "side_lidar_rear": return "bpearl_lidar_rear"
    elif name == "side_lidar_right": return "bpearl_lidar_right"
    else:
        return ""

def camera_name_c2f(name):
    if name == "svc_front": return "ofilm_around_front_190_3M"
    elif name == "svc_left": return "ofilm_around_left_190_3M"
    elif name == "svc_rear": return "ofilm_around_rear_190_3M"
    elif name == "svc_right": return "ofilm_around_right_190_3M"
    elif name == "front_wide": return "ofilm_surround_front_120_8M"
    elif name == "front_narrow": return "ofilm_surround_front_30_8M"
    elif name == "front_left": return "ofilm_surround_front_left_100_2M"
    elif name == "front_right": return "ofilm_surround_front_right_100_2M"
    elif name == "rear_left": return "ofilm_surround_rear_left_100_2M"
    elif name == "rear_right": return "ofilm_surround_rear_right_100_2M"
    elif name == "rear_narrow": return "ofilm_surround_rear_100_2M"
    else:
        return ""

def save_extrinsic_param(file_path, rvec, tvec):
    extrinsics_data = {
        "rvec": [[val] for val in rvec],
        "tvec": [[val] for val in tvec],
    }
    with open(file_path, 'w')as f:
        json.dump(extrinsics_data, f, indent=4)

def save_intrinsic_param(file_path, cam_intrinc_param):
    mtx = cam_intrinc_param["camera_matrix"]
    dist = cam_intrinc_param["distortion_coeffcients"]
    camera_model = cam_intrinc_param["camera_model"]
    image_size = (cam_intrinc_param["camera_width"], cam_intrinc_param["camera_height"])
    if camera_model == "rad-tan": 
        cam_model = "pinhole"
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(np.array(mtx), np.array(dist), image_size, alpha=0.0)
        undistored_mtx = new_camera_matrix.tolist()
        while len(dist) < 14:
            dist.append(0.0)
        dist = [dist]
    elif camera_model == "poly5":
        cam_model = "fisheye"
        undistored_mtx = mtx
        dist = [[val] for val in dist]
    else:
        raise ValueError(f"camera model {camera_model} is invalid")
    
    intrinsic_data = {
        "mtx": mtx,
        "dist": dist,
        "undistored_mtx": undistored_mtx,
        "sensor_type": "camera",
        "cam_model": cam_model,
        "image_size": image_size
    }
    with open(file_path, 'w') as f:
        json.dump(intrinsic_data, f, indent=4)

 
def lidar_transform_c2f(rvec, tvec, rvec_main, tvec_main):
    T_main2ego = vector_to_matrix(rvec_main, tvec_main)
    T_ego2main = np.linalg.inv(T_main2ego)
    T_sensor2ego = vector_to_matrix(rvec, tvec)
    T_sensor2main = np.dot(T_ego2main, T_sensor2ego)
    r_sensor2main, t_sensor2main = matrix_to_vector(T_sensor2main)
    return r_sensor2main, t_sensor2main

def camera_transform_c2f(rvec, tvec, rvec_main, tvec_main):
    T_main2ego = vector_to_matrix(rvec_main, tvec_main)
    T_ego2main = np.linalg.inv(T_main2ego)
    T_sensor2ego = vector_to_matrix(rvec, tvec)
    T_sensor2main = np.dot(T_ego2main, T_sensor2ego)
    T_main2sensor = np.linalg.inv(T_sensor2main)
    r_main2sensor, t_main2sensor = matrix_to_vector(T_main2sensor)
    return r_main2sensor, t_main2sensor

def get_date_string():
    date_data = datetime.now()
    return date_data.strftime("%Y-%m-%d-%H-%M-%S")

def generate_calib_params_json(output_path):
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
    calib_params = {
        "camera":[],
        "lidar":[],
        "gnss":[],
        "imu":[],
        "vehicle_info":{}
    }
    with open(output_path, 'w') as f:
        json.dump(calib_params, f, indent=4)
    
def vehicle_info_c2f(veh_name, wheel_radius):

    vehicle_data ={
        "which_car":veh_name,
        "param":{
            "wheel_radius_rear_left": wheel_radius
        }
    }
    return vehicle_data

def gnss_module_f2c(rvec_lidar2gnss, tvec_lidar2gnss, rvec_main2ego, tvec_main2ego):
    T_lidar2ego = vector_to_matrix(rvec_main2ego, tvec_main2ego)
    T_lidar2gnss = vector_to_matrix(rvec_lidar2gnss, tvec_lidar2gnss)
    T_gnss2lidar = np.linalg.inv(T_lidar2gnss)
    T_gnss2ego = np.dot(T_lidar2ego, T_gnss2lidar)
    rvec_gnss, tvec_gnss = matrix_to_vector(T_gnss2ego)

    gnss_data = {
            "calibration_info": {
                "algo_version": "",
                "calib_counter": 0,
                "calibration_details": "convert by calib_conversion_tool",
                "date": get_date_string(),
                "error_code": "none",
                "error_with_base": [],
                "error_with_design": [],
                "format_version": "v1.0.0",
                "last_algo_version": "",
                "last_date": ""
            },
            "extrinsic_param": {
                "rotation": rvec_gnss,
                "translation": tvec_gnss
            },
            "hardware_info": {
                "fix_height_std": 0.097,
                "fix_pos_std": 0.035,
                "fix_vel_std": 0.025,
                "fix_yaw_std": 0.217,
                "float_height_std": 0.539,
                "float_pos_std": 0.428,
                "float_vel_std": 0.025,
                "float_yaw_std": 0.203
            }, # these numbers are hard code
        "name": "gnss"
    }

    return gnss_data


def lidar_module_f2c(rvec, tvec, name):
    # lidar_name = ""
    if name == "bpearl_lidar_front": lidar_name = "side_lidar_front"
    elif name == "bpearl_lidar_left": lidar_name = "side_lidar_left"
    elif name == "bpearl_lidar_rear": lidar_name = "side_lidar_rear"
    elif name == "bpearl_lidar_right": lidar_name = "side_lidar_right"
    elif name == "lidar_top": lidar_name = name
    elif name == "falcon": lidar_name = name
    else:
        warnings.warn(f"Not support current lidar mode: {name}")
        return None
    
    rvec = np.array(rvec).flatten()
    tvec = np.array(tvec).flatten()

    lidar_data = {
            "calibration_info": {
            "algo_version": "",
            "calib_counter": 0,
            "calibration_details": "convert by calib_conversion_tool",
            "date": get_date_string(),
            "error_code": "none",
            "error_with_base": [],
            "error_with_design": [],
            "format_version": "v1.0.0",
            "last_algo_version": "",
            "last_date": ""
            },
            "extrinsic_param": {
                "rotation": rvec.tolist(),
                "translation": tvec.tolist()
            },
            "hardware_info": {},
            "name": lidar_name
    }            
    return lidar_data

def extract_extrincs_f2c(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    rvec = data.get("rvec", [])
    tvec = data.get("tvec", [])
    return rvec, tvec

def extract_intrinsics_f2c(file_path):
    with open(file_path, 'r') as file:
        intrinsics_data = json.load(file)

    camera_matrix = intrinsics_data.get("mtx")

    cam_model = intrinsics_data.get("cam_model")
    if cam_model == "fisheye": camera_model = "poly5"
    elif cam_model == "pinhole": camera_model = "rad-tan"
    else:
        raise ValueError(f"{cam_model} is invalid")

    dist =intrinsics_data.get("dist")
    distortion = np.array(dist).flatten()
    if len(distortion)>8: 
        distortion = distortion[:8]
    distortion_coefficients = distortion.tolist()

    image_size = intrinsics_data.get("image_size")
    camera_width = image_size[0]
    camera_height = image_size[1]
    
    return camera_height, camera_matrix, camera_model, camera_width, distortion_coefficients

def camera_name_f2c(file_path):
    cam_fold_name = Path(file_path).name
    camera_name = ""
    if cam_fold_name.startswith("ofilm_around_front_190"):  camera_name = "svc_front"  
    elif cam_fold_name.startswith("ofilm_around_left_190"): camera_name = "svc_left" 
    elif cam_fold_name.startswith("ofilm_around_rear_190"): camera_name = "svc_rear"
    elif cam_fold_name.startswith("ofilm_around_right_190"): camera_name = "svc_right"
    elif cam_fold_name.startswith("ofilm_surround_front_120"): camera_name = "front_wide"
    elif cam_fold_name.startswith("ofilm_surround_front_30"): camera_name = "front_narrow"
    elif cam_fold_name.startswith("ofilm_surround_front_left_100"): camera_name = "front_left"
    elif cam_fold_name.startswith("ofilm_surround_front_right_100"): camera_name = "front_right"
    elif cam_fold_name.startswith("ofilm_surround_rear_left_100"): camera_name = "rear_left"
    elif cam_fold_name.startswith("ofilm_surround_rear_right_100"): camera_name = "rear_right"
    elif cam_fold_name.startswith("ofilm_surround_rear_100"): camera_name = "rear_narrow"
    return camera_name


def camera_module_f2c(file_path, T_lidar2ego, global_config_path):
    extrinsics_path = os.path.join(file_path, 'extrinsics.json')
    intrinsics_path = os.path.join(file_path, 'intrinsics.json')
    if not os.path.exists(extrinsics_path):
        raise ValueError("Error: fold %s is missing file extrinsics.json", file_path)
    if not os.path.exists(intrinsics_path):
        raise ValueError("Error: fold %s is missing file intrinsics.json", file_path)
    print('load camera data from: ', file_path)
    
    camera_name = camera_name_f2c(file_path)
    sn = read_camera_info(global_config_path, camera_name)

    # extrinsics
    rvec_camera_around, tvec_camera_around = extract_extrincs_f2c(extrinsics_path)
    T_camera_around = vector_to_matrix(rvec_camera_around, tvec_camera_around)
    T_camera_around2lidar = np.linalg.inv(T_camera_around)
    T_camera_around2ego = np.dot(T_lidar2ego, T_camera_around2lidar)
    rvec_camera_around2ego, tvec_camera_around2ego = matrix_to_vector(T_camera_around2ego)

    # intrinsics
    camera_height, camera_matrix, camera_model, camera_width, distortion_coefficients = extract_intrinsics_f2c(intrinsics_path)

    camera_data ={
            "calibration_info": {
                "algo_version": "",
                "calib_counter": 0,
                "calibration_details": "convert by calib_conversion_tool",
                "date": get_date_string(),
                "error_code": "",
                "error_with_base": [],
                "error_with_design": [],
                "format_version": "v1.0.0",
                "last_algo_version": "",
                "last_date": ""
            },
            "extrinsic_param": {
                "rotation": rvec_camera_around2ego,
                "translation": tvec_camera_around2ego
            },
            "hardware_info": {
                "hardware_version": "",
                "sn": sn
            },
            "intrinsic_param": {
                "camera_height": camera_height,
                "camera_matrix": camera_matrix,
                "camera_model": camera_model,
                "camera_width": camera_width,
                "distortion_coeffcients": distortion_coefficients
            },
            "name": camera_name
    }

    return camera_data

def vehicle_info_f2c(global_config_path):
    with open(global_config_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
        print('load global_config from: ', global_config_path)
   
    wheel_radius = yaml_data['param']['wheel_radius_rear_left']
    veh_name = yaml_data['which_car']
    print('veh_name: ', veh_name)

    vehicle_data = {
            "veh_name" : veh_name,
            "wheel_radius": wheel_radius
    }

    return vehicle_data, wheel_radius

def read_camera_info(global_config_path, camera_name):
    if camera_name == "front_wide": name_yaml = "front_120"
    elif camera_name == "front_narrow": name_yaml = "front_30"
    elif camera_name == "rear_narrow": name_yaml = "rear_100"
    elif camera_name == "svc_front": name_yaml = "surround_front"
    elif camera_name == "svc_left": name_yaml = "surround_left"
    elif camera_name == "svc_right": name_yaml = "surround_right"
    elif camera_name == "svc_rear": name_yaml = "surround_rear"
    elif camera_name == "rear_left" or camera_name == "rear_right" or camera_name == "front_left" or camera_name == "front_right":
        name_yaml = camera_name
    else:
        raise ValueError(f"Error: Invalid camera {camera_name}")
    
    with open(global_config_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)

    camera_data = yaml_data.get('camera', {})
    if camera_data:
        camera_details =camera_data.get("camera_details", [])
        for camera in camera_details:
            name = camera.get("name", '')
            if name.startswith(name_yaml):
                return camera.get('sn', None)
        
    return ""

def lidar_to_ground_c2f (rvec_main, tvec_main):
    T_lidar2main = vector_to_matrix(rvec_main, tvec_main) 

    theta = 90 * np.pi / 180
    rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])
    
    T_main2ground = np.eye(4)
    T_main2ground[:3, :3] = rz 
    T_main2ground[1, 3] -= T_lidar2main[0, 3]
    T_main2ground[0, 3] -= T_lidar2main[1, 3]

    T_lidar2ground = np.dot(T_main2ground, T_lidar2main)
    rvec_lidar2ground, tvec_lidar2ground = matrix_to_vector(T_lidar2ground)
    return rvec_lidar2ground, tvec_lidar2ground

def gnss_transform_c2f(rvec_gnss2ego, tvec_gnss2ego, rvec_main, tvec_main):
    T_gnss2ego = vector_to_matrix(rvec_gnss2ego, tvec_gnss2ego)
    T_ego2gnss = np.linalg.inv(T_gnss2ego)
    T_lidar2ego = vector_to_matrix(rvec_main, tvec_main)
    T_lidar2gnss = np.dot(T_ego2gnss, T_lidar2ego)
    r_lidar2gnss, t_lidar2gnss = matrix_to_vector(T_lidar2gnss)
    return r_lidar2gnss, t_lidar2gnss

# mode 2
def calib_file_to_vehicle_fold(camlib_params_path, output_path,with_car_name=True,dump_global_config=True):
    # input_path, output_path, mode
    if not os.path.exists(camlib_params_path):
        raise FileNotFoundError("Error: Input file cannot be found")
    
    if not camlib_params_path.endswith(".json"):
        raise ValueError("Input path is not a JSON file")
    with open(camlib_params_path, 'r') as f:
        calib_params = json.load(f)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # vehicle_dir
    dir_name = calib_params["vehicle_info"]["veh_name"].upper().replace("-", "_")
    if with_car_name:
        vehicle_path = os.path.join(output_path, dir_name)
    else:
        vehicle_path = output_path
    os.makedirs(vehicle_path,exist_ok=True)

    wheel_radius = calib_params["vehicle_info"]["wheel_radius"]
    
    if dump_global_config:
    # global_config
        global_config_path = os.path.join(vehicle_path, 'global_config.yaml')
        vehicle_data = vehicle_info_c2f(dir_name, wheel_radius)
        with open(global_config_path, 'w')as f:
            yaml.dump(vehicle_data, f)
    else:
        print("Skip dump global_config.yaml")
    
    # lidar
    # main_lidar_data
    lidars_path = os.path.join(vehicle_path, 'lidar')
    if not os.path.exists(lidars_path):
        os.mkdir(lidars_path)
    rvec_main, tvec_main = None, None
    bpearl_data = {}
    for lidar in calib_params.get("lidar", []):
        if lidar.get("name") == "lidar_top": 
            rvec_main = lidar["extrinsic_param"].get("rotation", [])
            tvec_main = lidar["extrinsic_param"].get("translation", [])
            break
    if rvec_main is None or tvec_main is None:
        warnings.warn(' Missing Lidar_top data! ')    

    # lidar_to_ego
    if len(rvec_main) == 0 or len(tvec_main) == 0:
        warnings.warn(f"extrinsic of lidar_top is empty.")  
    else:
        r_lidar2ego = copy.deepcopy(rvec_main)
        t_lidar2ego = copy.deepcopy(tvec_main)

        t_lidar2ego[2] = t_lidar2ego[2] - wheel_radius

        lidar_main_path = os.path.join(lidars_path, 'lidar_to_ego.json')
        save_extrinsic_param(lidar_main_path, r_lidar2ego, t_lidar2ego)
        print('load lidar_to_ego to', lidar_main_path)

    # lidar to ground
    rvec_l2g, tvec_l2g = lidar_to_ground_c2f(rvec_main, tvec_main)
    lidar2Ground_path = os.path.join(lidars_path, 'lidar_to_ground.json')
    if len(rvec_l2g) and len(tvec_l2g):
        save_extrinsic_param(lidar2Ground_path, rvec_l2g, tvec_l2g)
        print('load lidar_to_ground to', lidar2Ground_path)

    # lidar to gnss
    gnss_params = calib_params.get("gnss", [])
    if gnss_params:
        gnss_data = gnss_params[0] #list
        lidat_to_gnss_path = os.path.join(lidars_path, "lidar_to_gnss.json")
        rvec_gnss2ego = gnss_data["extrinsic_param"].get("rotation", [])
        tvec_gnss2ego = gnss_data["extrinsic_param"].get("translation", [])
        if len(rvec_gnss2ego) and len(tvec_gnss2ego):
            r_lidar2gnss, t_lidar2gnss = gnss_transform_c2f(rvec_gnss2ego, tvec_gnss2ego, rvec_main, tvec_main)
            save_extrinsic_param(lidat_to_gnss_path, r_lidar2gnss, t_lidar2gnss)
            print('load lidar_to_gnss to', lidat_to_gnss_path)
        else:
            warnings.warn(f"extrinsic of gnss is empty.") 
    else: 
        warnings.warn("calib_params.json which input is lack of gnss.")  
    
    # bpearl_lidar & falcon
    for lidar in calib_params.get("lidar", []):
        lidar_name = lidar.get("name")
        if  lidar_name == "falcon":
            falconLidar_path = os.path.join(lidars_path, 'falcon_lidar_to_main_lidar.json')
            r_falcon2ego = lidar["extrinsic_param"].get("rotation", [])
            t_falcon2ego = lidar["extrinsic_param"].get("translation", [])
            if len(r_falcon2ego) and len(t_falcon2ego):
                if len(rvec_main) and len(tvec_main) :
                    r_falcon2main, t_falcon2main = lidar_transform_c2f(r_falcon2ego, t_falcon2ego, rvec_main, tvec_main)
                    save_extrinsic_param(falconLidar_path, r_falcon2main, t_falcon2main)
                    print('load falcon to', falconLidar_path)
            else:
                warnings.warn(f"extrinsic of lidar {lidar_name} is empty.")  

        if lidar_name.startswith("side_lidar"):
            bp_name = bpearl_name_c2f(lidar_name)
            r_bpearl2ego = lidar["extrinsic_param"].get("rotation", [])
            t_bpearl2ego = lidar["extrinsic_param"].get("translation", [])
            if len(r_bpearl2ego) == 0 or len(t_bpearl2ego) == 0 or len(rvec_main) == 0 or len(tvec_main) == 0:
                warnings.warn(f"extrinsic of lidar {lidar_name} is empty.")  
            else: 
                r_bpearl2main, t_bpearl2main = lidar_transform_c2f(r_bpearl2ego, t_bpearl2ego, rvec_main, tvec_main)
                bpearl_data[bp_name]= {
                    "rvec": [[val] for val in r_bpearl2main],
                    "tvec": [[val] for val in t_bpearl2main]
                }
                sideLidar_path = os.path.join(lidars_path, 'bpearl_lidar_to_main_lidar.json')
                with open(sideLidar_path, 'w') as bpfile:
                    json.dump(bpearl_data, bpfile, indent=4)
                print('load bpear_lidar to', sideLidar_path)

    for camera in calib_params.get("camera", []):
        resolution_1080p_path = os.path.join(vehicle_path, 'resolution_1080P')
        os.makedirs(resolution_1080p_path, exist_ok=True)
        camera_name = camera_name_c2f(camera.get("name"))

        rvec_cam = camera["extrinsic_param"]["rotation"]
        tvec_cam = camera["extrinsic_param"]["translation"]
        if len(rvec_cam) == 0 or len(tvec_cam) == 0 or len(rvec_main) == 0 or len(tvec_main) == 0:
            warnings.warn(f"extrinsic of camera {camera_name} is empty.")       
        else:
            camera_file_path = os.path.join(resolution_1080p_path, camera_name)
            if not os.path.exists(camera_file_path):
                os.mkdir(camera_file_path)
            cam_extrinsic_path = os.path.join(camera_file_path, 'extrinsics.json')
            r_came2main,t_cam2main = camera_transform_c2f(rvec_cam, tvec_cam, rvec_main, tvec_main)
            save_extrinsic_param(cam_extrinsic_path, r_came2main, t_cam2main)
            print('load camera extrinsic_data to', cam_extrinsic_path)

            cam_intrinsic_path = os.path.join(camera_file_path, 'intrinsics.json')
            save_intrinsic_param(cam_intrinsic_path, camera["intrinsic_param"])
            print('load camera intrinsic_data to', cam_intrinsic_path)

## mode 1
def vehicleFold_To_calibFile(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError("Error: Input fold cannot be found")

    if not output_path.endswith(".json"):
        raise ValueError("Output path is not a JSON file")

    generate_calib_params_json(output_path)
    with open(output_path, 'r') as f:
        calib_params = json.load(f)

    for root, folds, files in os.walk(input_path):
        #load "global_config.yaml"
        if not "global_config.yaml" in files:
            raise ValueError("Error: cannot find global_config.yaml")
        global_config_path = os.path.join(input_path, 'global_config.yaml')
        vehicle_data, wheel_radius = vehicle_info_f2c(global_config_path)
        calib_params["vehicle_info"] = vehicle_data

        # load "lidar"
        if not "lidar" in folds:
            raise ValueError("Error: Unable to obtain Lidar information")
        lidar_path = os.path.join(root, 'lidar')
        # lidar_to_ego
        lidar_to_ego_path = os.path.join(lidar_path, 'lidar_to_ego.json')
        if not os.path.exists(lidar_to_ego_path):
            raise ValueError("Error: Unable to get main Lidar")
        print('load main lidar data from: ', lidar_to_ego_path)
        rvec_main, tvec_main = extract_extrincs_f2c(lidar_to_ego_path)
        tvec_main[2][0] = tvec_main[2][0] + wheel_radius
        lidar_data = lidar_module_f2c(rvec_main, tvec_main, "lidar_top")
        calib_params["lidar"].append(lidar_data) 

        T_lidar2ego = vector_to_matrix(rvec_main, tvec_main)
        # bpearl_lidar_to_main_lidar
        bpearl_lidar_path = os.path.join(lidar_path, 'bpearl_lidar_to_main_lidar.json')
        if os.path.exists(bpearl_lidar_path):
            with open(bpearl_lidar_path, 'r') as f:
                bpearlLidar_data = json.load(f)
                print('load bpearl Lidar_data from: ', bpearl_lidar_path)

            for lidar_name, lidar_info in bpearlLidar_data.items():
                rvec_bp2main = lidar_info.get("rvec", [])
                tvec_bp2main = lidar_info.get("tvec", [])
                T_bpearl2main = vector_to_matrix(rvec_bp2main, tvec_bp2main)
                T_bpearl2ego = np.dot(T_lidar2ego, T_bpearl2main)
                rvec_bp2ego, tvec_bp2ego = matrix_to_vector(T_bpearl2ego)
                bpearl_lidar_data = lidar_module_f2c(rvec_bp2ego, tvec_bp2ego, lidar_name) 
                calib_params["lidar"].append(bpearl_lidar_data) 

        # falcon_lidar_to_main_lidar
        falcon_lidar_path = os.path.join(lidar_path, 'falcon_lidar_to_main_lidar.json')
        if os.path.exists(falcon_lidar_path):
            with open(falcon_lidar_path, 'r') as f:
                falconLidar_data = json.load(f)
                print('load falcon lidar data from: ', falcon_lidar_path)
            rvec_fal2main, tvec_fal2main = extract_extrincs_f2c(falcon_lidar_path)
            T_falconl2main = vector_to_matrix(rvec_fal2main, tvec_fal2main)
            T_falconl2ego = np.dot(T_lidar2ego, T_falconl2main)
            rvec_fal2ego, tvec_fal2ego = matrix_to_vector(T_falconl2ego)
            falcon_lidar_data = lidar_module_f2c(rvec_fal2ego, tvec_fal2ego, "falcon")
            calib_params["lidar"].append(falcon_lidar_data) 

        # gnss
        gnss_path = os.path.join(lidar_path, 'lidar_to_gnss.json')
        if os.path.exists(gnss_path):
            with open(gnss_path, 'r') as f:
                lidat_to_gnss = json.load(f)
                print('load gnss data from: ', gnss_path)
            rvec_lidar2gnss, tvec_lidar2gnss = extract_extrincs_f2c(gnss_path)
            gnss_data = gnss_module_f2c(rvec_lidar2gnss, tvec_lidar2gnss, rvec_main, tvec_main)
            calib_params["gnss"].append(gnss_data) 

        # load "camera"
        for fold in folds:
            if fold.startswith("ofilm"):
                invalid_camera_fold = os.path.join(root, fold)
                warnings.warn(f"invalid camera fold: {invalid_camera_fold}")
        resolution_1080p_path = os.path.join(root, 'resolution_1080P')
        if not "resolution_1080P" in folds:
            raise ValueError("Error: The 'resolution_1080P' folder does not exist")
        for root_1080P, cams_folders, cam_files in os.walk(resolution_1080p_path):
            for cams in cams_folders:
                cams_path = os.path.join(resolution_1080p_path, cams)
                camera_1080_data = camera_module_f2c(cams_path, T_lidar2ego, global_config_path)
                calib_params["camera"].append(camera_1080_data)
        break

    with open(output_path, 'w') as f:
        json.dump(calib_params, f, indent=4)

def main(json_file_path, output_dir):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', type=str, required=True, help="path of calib_params.json")
    # parser.add_argument('-o', '--output', type=str, required=True, help="Output path")
    # parser.add_argument('-m', '--mode', type=str, choices=['f2c', 'c2f'], required=True, help="Mode of operation(f2c: to calib.json, c2f: to vehicle_fold)")
    # args = parser.parse_args()

    calib_file_to_vehicle_fold(json_file_path, output_dir)
    # if args.mode == 'f2c':
    #     vehicleFold_To_calibFile(args.input, args.output)
    # elif args.mode == 'c2f':
    #     calib_file_to_vehicle_fold(args.input, args.output)
    # else:
    #     raise ValueError("Error: Invalid mode")


""" 
功能:将calib_params.json文件转换成旧格式的内外参形式
示例:
    python calib_file_conversion.py -i calib_params.json -o /xxx/xxx/vehicle_fold
"""

if __name__ == "__main__":

    if len(sys.argv) !=3:
        print("Usage: python calib_file_conversion.py calib_params.json output_dir")
        sys.exit(1)

    json_file_path = sys.argv[1]
    output_dir = sys.argv[2]
    main(json_file_path, output_dir)

