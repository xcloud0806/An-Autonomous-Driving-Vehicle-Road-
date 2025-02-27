import os, sys
import argparse
import json
from uuid import uuid4
from . import CarMeta
from .cam_utils import cam_position

META_FILE = "car_meta.json"

IN_FILE = "intrinsics.json"
EX_FILE = "extrinsics.json"
EGO_FILE = "lidar_to_ego.json"
GNSS_FILE = "lidar_to_gnss.json"
GROUND_FILE = "lidar_to_ground.json"
INNO_FILE = "lidar_to_inno.json"

def cam_positon_split(pos:str):
    ret = pos.split("_")
    fov = ret[-2]
    cnt = len(ret)
    start = 0
    resolution = ret[-1]
    supplier = "Sensing"
    if ret[0].upper() == "OFILM":
        supplier = "Ofilm"
        start = 1
    pos = "_".join(item for item in ret[start: cnt - 2])
    return supplier, pos, fov, resolution

def parse_args():
    parser = argparse.ArgumentParser(description="car calibration combine")
    parser.add_argument('--input', '-i', type=str, default='/data_autodrive/auto/calibration/' )
    parser.add_argument('--car', '-c', type=str, default='sihao_3xx23')
    parser.add_argument('--date', '-d', type=str, required=True)
    
    args = parser.parse_args()
    return args

def func_combine(car_root_path):
    ret = {}
    ret['uuid'] = str(uuid4())
    date = os.path.basename(car_root_path)
    if date == '':
        date = os.path.basename(car_root_path[:-1])
    ret['date'] = date
    ret['sensors'] = ['gnss', 'vehicle', 'lidar']
    exts = []
    ints = []

    car_meta_json = os.path.join(car_root_path, META_FILE)
    car_meta = CarMeta()
    car_meta.from_json(car_meta_json)
    car_meta_cameras = car_meta.cameras

    if len(car_meta.other_sensors_info) > 0:
        if len(car_meta.bpearl_lidars) > 0:
            ret['sensors'].extend(car_meta.bpearl_lidars)
            
        if len(car_meta.inno_lidars) > 0:
            ret['sensors'].extend(car_meta.inno_lidars)
        
        ret['sensors_info'] = car_meta.other_sensors_info

    for cam_id, cam in enumerate(cam_position):
        # cam_name = cam_names[cam_id]
        calib_cam_path = os.path.abspath(os.path.join(car_root_path, cam))
        if not os.path.exists(calib_cam_path) or cam not in car_meta_cameras:
            continue
        ret['sensors'].append(cam)
        supplier, cam_pos, cam_fov, cam_resolution = cam_positon_split(cam)
        int_file = os.path.join(calib_cam_path, IN_FILE)
        with open(int_file, "r") as fp:
            intrinsics = json.load(fp)
            intrinsics['sensor_position'] = cam
            intrinsics['sensor_name'] = cam
            intrinsics['sensor_fov'] = cam_fov
            ints.append(intrinsics)
        ext_file = os.path.join(calib_cam_path, EX_FILE)
        with open(ext_file, "r") as fp:
            extrinsics = json.load(fp)
            # rvec = np.array(extrinsics.pop('rvec'), dtype=np.float32)
            extrinsics["source"]= 'lidar'
            extrinsics["target"]= cam
            # extrinsics['rmat'] = cv2.Rodrigues(rvec)[0].tolist()
            exts.append(extrinsics)
    
    gnss_file = os.path.join(car_root_path, GNSS_FILE)
    ground_file = os.path.join(car_root_path, GROUND_FILE)
    ego_file = os.path.join(car_root_path, EGO_FILE)
    inno_file = os.path.join(car_root_path, INNO_FILE)
    bpearl_front_file = os.path.join(car_root_path, "bpearls", "lidar_to_front.json")
    bpearl_rear_file = os.path.join(car_root_path, "bpearls", "lidar_to_rear.json")
    bpearl_left_file = os.path.join(car_root_path, "bpearls", "lidar_to_left.json")
    bpearl_right_file = os.path.join(car_root_path, "bpearls", "lidar_to_right.json")

    if not os.path.exists(gnss_file) or not os.path.exists(ground_file) or not os.path.exists(ego_file):
        print("Warn: No GNSS/GROUND/EGO Calib File.")
    else:
        with open(gnss_file, "r") as fp:
            gnss_calib = json.load(fp)
            gnss_calib['source'] = 'lidar'
            gnss_calib['target'] = 'gnss'
            exts.append(gnss_calib)
            
        with open(ego_file, "r") as fp:
            ego_calib = json.load(fp)
            ego_calib['source'] = 'lidar'
            ego_calib['target'] = 'ego'
            exts.append(ego_calib)
        
        with open(ground_file, "r") as fp:
            groud_calib = json.load(fp)
            groud_calib['source'] = 'lidar'
            groud_calib['target'] = 'ground'
            exts.append(groud_calib)
            
    if os.path.exists(inno_file):
        with open(inno_file, "r") as fp:
            inno_calib = json.load(fp)
            inno_calib['source'] = 'lidar'
            inno_calib['target'] = 'inno_lidar'
            exts.append(inno_calib)
    
    if not os.path.exists(bpearl_front_file) or not os.path.exists(bpearl_rear_file) \
        or not os.path.exists(bpearl_left_file) or not os.path.exists(bpearl_right_file):
        print("Warn: No BPEARL Calib File.")
    else:
        with open(bpearl_front_file, "r") as fp:
            front_calib = json.load(fp)
            front_calib['source'] = 'lidar'
            front_calib['target'] = 'bpearl_lidar_front'
            exts.append(front_calib)
        with open(bpearl_rear_file, "r") as fp:
            rear_calib = json.load(fp)
            rear_calib['source'] = 'lidar'
            rear_calib['target'] = 'bpearl_lidar_rear'
            exts.append(rear_calib)
        with open(bpearl_left_file, "r") as fp:
            left_calib = json.load(fp)
            left_calib['source'] = 'lidar'
            left_calib['target'] = 'bpearl_lidar_left'
            exts.append(left_calib)
        with open(bpearl_right_file, "r") as fp:
            right_calib = json.load(fp)
            right_calib['source'] = 'lidar'
            right_calib['target'] = 'bpearl_lidar_right'
            exts.append(right_calib)
            
    ret['intrinsics'] = ints
    ret['extrinsics'] = exts
    return ret


def func_combine_v1(rootpath, carname, date):
    car_root_path = os.path.join(rootpath, carname, date)
    ret = {}
    ret['uuid'] = str(uuid4())
    ret['date'] = date
    ret['sensors'] = ['gnss', 'vehicle', 'lidar']
    exts = []
    ints = []

    car_meta_json = os.path.join(car_root_path, META_FILE)
    car_meta = CarMeta()
    car_meta.from_json(car_meta_json)
    car_meta_cameras = car_meta.cameras

    if len(car_meta.other_sensors_info) > 0:
        if len(car_meta.bpearl_lidars) > 0:
            ret['sensors'].extend(car_meta.bpearl_lidars)
            
        if len(car_meta.inno_lidars) > 0:
            ret['sensors'].extend(car_meta.inno_lidars)
        
        ret['sensors_info'] = car_meta.other_sensors_info

    for cam_id, cam in enumerate(cam_position):
        # cam_name = cam_names[cam_id]
        calib_cam_path = os.path.abspath(os.path.join(car_root_path, cam))
        if not os.path.exists(calib_cam_path) or cam not in car_meta_cameras:
            continue
        ret['sensors'].append(cam)
        supplier, cam_pos, cam_fov, cam_resolution = cam_positon_split(cam)
        int_file = os.path.join(calib_cam_path, IN_FILE)
        with open(int_file, "r") as fp:
            intrinsics = json.load(fp)
            intrinsics['sensor_position'] = cam
            intrinsics['sensor_name'] = cam
            intrinsics['sensor_fov'] = cam_fov
            ints.append(intrinsics)
        ext_file = os.path.join(calib_cam_path, EX_FILE)
        with open(ext_file, "r") as fp:
            extrinsics = json.load(fp)
            # rvec = np.array(extrinsics.pop('rvec'), dtype=np.float32)
            extrinsics["source"]= 'lidar'
            extrinsics["target"]= cam
            # extrinsics['rmat'] = cv2.Rodrigues(rvec)[0].tolist()
            exts.append(extrinsics)
    
    gnss_file = os.path.join(car_root_path, GNSS_FILE)
    ground_file = os.path.join(car_root_path, GROUND_FILE)
    ego_file = os.path.join(car_root_path, EGO_FILE)
    inno_file = os.path.join(car_root_path, INNO_FILE)
    bpearl_front_file = os.path.join(car_root_path, "bpearls", "lidar_to_front.json")
    bpearl_rear_file = os.path.join(car_root_path, "bpearls", "lidar_to_rear.json")
    bpearl_left_file = os.path.join(car_root_path, "bpearls", "lidar_to_left.json")
    bpearl_right_file = os.path.join(car_root_path, "bpearls", "lidar_to_right.json")

    if not os.path.exists(gnss_file) or not os.path.exists(ground_file) or not os.path.exists(ego_file):
        print("Warn: No GNSS/GROUND/EGO Calib File.")
    else:
        with open(gnss_file, "r") as fp:
            gnss_calib = json.load(fp)
            gnss_calib['source'] = 'lidar'
            gnss_calib['target'] = 'gnss'
            exts.append(gnss_calib)
            
        with open(ego_file, "r") as fp:
            ego_calib = json.load(fp)
            ego_calib['source'] = 'lidar'
            ego_calib['target'] = 'ego'
            exts.append(ego_calib)
        
        with open(ground_file, "r") as fp:
            groud_calib = json.load(fp)
            groud_calib['source'] = 'lidar'
            groud_calib['target'] = 'ground'
            exts.append(groud_calib)
            
    if os.path.exists(inno_file):
        with open(inno_file, "r") as fp:
            inno_calib = json.load(fp)
            inno_calib['source'] = 'lidar'
            inno_calib['target'] = 'inno_lidar'
            exts.append(inno_calib)
    
    if not os.path.exists(bpearl_front_file) or not os.path.exists(bpearl_rear_file) \
        or not os.path.exists(bpearl_left_file) or not os.path.exists(bpearl_right_file):
        print("Warn: No BPEARL Calib File.")
    else:
        with open(bpearl_front_file, "r") as fp:
            front_calib = json.load(fp)
            front_calib['source'] = 'lidar'
            front_calib['target'] = 'bpearl_lidar_front'
            exts.append(front_calib)
        with open(bpearl_rear_file, "r") as fp:
            rear_calib = json.load(fp)
            rear_calib['source'] = 'lidar'
            rear_calib['target'] = 'bpearl_lidar_rear'
            exts.append(rear_calib)
        with open(bpearl_left_file, "r") as fp:
            left_calib = json.load(fp)
            left_calib['source'] = 'lidar'
            left_calib['target'] = 'bpearl_lidar_left'
            exts.append(left_calib)
        with open(bpearl_right_file, "r") as fp:
            right_calib = json.load(fp)
            right_calib['source'] = 'lidar'
            right_calib['target'] = 'bpearl_lidar_right'
            exts.append(right_calib)
            
    ret['intrinsics'] = ints
    ret['extrinsics'] = exts
    return ret

if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = parse_args()
        car_root_path = os.path.join(args.input, args.car, args.date)
        func_combine(car_root_path)
    else:
        car_root_path = "/data_autodrive/users/brli/dev_raw_data/calibs/20241119"
        res = func_combine(car_root_path)
        with open("res.json", "w") as fp:
            ss = json.dumps(res, indent=2)
            fp.write(ss)


