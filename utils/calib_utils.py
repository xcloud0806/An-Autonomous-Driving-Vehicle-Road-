import os
import json
import copy
import numpy as np
import cv2

def project_lidar2img(points, image, calib_camera, point_size):
    H, W = image.shape[:2]
    lidar_to_image = calib_camera['new_lidar_to_image']
    points = np.matmul(points, lidar_to_image[:3, :3].T) + lidar_to_image[:3, 3]
    points[:, :2] /= points[:, 2:3]

    extended_points = [points]
    for i in range(-point_size, point_size + 1):
        for j in range(-point_size, point_size + 1):
            new_points = np.copy(points)
            new_points[:, 0] += i
            new_points[:, 1] += j

            extended_points.append(new_points)
    points = np.concatenate(extended_points, axis=0)

    mask = (
        (0 <= points[:, 0])
        & (points[:, 0] < W)
        & (0 <= points[:, 1])
        & (points[:, 1] < H)
        & (0 < points[:, 2])
    )
    points = points[mask]

    mean = np.mean(points[:, 2])
    points[points[:, 2] > (2 * mean), 2] = 2 * mean
    points[:, 2] /= 4 * mean

    depth_map = np.zeros([H, W], dtype=np.float32)
    depth_map[points[:, 1].astype(int), points[:, 0].astype(int)] = points[:, 2]

    mask = depth_map > 0

    depth_map = 255 * (1 - depth_map)
    depth_map = cv2.applyColorMap(depth_map.astype(np.uint8), cv2.COLORMAP_JET)

    image = image.astype(np.float32)
    image[mask] += depth_map[mask] / 2

    return image

def load_meta(meta, cam_names):
    calibration = meta['calibration']
    exts = calibration['extrinsics']
    ints = calibration['intrinsics']
    calibs = {}
    for cam_name in cam_names:
        calibs[cam_name] = {}

    for ext_ in exts:
        cam_name = ext_['target']
        if cam_name in cam_names:
            r, t = np.array(ext_["rvec"]), np.array(ext_["tvec"])
            r = cv2.Rodrigues(r)[0]
            r = np.reshape(r, [3, 3])
            t = np.reshape(t, [3, 1])
            extrinsic = np.concatenate([r, t], -1)
            calibs[cam_name]['extrinsics'] = extrinsic

    for int_ in ints:
        cam_name = int_['sensor_position']
        if cam_name in cam_names:
            intrinsic = np.array(int_["mtx"], dtype=np.float32).reshape([3, 3])
            distortion = np.array(int_["dist"], dtype=np.float32).reshape([-1])
            calibs[cam_name]['intrinsics'] = intrinsic
            calibs[cam_name]['distortion'] = distortion
            calibs[cam_name]['image_size'] = int_['image_size']
            calibs[cam_name]['cam_model'] = int_["cam_model"]

    return calibs

def undistort(calib_camera, img):
    image_shape=calib_camera['image_shape']

    if img is not None:
        if img.shape[1]!=image_shape[0] or img.shape[0]!=image_shape[1]:
            img = cv2.resize(img, image_shape, interpolation=cv2.INTER_LINEAR)

    if calib_camera['undistort_mode'] == "fisheye":
        img = cv2.remap(
            img,
            calib_camera['map1'],
            calib_camera['map2'],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
    elif calib_camera['undistort_mode'] == "pinhole":
        img = cv2.undistort(
            img,
            calib_camera['intrinsic'],
            calib_camera['distortion'],
            newCameraMatrix=calib_camera['undist_intrinsic'],
        )
    else:
        raise TypeError("wrong mode: %s" % calib_camera['mode'])
    return img

def load_calibration(meta, cam_name, new_image_shape=None):
    camera_names = meta['cameras']
    calibs = load_meta(meta, camera_names)
    calib_raw_dict = calibs[cam_name]
    extrinsic = calib_raw_dict["extrinsics"]

    intrinsic = calib_raw_dict['intrinsics']
    distortion = calib_raw_dict['distortion']
    image_shape = calib_raw_dict["image_size"]
    undistort_mode = calib_raw_dict["cam_model"]
    
    calib_camera = {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "distortion": distortion,
        "image_shape": image_shape,
        "undistort_mode": undistort_mode,
    }

    if undistort_mode == "fisheye":
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            intrinsic, distortion, np.eye(3), intrinsic, image_shape, cv2.CV_16SC2
        )
        calib_camera['map1'] = map1
        calib_camera['map2'] = map2
        calib_camera['undist_intrinsic'] = intrinsic
    elif undistort_mode == "pinhole":
        undist_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
            intrinsic,
            distortion,
            image_shape,
            alpha=0.0,
            newImgSize=image_shape,
        )
        calib_camera['undist_intrinsic'] = undist_intrinsic
    else:
        raise TypeError("wrong mode: %s" % undistort_mode)

    if new_image_shape is not None:
        calib_camera['new_image_shape'] = new_image_shape
        calib_camera['new_undist_intrinsic'] = copy.deepcopy(calib_camera['undist_intrinsic'])
        calib_camera['new_undist_intrinsic'][0] = new_image_shape[0]/image_shape[0]*calib_camera['new_undist_intrinsic'][0]
        calib_camera['new_undist_intrinsic'][1] = new_image_shape[1]/image_shape[1]*calib_camera['new_undist_intrinsic'][1]
        calib_camera['new_lidar_to_image'] = np.matmul(np.array(calib_camera['new_undist_intrinsic']), 
                                np.array(calib_camera["extrinsic"]))
    return calib_camera

def load_bpearls(meta):
    calibration = meta['calibration']
    exts = calibration['extrinsics']

    ret = {}
    for ext_ in exts:
        sensor = ext_['target']
        if 'bpearl' in sensor or 'inno' in sensor:
            r, t = np.array(ext_['rvec']), np.array(ext_['tvec'])
            r = cv2.Rodrigues(r)[0]
            r = np.reshape(r, [3, 3])
            t = np.reshape(t, [3, 1])
            extrinsic = np.concatenate([r, t], -1)
            ret[sensor] = extrinsic
    return ret
