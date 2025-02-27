import os, sys
import numpy as np
from copy import deepcopy
import cv2
import json
from tsmoothie.smoother import LowessSmoother
import PIL.Image as Image
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

RGB_WHITE=(255,255,255)
RGB_YELLOW=(0,255,255)
RGB_BLUE=(255,0,0)
RGB_RED=(0,0,255)
RGB_GREEN=(0,255,0)
RGB_BLACK=(0,0,0)

def get_color(idx, norm=False, plt=False):
    idx = int(idx)
    idx = idx * 3
    # openc:bgr, image:rgb
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    if norm:
        color = (color[0]/255, color[1]/255, color[2]/255)
    if plt:
        color = (color[2], color[1], color[0])
    return color

def gen_clip_lane_info(clip_lane:str):
    rs_root_path = clip_lane
    for f in os.listdir(rs_root_path):        
        if f.endswith("_rgb.jpg") or f.endswith('_rgb.jpeg'):
            rs_img_path = os.path.join(rs_root_path, f)
            rs_img = cv2.imread(rs_img_path)

        if f.endswith('npy'):
            rs_height_path = os.path.join(rs_root_path, f)
            rs_height = np.load(rs_height_path)
    
    return rs_img, rs_height

def matrixProduct3x4(matrix, point):
    out = [0,0,0]
    out[0] = matrix[0][0]*point[0]+matrix[0][1]*point[1]+matrix[0][2]*point[2]+matrix[0][3]
    out[1] = matrix[1][0]*point[0]+matrix[1][1]*point[1]+matrix[1][2]*point[2]+matrix[1][3]
    out[2] = matrix[2][0]*point[0]+matrix[2][1]*point[1]+matrix[2][2]*point[2]+matrix[2][3]
    return out

def point_bev_to_world(world_to_img, img_to_world, point_bev):
    uv = np.array([point_bev[0], point_bev[1]])
    xy = np.matmul(img_to_world[:2, :2], uv.T) + img_to_world[ :2, 3:].T
    return [float(xy[0][0]), float(xy[0][1]), point_bev[2]]

def point_bev_to_world_new(world_to_img, img_to_world, point_bev):
    point_2d_label = [point_bev[0],point_bev[1],0]
    point_world = matrixProduct3x4(img_to_world, point_2d_label)
    point_world[2] = point_bev[2]
    return [float(point_world[0]), float(point_world[1]), point_bev[2]] 

def point_world_to_bev(world_to_img, img_to_world, point_world):
    xy = np.array(point_world).reshape((3,))
    pq = np.matmul(world_to_img[:3, :3], xy.T) 
    uv = pq + world_to_img[:3, 3:].T
    uv = uv.reshape((3,))
    return [float(uv[0]), float(uv[1]), float(uv[2])]

def point_world_to_bev_new(world_to_img, img_to_world, point_world):
    point_img = matrixProduct3x4(world_to_img, point_world)
    # point_world[2] = point_bev[2]
    return [float(point_img[0]), float(point_img[1]), point_img[2]] 

def pose_to_bev(pose, world_to_img, img_to_world):
    pt = pose.T[3:, :3]
    pt = pt.reshape((3,)) # [:2].astype(np.int32).tolist()
    pt_cur_bev = point_world_to_bev(world_to_img, img_to_world, pt)
    pt = [int(pt_cur_bev[0]), int(pt_cur_bev[1])]   
    return pt  

def main(seg_root:str):
    meta_file = ""
    for f in os.listdir(seg_root):
        _f = os.path.join(seg_root, f)
        if os.path.isfile(_f) and f.endswith('multi_meta.json'):
            meta_file = _f
            break
    if not os.path.exists(meta_file):
        return
    # meta_file = os.path.join(seg_root, f"{coll_id}_multi_meta.json")
    meta = json.load(open(meta_file, 'r'))
    frames = meta['frames']
    lidar_poses = []
    for frame in frames:
        lidar_poses.append(frame['lidar']['pose'])

    reconstruct_dir = os.path.join(seg_root, "reconstruct")
    if not os.path.exists(reconstruct_dir):
        return
    rs_img, rs_height = gen_clip_lane_info(reconstruct_dir)
    transform_json = os.path.join(reconstruct_dir, "transform_matrix.json")
    transform = json.load(open(transform_json, 'r'))
    region = transform['region']
    img_to_world = np.array(transform['img_to_world'])
    world_to_img = np.linalg.pinv(img_to_world)
    for i in range(len(lidar_poses)):
        pose = np.array(lidar_poses[i])
        pt = pose_to_bev(pose, world_to_img, img_to_world)
        cv2.circle(rs_img, (int(pt[0]), int(pt[1])), 5, (0,0,255), thickness=2)
    cv2.imwrite(os.path.join(seg_root, "bev_traj_visual.jpg"), rs_img)

if __name__ == "__main__":
    # main(
    #     "/data_cold2/ripples/chery_13484/custom_seg/lane_change_crimping/20240521_n/chery_13484_20240522-00-06-19_seg21", 
    #     "20240523-14-34-08-nyXDmB25"
    #     )
    root_path = "/data_cold2/origin_data/chery_13484/custom_seg/frwang_chadaohuichu/night/20240603_1"
    segs = os.listdir(root_path)
    segs.sort()
    for i, seg in enumerate(segs):
        seg_path = os.path.join(root_path, seg)
        print(f"visual {i}/{len(segs)} <-> {seg}")
        main(seg_path)
