import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import json
import torch
import onnxruntime
import multiprocessing as mp
from multiprocessing import Process, Pool
from utils import undistort, load_calibration
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import pcl
from python.yolo_onnx_v1 import yolov5_iflytek

def pool_oriimg(height):
    original_data = height
    original_tensor = torch.from_numpy(original_data).unsqueeze(0).unsqueeze(0).float()
    pooling = torch.nn.AvgPool2d(2)
    downsampled_tensor = pooling(original_tensor)
    downsampled_data = downsampled_tensor.squeeze(0).squeeze(0).numpy()
    return downsampled_data

def transnpy(height):
    filter_size = 3
    h = height.shape[0]
    w = height.shape[1]
    height = torch.tensor(height)
    height = height.unsqueeze(0).unsqueeze(0)
    result = torch.zeros_like(height)
    for m in range(-filter_size // 2, filter_size // 2 + 1):
        for n in range(-filter_size // 2, filter_size // 2 + 1):
            if m == 0 and n == 0:
                continue
            shifted_height = height[:, :, max(0, -m):h - m, max(0, -n):w - n]
            nonzero_indices = torch.nonzero(shifted_height)
            result[:, :, nonzero_indices[:, 2] + m, nonzero_indices[:, 3] + n] = shifted_height[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2], nonzero_indices[:, 3]]
    result = result.squeeze().squeeze().numpy()
    return result


def split_list(lst, segment_length):
    return [lst[i:i+segment_length] for i in range(0, len(lst), segment_length)]

def read_images(pair_list,cam_name,root_path,calib_camera): #16张在一起读取一起绘图
    temp_dict = {}
    for frame in pair_list:
        if cam_name not in frame['images']:
            print('{} is not in pair'.format(cam_name))
            continue
        img_path = os.path.join(root_path,frame['images'][cam_name]['path'])
        img = cv2.imread(img_path)
        img = undistort(calib_camera, img)  # 读取图像和校正畸变
        temp_dict[frame['images'][cam_name]['path']] = img
    return temp_dict


def get_lidar_to_ground(infos):
    for tempe in infos["calibration"]["extrinsics"]:
        if tempe["target"] == 'ground':
            r, t = np.array(tempe["rvec"]), np.array(tempe["tvec"])
            r = cv2.Rodrigues(r)[0]
            r = np.reshape(r, [3, 3])
            t = np.reshape(t, [3, 1])
            extrinsico = np.concatenate([r, t], -1)
            extrinsic = np.vstack([extrinsico, [0, 0, 0, 1]])
            return extrinsic


def get_world_to_img(region, resolution):
    world_to_img = np.array([[1/resolution[0], 0, 0, -region[0]/resolution[0]],
                             [0, 1/resolution[1], 0, -region[2]/resolution[1]],
                             [0, 0, 1/resolution[2], -region[4]/resolution[2]]])
    world_to_img[1] = -world_to_img[1]  # flip top down
    world_to_img[1, 3] = ((region[3]-region[2]) /
                             resolution[1])+world_to_img[1, 3]   
    return world_to_img


def get_img_to_world(world_to_img):
    return np.linalg.inv(world_to_img)

def get_frame_to_point(world_to_img, pose):
    pose_img = np.matmul(world_to_img[:2, :2], pose[:2, 3])+world_to_img[:2, 3]
    pose_img = pose_img[:2].tolist()
    return pose_img

def selectframe(pre_pose,cur_pose):
    pre_position = pre_pose[:3, 3]
    cur_position = cur_pose[:3, 3]
    translation_diff = cur_position - pre_position
    translation_distance = np.linalg.norm(translation_diff)
    return translation_distance

class LidarShowVoxel():
    def __init__(self, region, resolution):
        self.region = region
        self.resolution = resolution
        self.ipx = int((region[1]-region[0])/resolution[0])
        self.ipy = int((region[3]-region[2])/resolution[1])
        self.ipz = int((region[5]-region[4])/resolution[2])

        self.world_to_img = get_world_to_img(region, resolution)
       
        self.img = np.zeros((self.ipy, self.ipx, 3), np.uint8)
        self.voxel_map_back = np.zeros(
            (self.ipx, self.ipy, self.ipz, 3), np.float32)
        self.voxel_map_other = np.zeros(
            (self.ipx, self.ipy, self.ipz, 3), np.float32)
        self.voxel_value = np.zeros(
            (self.ipx, self.ipy, self.ipz, 3), np.float32)
        self.voxel_cnt_back = np.zeros(
            (self.ipx, self.ipy, self.ipz), np.int32)
        self.voxel_cnt_other = np.zeros(
            (self.ipx, self.ipy, self.ipz), np.int32)
        self.voxel_cnt = np.zeros((self.ipx, self.ipy, self.ipz), np.int32)
        self.height = np.zeros((self.ipx, self.ipy, 3),
                               np.float32) 
    def draw(self, points_world, points_color,camera_name,cam):
        points_world_img = (np.matmul(
            self.world_to_img[:3, :3], points_world.T)+self.world_to_img[:3, [3]]).T
        points_world_img = points_world_img.astype(np.int32)

        flag_valid = (points_world_img[:, 0] > 0) & (points_world_img[:, 1] > 0) & (points_world_img[:, 2] > 0) & (
            points_world_img[:, 0] < self.ipx) & (points_world_img[:, 1] < self.ipy) & (points_world_img[:, 2] < self.ipz)
        points_world = points_world[flag_valid]
        points_world_img = points_world_img[flag_valid]
        points_color = points_color[flag_valid]

        if camera_name == cam:
            self.voxel_map_back[points_world_img[:,0],points_world_img[:,1],points_world_img[:,2],:]+=points_color
            self.voxel_cnt_back[points_world_img[:, 0],
                                points_world_img[:, 1], points_world_img[:, 2]] += 1
        else:
            self.voxel_map_other[points_world_img[:, 0],
                                 points_world_img[:, 1], points_world_img[:, 2], :] += points_color
            self.voxel_cnt_other[points_world_img[:, 0],
                                 points_world_img[:, 1], points_world_img[:, 2]] += 1
        self.voxel_value[points_world_img[:, 0], points_world_img[:, 1], points_world_img[:, 2]] += points_world
        self.voxel_cnt[points_world_img[:, 0],
                       points_world_img[:, 1], points_world_img[:, 2]] += 1

    def show(self,):
        voxel_sum = self.voxel_cnt_back.sum(-1)
        valid_x, valid_y = np.where(voxel_sum > 0)
        voxel_valid = self.voxel_cnt_back[valid_x, valid_y]

        voxel_valid[voxel_valid == 0] = 1e6
        min_z_idx = voxel_valid.argmin(axis=1)
        min_z_idx = min_z_idx.clip(0, self.ipz-2)

        self.voxel_map_back[valid_x,valid_y,min_z_idx]+=self.voxel_map_back[valid_x,valid_y,min_z_idx+1]
        self.voxel_cnt_back[valid_x,valid_y,min_z_idx]+=self.voxel_cnt_back[valid_x,valid_y,min_z_idx+1]
        self.voxel_cnt[valid_x,valid_y,min_z_idx]+=self.voxel_cnt[valid_x,valid_y,min_z_idx+1]
        self.voxel_value[valid_x,valid_y,min_z_idx]+=self.voxel_value[valid_x,valid_y,min_z_idx+1]

        points_xyz = self.voxel_map_back[valid_x,valid_y,min_z_idx]/self.voxel_cnt_back[valid_x,valid_y,min_z_idx][...,None]
        points_xyz = points_xyz.clip(0,255).astype(np.uint8).tolist()
        self.height[valid_x,valid_y,:3] = self.voxel_value[valid_x,valid_y,min_z_idx]/self.voxel_cnt[valid_x,valid_y,min_z_idx][...,None]
        
        for xi, yi, point_xyz in zip(valid_x, valid_y, points_xyz):
            cv2.circle(self.img, (xi,yi), 1, point_xyz, -1)
        voxel_sum_others = self.voxel_cnt_other.sum(-1)
        valid_x, valid_y = np.where((voxel_sum==0)&(voxel_sum_others>0)&(self.img.transpose(1,0,2).sum(-1)==0))
        if len(valid_x)>0:
            voxel_valid = self.voxel_cnt_other[valid_x,valid_y]
            voxel_valid[voxel_valid==0] = 1e6
            min_z_idx = voxel_valid.argmin(axis=1)
            min_z_idx = min_z_idx.clip(0,self.ipz-2)

            self.voxel_map_other[valid_x,valid_y,min_z_idx]+=self.voxel_map_other[valid_x,valid_y,min_z_idx+1]
            self.voxel_cnt_other[valid_x,valid_y,min_z_idx]+=self.voxel_cnt_other[valid_x,valid_y,min_z_idx+1]
            self.voxel_cnt[valid_x,valid_y,min_z_idx]+=self.voxel_cnt[valid_x,valid_y,min_z_idx+1]
            self.voxel_value[valid_x,valid_y,min_z_idx]+=self.voxel_value[valid_x,valid_y,min_z_idx+1]

            points_xyz = self.voxel_map_other[valid_x,valid_y,min_z_idx]/self.voxel_cnt_other[valid_x,valid_y,min_z_idx][...,None]
            points_xyz = points_xyz.clip(0,255).astype(np.uint8).tolist()
            self.height[valid_x,valid_y,:3] = self.voxel_value[valid_x,valid_y,min_z_idx]/self.voxel_cnt[valid_x,valid_y,min_z_idx][...,None]
            for xi, yi, point_xyz in zip(valid_x, valid_y, points_xyz):
                cv2.circle(self.img, (xi, yi), 1, point_xyz, -1)
        return self.img, self.height

class ReconstructionColormap():
    def __init__(self, args):
        self.root_path = args.root_path
        self.out_root_path = args.out_root_path
        self.camera_list = []
        self.map_camera_name = args.map_camera_name
        self.map_camera_name_add = args.map_camera_name_add
        self.image_shape = args.image_shape
        self.batch_size = args.batch_size
        

    def det_obstacle(self,img_lists):
        dets_lists = self.net2d_img_detect.detect(img_lists)
        masks = []
        for dets in dets_lists:
            mask = np.ones(img_lists[0].shape)*255
            for bbox in dets:
                cv2.rectangle(mask, (int(bbox[0]-5), int(bbox[1]-5)), (int(bbox[2]+5), int(bbox[3]+5)), color=(0, 0, 0), thickness=-1)
            masks.append(mask)
        return masks

   

    def odometer_by_pose(self, pair_pose_list, args):
        pair_list_cur = list()
        pose_list_cur = list()
        pair_list_after = list()
        pose_list_after = list()
        pair_list_before = list()
        pose_list_before = list()
        pose_before = np.identity(4) 
        cur_distance=0
        near_dis = 0
        region = [-20,20,1000,-1000,-3,3]
        start_flag = 0 
        recon_flag = -1
        pre_recon = False
        #确定起始帧 找出150m确定为起始帧，判断是否有按要求dis+extern_dis的距离 没有就不重建
        for li, pair in enumerate(pair_pose_list):
            if li==0:
                pose_before = np.array(pair["lidar"]["pose"])
            else:
                pose_cur = np.array(pair['lidar']['pose'])
                near_dis = selectframe(pose_before,pose_cur)
                if(near_dis<0.3):
                    continue
                if(near_dis > 4):
                    recon_flag = 1
                    pose_before = np.array(pair["lidar"]["pose"])
                    return pair_list_cur, pose_list_cur,  region, pair_list_before, pose_list_before, pair_list_after, pose_list_after,recon_flag
                else:
                    pose_before = np.array(pair["lidar"]["pose"])
                    cur_distance+= near_dis
            if cur_distance>args.extern_frame_distance and start_flag == 0:
                start_idx = li
                start_flag = 1
            if cur_distance> args.distance +  2*args.extern_distance + 50:
                pre_recon =True
                break
        if(pre_recon == False):
            if(args.rebuild_type=='changjing'):
                args.distance = cur_distance - 2*args.extern_frame_distance
                print("当前中间重建距离为:",args.distance)
                if(args.distance>100):
                    pre_recon = True
                else:
                    recon_flag = 0
                    return pair_list_cur, pose_list_cur,  region, pair_list_before, pose_list_before, pair_list_after, pose_list_after,recon_flag
        cur_distance = 0       
        for li, pair in enumerate(pair_pose_list):
            if li<start_idx:
                continue
            if(li==start_idx):
                pair_list_cur.append(pair)
                pose_list_cur.append(np.array(pair["lidar"]["pose"]))
                pose_before = np.array(pair["lidar"]["pose"])
            else:
                pose_cur = np.array(pair['lidar']['pose'])
                near_dis = selectframe(pose_before,pose_cur)
                if(near_dis<0.6):
                    continue
                if(near_dis > 4):
                    recon_flag = 1
                    return pair_list_cur, pose_list_cur,  region, pair_list_before, pose_list_before, pair_list_after, pose_list_after,recon_flag
                else:
                    pair_list_cur.append(pair)
                    pose_list_cur.append(np.array(pair["lidar"]["pose"]))
                    cur_distance+= near_dis
                    pose_before = np.array(pair["lidar"]["pose"])
            cur_position = pose_list_cur[-1][:3,3] 
            region = [min(region[0],cur_position[0]), max(region[1],cur_position[0]),
                      min(region[2],cur_position[1]), max(region[3],cur_position[1]),
                      min(region[4],cur_position[2]), max(region[5],cur_position[2])]          
            if cur_distance>args.distance:
                end_idx = li
                break
            if li==len(pair_pose_list)-1:
                end_idx = li                
        
        cur_distance = 0
        end_idx = li

        for li in range(end_idx+1, len(pair_pose_list)):
            pair = pair_pose_list[li]
            
            if(li==end_idx+1):
                pair_list_after.append(pair)
                pose_list_after.append(np.array(pair["lidar"]["pose"]))
                pose_before = np.array(pair["lidar"]["pose"])
            else:
                pose_cur = np.array(pair['lidar']['pose'])
                near_dis = selectframe(pose_before,pose_cur)
                if(near_dis<0.6):
                    continue
                if(near_dis > 4):
                    recon_flag = 1
                    return pair_list_cur, pose_list_cur,  region, pair_list_before, pose_list_before, pair_list_after, pose_list_after,recon_flag
                else:
                    pair_list_after.append(pair)
                    pose_list_after.append(np.array(pair["lidar"]["pose"]))
                    cur_distance+= near_dis
                    pose_before = np.array(pair["lidar"]["pose"])
       
            if cur_distance < args.extern_distance:
                cur_position = pose_list_after[-1][:3,3]
                region = [min(region[0],cur_position[0]), max(region[1],cur_position[0]),
                        min(region[2],cur_position[1]), max(region[3],cur_position[1]),
                        min(region[4],cur_position[2]), max(region[5],cur_position[2])] 
            
            if cur_distance>args.extern_frame_distance:
                break 
        
        cur_distance = 0

        for li in range(start_idx-1, 1,-1):
            pair = pair_pose_list[li]
            if(li==start_idx-1):
                pair_list_before.append(pair)
                pose_list_before.append(np.array(pair["lidar"]["pose"]))
                pose_before = np.array(pair["lidar"]["pose"])
            else:
                pose_cur = np.array(pair['lidar']['pose'])
                near_dis = selectframe(pose_before,pose_cur)
                if(near_dis<0.6):
                    continue
                if(near_dis > 4):
                    recon_flag = 1
                    return pair_list_cur, pose_list_cur,  region, pair_list_before, pose_list_before, pair_list_after, pose_list_after,recon_flag
                else:
                    pair_list_before.append(pair)
                    pose_list_before.append(np.array(pair["lidar"]["pose"]))
                    cur_distance+= near_dis
                    pose_before = np.array(pair["lidar"]["pose"])
            if cur_distance < args.extern_distance:
                cur_position = pose_list_before[-1][:3,3]
                region = [min(region[0],cur_position[0]), max(region[1],cur_position[0]),
                      min(region[2],cur_position[1]), max(region[3],cur_position[1]),
                      min(region[4],cur_position[2]), max(region[5],cur_position[2])] 

            if cur_distance>args.extern_frame_distance:
                break 
        region = np.array(region)+np.array(args.extern_region).tolist() 
        return pair_list_cur, pose_list_cur,  region, pair_list_before, pose_list_before, pair_list_after, pose_list_after,recon_flag

    
    def calpose(self,pair,pose,map_camera_name,calib_camera,img_w,img_h,calib_ground):
        points_cameras = []
        points_imgs = []
        points_worlds = []
        for idx, (each_pair, each_pose) in enumerate(zip(pair, pose)):
            lidar_path = each_pair['lidar']['undistort_path']            
            points = np.array(pcl.load_XYZI(lidar_path))[:, :3]
            image_to_map = np.array(each_pair['images'][map_camera_name]['pose'])
            extrinsic = np.matmul(np.linalg.pinv(image_to_map), each_pose)  # 雷达到相机外参

            #过滤地面上点
            ltog_extri = np.array(calib_ground)
            points_ground = (np.matmul(ltog_extri[:3, :3], points.T)+ltog_extri[:3, 3:]).T
            ground_flag = abs(points_ground[:, 2])<1
            points = points[ground_flag]

            points_camera = (np.matmul(extrinsic[:3, :3], points.T)+extrinsic[:3, 3:]).T
            reserve_flag = (abs(points_camera[:, 0]) < 30) & (points_camera[:, 2] > 0) & (points_camera[:, 2] < 35)  
            points_camera = points_camera[reserve_flag]
            points_img = np.matmul(
                np.array(calib_camera["new_undist_intrinsic"]), points_camera.T).T
            points_img = points_img[:, :2]/points_img[:, [2]]
            points_img = points_img.astype(np.int32)
            points_world = (np.matmul(each_pose[:3, :3], points[reserve_flag].T)+each_pose[:3, 3:]).T       
            flag_in_img = (points_img[:, 0] > 0) & (points_img[:, 1] > 0) & (
                points_img[:, 0] < img_w) & (points_img[:, 1] < img_h)
            points_camera = points_camera[flag_in_img]
            points_img = points_img[flag_in_img]
            points_world = points_world[flag_in_img]
            points_cameras.append(points_camera)
            points_imgs.append(points_img)
            points_worlds.append(points_world)
        return points_cameras,points_imgs,points_worlds
    
    def reconstruction_by_pose_color_map(self, lidar_show, out_date_path, pair_list, pose_list, map_camera_name, temp_dict,calib_camera,calib_ground):
        pool = ThreadPoolExecutor(max_workers=4)
        img_lists = []
        pair_lists = []
        pose_lists = []
        for idx, (pair, pose) in enumerate(zip(pair_list, pose_list)):
            pair = pair_list[idx]
            pose = pose_list[idx]
            pose = np.array(pose)
            if map_camera_name not in pair['images']:
                print('{} is not in pair'.format(map_camera_name))
                continue
            img = temp_dict[pair['images'][map_camera_name]['path']]

            if img.shape[0] != self.image_shape[1] or img.shape[1] != self.image_shape[0]:
                img = cv2.resize(img, self.image_shape)
            
            img_h, img_w = img.shape[:2]
            #有用数据
            img_lists.append(img)
            pair_lists.append(pair)
            pose_lists.append(pose)


        # det_time = time.time()

        th1 = pool.submit(self.det_obstacle,img_lists)
        th2 = pool.submit(self.calpose,pair_lists,pose_lists,map_camera_name,calib_camera,img_w,img_h,calib_ground)
        
    
        masks = th1.result()
        points_cameras,points_imgs,points_worlds = th2.result()
        # end_time = time.time()-det_time

        # print("end_time:",end_time)
        # print("len_img_lists:",len(img_lists))
        # masks = self.det_obstacle(img_lists)

        # end_time = time.time()-det_time

        # print("end_time:",end_time)
        # points_cameras,points_imgs,points_worlds = self.calpose(pair_lists,pose_lists,map_camera_name,calib_camera,img_w,img_h,calib_ground)


        for i in range(len(img_lists)):
            img = img_lists[i]
            mask = masks[i]
            points_camera = points_cameras[i]
            points_img = points_imgs [i]
            points_world = points_worlds[i] 

            mask_valid = mask[points_img[:, 1], points_img[:, 0], 0] != 0
            points_camera = points_camera[mask_valid]
            points_img = points_img[mask_valid]

            points_color = img[points_img[:, 1], points_img[:, 0]]
            points_world = points_world[mask_valid]
            
            cam = self.map_camera_name
            lidar_show.draw(points_world, points_color,map_camera_name,cam)

    
    def save_trans_json(self,transform_matrix,pair_list,pose_list,map_camera_name,lidar_show,out_date_path,root_path,flag):
        calib_camera = self.calib_dict[map_camera_name]
        
        world_to_img = lidar_show.world_to_img
        world_to_img = np.vstack([world_to_img, [0, 0, 0, 1]])
        img_to_world = get_img_to_world(world_to_img).tolist() #used
        
        frame_to_points = dict() #used
        world_to_cams = dict() #used
        for pi, part in enumerate([1, 3/4, 2/4, 1/4, 0]): #TODO 第一张最后一张都要取
            part_name = '{:0>6d}'.format(pi)
            idx = int(len(pair_list)*part)
            if(idx==len(pair_list)):
                idx = idx -1
            pair = pair_list[idx]
            pose = np.array(pose_list[idx])
            image_to_map = np.array(pair['images'][map_camera_name]['pose'])
            map_to_image = np.linalg.pinv(image_to_map)[0:3]
            undis_intri = np.array(calib_camera["new_undist_intrinsic"])           
            world_to_cams[part_name]= np.matmul(undis_intri,map_to_image).tolist()
            if(flag):
                frame_to_point = get_frame_to_point(world_to_img,pose) 
                frame_to_points[part_name] = frame_to_point
            img_path = os.path.join(root_path, pair['images'][map_camera_name]['path'])
            img = cv2.imread(img_path)
            if img is None:
                print('{} is not exist'.format(img_path))
                continue
            img = undistort(calib_camera, img) #读取图像和校正畸变
            ann_camera_camera = os.path.join(out_date_path, 'images', map_camera_name)
            if not os.path.exists(ann_camera_camera):
                os.makedirs(ann_camera_camera, mode=0o777, exist_ok=True)
            if img.shape[0]!=self.image_shape[1] or img.shape[1]!=self.image_shape[0]:
                img = cv2.resize(img, self.image_shape)
                cv2.imwrite(os.path.join(ann_camera_camera, part_name+'.jpg'),img)
            else:
                cv2.imwrite(os.path.join(ann_camera_camera, part_name+'.jpg'),img)
        transform_matrix['world_to_image'] = world_to_cams   
        return transform_matrix,img_to_world,frame_to_points

    def save_old_json(self,transform_matrix,pair_list,map_camera_name,lidar_show,out_date_path,root_path):
        world_to_img = lidar_show.world_to_img
        world_to_img = np.vstack([world_to_img, [0, 0, 0, 1]])
        img_to_world = get_img_to_world(world_to_img).tolist()
        world_to_egos = dict()
        frame_to_points = dict() 
        calib_camera = self.calib_dict[map_camera_name]
        for pi, part in enumerate([5/6,4/6,3/6,2/6,1/6]):
            part_name = '{:0>6d}'.format(pi)
            idx = int(len(pair_list)*part)
            pair = pair_list[idx]
            lidar_to_image = np.array(calib_camera["extrinsic"])
            image_to_map = np.array(pair['images'][map_camera_name]['pose'])
            
            transworld_to_ego = np.matmul(np.linalg.pinv(lidar_to_image),np.linalg.pinv(image_to_map))
            transego_to_world = np.linalg.pinv(transworld_to_ego)
    
            frame_to_point = get_frame_to_point(world_to_img,transego_to_world)
            world_to_egos[part_name] = transworld_to_ego.tolist()
            frame_to_points[part_name] = frame_to_point

            img_path = os.path.join(root_path, pair['images'][map_camera_name]['path'])
            img = cv2.imread(img_path)
            if img is None:
                print('{} is not exist'.format(img_path))
                continue
            img = undistort(calib_camera, img) #读取图像和校正畸变
            ann_camera_camera = os.path.join(out_date_path, 'images', map_camera_name)
            if not os.path.exists(ann_camera_camera):
                os.makedirs(ann_camera_camera, mode=0o777, exist_ok=True)
            if img.shape[0]!=self.image_shape[1] or img.shape[1]!=self.image_shape[0]:
                img = cv2.resize(img, self.image_shape)
                cv2.imwrite(os.path.join(ann_camera_camera, part_name+'.jpg'),img)
            else:
                cv2.imwrite(os.path.join(ann_camera_camera, part_name+'.jpg'),img)

        transform_matrix[map_camera_name] = calib_camera['new_lidar_to_image'].tolist()
        transform_matrix['world_to_ego'] = world_to_egos  
        transform_matrix['frame_to_point'] = frame_to_points  
        transform_matrix['img_to_world'] = img_to_world
        return transform_matrix
    
    def multi_draw(self,lidar_show,pair_all,pose_all,cam_name,root_path,calib_camera,calib_ground):
        out_date_path = self.out_root_path
        pair_list_split = split_list(pair_all,self.batch_size*2)
        pose_list_split = split_list(pose_all,self.batch_size*2)
        for idx, (pair, pose) in enumerate(zip(pair_list_split, pose_list_split)):
            temp_dict = read_images(pair,cam_name,root_path,calib_camera)
            self.reconstruction_by_pose_color_map(lidar_show, out_date_path, pair, pose, cam_name, temp_dict,calib_camera,calib_ground)

    def run_single(self, meta_json, args, gpu_id,max_num_each_gpu=6):
        gpu_number = torch.cuda.device_count()        
        assert gpu_id < gpu_number*max_num_each_gpu
        providers = [('CUDAExecutionProvider', {'device_id': gpu_id}), 'CPUExecutionProvider',]
        self.net2d_img_detect = yolov5_iflytek(confThreshold=0.4, providers=providers)
        pair_pose_list = meta_json
        calib_dict = dict()
        calib_list = []
        for i in range(len(pair_pose_list["calibration"]["extrinsics"])):
            calib_name = pair_pose_list["calibration"]["extrinsics"][i]["target"]
            if(calib_name=='gnss' or calib_name=='ground' or calib_name=='ego'):
                continue
            else:
                calib_list.append(calib_name)
        cam_list = pair_pose_list["cameras"]
        set1 = set(calib_list)
        set2 = set(cam_list)
        intersection = set1.intersection(set2)
        self.camera_list = list(intersection)
        for camera_name in self.camera_list:
            calib_dict[camera_name] = load_calibration(
                pair_pose_list, camera_name, new_image_shape=self.image_shape)
        
        calib_dict['ground'] = get_lidar_to_ground(pair_pose_list)
        self.calib_dict = calib_dict


        root_path = pair_pose_list["frames_path"]  # 主路径
        segment_name = pair_pose_list['seg_uid']

        out_date_path = self.out_root_path
        # out_date_path = os.path.join(self.out_root_path,segment_name)
        if not os.path.exists(out_date_path):
            os.makedirs(out_date_path, mode=0o775, exist_ok=True)

       
        pair_list_cur, pose_list_cur,region,pair_list_before, pose_list_before,pair_list_after, pose_list_after,recon_flag = self.odometer_by_pose(pair_pose_list["frames"], args)
        if(recon_flag==0):
            print("The current segment distance does not meet the reconstruction requirements!")
            return
        if(recon_flag==1):
            print("The current vehicle speed is too fast to meet the requirements of reconstruction!")
            return 
        start_time = time.time()
        lidar_show = LidarShowVoxel(region, args.resolution)    
        img_suffix = '{}_{}_{}'.format(segment_name,pair_list_cur[0]['lidar']['timestamp'], pair_list_cur[-1]['lidar']['timestamp'])          
        
        pair_list_all = pair_list_before + pair_list_cur + pair_list_after
        pose_list_all = pose_list_before + pose_list_cur + pose_list_after
        # 提取前一半
        pair_list_first_half = pair_list_all[:len(pair_list_all)//2]
        pair_list_second_half = pair_list_all[len(pair_list_all)//2:]
        pose_list_first_half = pose_list_all[:len(pose_list_all)//2]
        pose_list_second_half = pose_list_all[len(pose_list_all)//2:]
        # 读取图像
       
        transform_matrix = dict()
        transform_matrix["version"] = "V2"
        try:
            calib_camera = self.calib_dict[self.map_camera_name]
            calib_ground = self.calib_dict['ground']
            thread1 = threading.Thread(target=self.multi_draw,args=(lidar_show, pair_list_first_half, pose_list_first_half, self.map_camera_name,root_path,calib_camera,calib_ground))
            thread2 = threading.Thread(target=self.multi_draw,args=(lidar_show, pair_list_second_half, pose_list_second_half, self.map_camera_name,root_path,calib_camera,calib_ground))
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()
            if self.map_camera_name_add in self.camera_list:
                calib_camera_add = self.calib_dict[self.map_camera_name_add]
                thread21 = threading.Thread(target=self.multi_draw,args=(lidar_show,pair_list_first_half, pose_list_first_half, self.map_camera_name_add,root_path,calib_camera_add,calib_ground))
                thread22 = threading.Thread(target=self.multi_draw,args=(lidar_show,pair_list_second_half, pose_list_second_half, self.map_camera_name_add,root_path,calib_camera_add,calib_ground))
                thread21.start()
                thread22.start()
                thread21.join()
                thread22.join()
        except:
            print("rebuild_error!")
        try:
            for cam_name in self.camera_list:
                temp_trans = dict()
                if(cam_name==self.map_camera_name):
                    temp_trans,temp_img_to_world,temp_frame_to_points = self.save_trans_json(temp_trans,pair_list_cur, pose_list_cur,cam_name,lidar_show,out_date_path,root_path,True)
                    frame_to_points = temp_frame_to_points
                    img_to_world = temp_img_to_world
                else:
                    temp_trans,temp_img_to_world,temp_frame_to_points = self.save_trans_json(temp_trans,pair_list_cur, pose_list_cur,cam_name,lidar_show,out_date_path,root_path,False)
                transform_matrix[cam_name]=temp_trans

            transform_matrix['frame_to_point'] = frame_to_points
            transform_matrix['img_to_world'] = img_to_world
            json.dump(transform_matrix, open(os.path.join(out_date_path, 'transform_matrix.json'), 'w'))
        
            img_colormap, img_height = lidar_show.show()
            img_height = img_height[:,:,2]
            
            img_height = transnpy(img_height)
            img_height = pool_oriimg(img_height)
            
            cv2.imwrite(os.path.join(out_date_path,'{}_rgb.jpg'.format(img_suffix)), img_colormap)
            np.save(os.path.join(out_date_path, "{}_height_V2.npy".format(img_suffix)), img_height.astype(np.float16))   
            # np.save(os.path.join(out_date_path, "{}_height.npy".format(img_suffix)), img_height.astype(np.float16))
        except:
            print("save_json_error!")
        print('time:', time.time()-start_time)

    def run_mp(self, json_path_lists, args):
        num_worker = args.num_worker
        process_list = []
        for pi in range(num_worker):
            json_path_list_process = json_path_lists[pi::num_worker]
            p = Process(target=self.run_single, args=(
                json_path_list_process, args, pi))
            p.start()
            process_list.append(p)

        for i in process_list:
            p.join()

class dft_args:
    root_path = ""
    out_root_path = ""
    map_camera_name = "surround_rear_120_8M"
    map_camera_name_add = "surround_front_60_8M"
    num_worker = 4
    image_shape = (1920, 1080)
    distance = 250
    extern_distance = 150 #用于确定region
    extern_frame_distance = 175  #用于确定扩展帧
    batch_size=4
    rebuild_type = "common"
    extern_region = [-30, 30, -25, 25, -4, 2]
    resolution = (0.05, 0.1, 0.5)

    camera_list = ['surround_rear_120_8M',
                   'surround_front_120_8M', "surround_front_60_8M"]
    
# {
#     "map_camera_name": "surround_rear_120_8M",
#     "map_camera_name_add": "surround_front_120_8M",
#     "distance": 250,
#     "extern_distance": 150,
#     "extern_frame_distance": 175,
#     "resolution": [
#         0.05,
#         0.10,
#         0.5
#     ]
# }
def func_run_reconstruction_colormap(rootpath, gpu_id: int, meta_file:str, rec_cfg:dict):
    meta_fp = open(meta_file)
    meta_json = json.load(meta_fp)

    output_root = os.path.join(rootpath, "reconstruct")
    os.makedirs(output_root, mode=0o775, exist_ok=True)

    args = dft_args()
    
    args.root_path = meta_json['frames_path']
    args.out_root_path = output_root
    args.map_camera_name = rec_cfg['map_camera_name']
    args.map_camera_name_add = rec_cfg['map_camera_name_add']
    args.distance = rec_cfg['distance']
    args.extern_distance = rec_cfg['extern_distance']
    args.extern_frame_distance= rec_cfg['extern_frame_distance']
    args.resolution =  rec_cfg['resolution']

    inst = ReconstructionColormap(args)
    inst.run_single(meta_json, args, gpu_id)