import os
import json
import numpy as np
import pandas as pd
import cv2
import math
from tqdm import tqdm
from geopy.distance import geodesic

def collect_gnss_info(gnss_path):
    collected_gnss_info = list()
    gnss_info = pd.read_csv(gnss_path, keep_default_na=False, low_memory=False)
    #'LocalTime', 'UtcTime'
    ts, xs, ys = (
        gnss_info["utc_time"].values,
        gnss_info["latitude"].values,
        gnss_info["longitude"].values,
    )
    for t,x,y in zip(ts,xs,ys):
        if (t != "na") & (x != "na") & (y != "na"):
            collected_gnss_info.append([float(t),float(x),float(y)])
    collected_gnss_info = np.vstack(collected_gnss_info)
    return collected_gnss_info

def get_world_to_img(region, world_resolution):
    gnss_ref_eta = 1e-6
    gnss_ref = [31.70000000, 117.30000000]
    gnss_ref_x = [31.70000000+gnss_ref_eta, 117.30000000]
    gnss_ref_y = [31.70000000, 117.30000000+gnss_ref_eta]
    distance_x = geodesic(gnss_ref, gnss_ref_x).m
    distance_y = geodesic(gnss_ref, gnss_ref_y).m
    gnss_resolution = (world_resolution[0]/distance_x*gnss_ref_eta, world_resolution[1]/distance_y*gnss_ref_eta)
    
    gnss_to_img = np.array([[1/gnss_resolution[0], 0, -region[0,0]/gnss_resolution[0]],
                            [0, 1/gnss_resolution[1], -region[0,1]/gnss_resolution[1]],
                            [0,0,1]])

    image_size = np.matmul(gnss_to_img[:2,:2], region[1]) + gnss_to_img[:2,2]
    image_size = [math.ceil(size) for size in image_size]
    return gnss_to_img, image_size

def show_gnss(collected_gnss_info, world_resolution=(10,10), img_name='img_gnss.jpg'):
    region = np.array([collected_gnss_info[:,1:].min(axis=0), 
                       collected_gnss_info[:,1:].max(axis=0)])
    # 扩充范围
    region[0] -= 2e-3 # 1e-3 约等于100m
    region[1] += 2e-3
    gnss_to_img, image_size = get_world_to_img(region, world_resolution)
    img_gnss = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
    for gnss in collected_gnss_info:
        img_point = np.matmul(gnss_to_img[:2,:2], np.array(gnss[1:])) + gnss_to_img[:2,2]
        cv2.circle(img_gnss, (round(img_point[0]), round(img_point[1])), 1, (255), -1)
    # cv2.imwrite('img_gnss.jpg', img_gnss) 
    
    kernel = np.ones((7,7),np.uint8)
    dilation = cv2.dilate(img_gnss, kernel, iterations=8)
    # cv2.imwrite('dilation.jpg', dilation)
    
    ret, binary = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    assert len(contours)==1
    cv2.drawContours(dilation, contours, -1, (128), 1)
    cv2.imwrite(img_name, dilation)
    
    contours = contours[0].reshape(len(contours[0]),2)
    return gnss_to_img, region, image_size, contours

root_path = '/data_cold2/origin_data/chery_53054/custom_frame/hbfan_wuhu_testset'
date_list = ['20231223',]
road_name = 'wuhu_test_cityway'
highway_list = [
    "20231223-08-10-08",     
    "20231223-08-27-36",    
    "20231223-08-49-15",   
    "20231223-09-08-20",
    "20231223-13-44-58",     
    "20231223-14-06-00",     
    "20231223-14-26-12"       
]
cityway_list = [
    "20231223-14-44-52",     
    "20231223-15-19-08",     
    "20231223-16-13-24",     
    "20231223-16-44-42",     
    "20231223-18-02-07",     
    "20231223-18-51-24",     
    "20231223-19-42-38",     
    "20231223-09-38-44",     
    "20231223-10-04-41",     
    "20231223-10-26-24",     
    "20231223-10-53-48",     
    "20231223-11-29-46"     
]
test_gnss_info = dict()
for date_name in tqdm(date_list):
    gnss_info_list = list()
    date_path = os.path.join(root_path, date_name)
    segment_list = os.listdir(date_path)
    for si, segment_name in enumerate(segment_list):
        if segment_name not in cityway_list:
            continue
        gnss_path = os.path.join(date_path, segment_name, 'gnss.csv')
        gnss_info = collect_gnss_info(gnss_path)
        gnss_info_list.append(gnss_info)
    gnss_info = np.concatenate(gnss_info_list, axis=0)
    gnss_to_img, region, image_size, contours = show_gnss(gnss_info, world_resolution=(20,20), img_name='{}.jpg'.format(road_name))
    test_gnss_info[road_name] = {
                                    'gnss_to_img': gnss_to_img.tolist(),
                                    'region': region.tolist(),
                                    'image_size': image_size,
                                    'contours': contours.tolist(),
                                }
    
    img = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
    cv2.fillPoly(img, [np.array(contours)], (255))
    cv2.imwrite('{}_fill.jpg'.format(road_name), img)
    
json.dump(test_gnss_info, open('test_gnss_info.json','w'))
