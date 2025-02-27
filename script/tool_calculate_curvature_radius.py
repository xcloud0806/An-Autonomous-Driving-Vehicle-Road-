# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import math
import shutil
import numpy as np
import pandas as pd
import multiprocessing as mp
import folium
import argparse
from collections import OrderedDict
from haversine import haversine, Unit
import logging

logging.basicConfig(level=logging.DEBUG,  # 设置日志级别为DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 日志格式
logger = logging.getLogger(__file__)

FLAG = True
PI = 3.141592653589793234  # π
R_PI = PI * 3000.0 / 180.0
LA = 6378245.0  # 长半轴
OB = 0.00669342162296594323  # 扁率
MULTI_INFO_FILE = "multi_info.json"

COLORS = ['#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347','#DC143C', 
         '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', 
         '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347',
         '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347','#DC143C', 
         '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', 
         '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347','#000000'] # 121 num

# 经纬度计算功能类
def transformlat(lon, lat):
    r = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * math.sqrt(math.fabs(lon))
    r += (20.0 * math.sin(6.0 * lon * PI) + 20.0 * math.sin(2.0 * lon * PI)) * 2.0 / 3.0
    r += (20.0 * math.sin(lat * PI) + 40.0 * math.sin(lat / 3.0 * PI)) * 2.0 / 3.0
    r += (160.0 * math.sin(lat / 12.0 * PI) + 320 * math.sin(lat * PI / 30.0)) * 2.0 / 3.0
    return r

def transformlng(lon, lat):
    r = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * math.sqrt(math.fabs(lon))
    r += (20.0 * math.sin(6.0 * lon * PI) + 20.0 * math.sin(2.0 * lon * PI)) * 2.0 / 3.0
    r += (20.0 * math.sin(lon * PI) + 40.0 * math.sin(lon / 3.0 * PI)) * 2.0 / 3.0
    r += (150.0 * math.sin(lon / 12.0 * PI) + 300.0 * math.sin(lon / 30.0 * PI)) * 2.0 / 3.0
    return r

# 简单判断坐标点是否在中国
# 不在的话返回True
# 在的话返回False
def judge_China(lon, lat):
    if lon < 70 or lon > 140:
        return True
    if lat < 0 or lat > 55:
        return True
    return False

# wgs84 -> gcj02
# lng为wgs84的经度
# lat为wgs84的纬度
# 返回值为转换后坐标的列表形式，[经度, 纬度]
def wgs84_gcj02(lon_wgs84, lat_wgs84):
    if judge_China(lon_wgs84, lat_wgs84):  # 判断是否在国内
        return [None, None]
    tlat = transformlat(lon_wgs84 - 105.0, lat_wgs84 - 35.0)
    tlng = transformlng(lon_wgs84 - 105.0, lat_wgs84 - 35.0)
    rlat = lat_wgs84 / 180.0 * PI
    m = math.sin(rlat)
    m = 1 - OB * m * m
    sm = math.sqrt(m)
    tlat = (tlat * 180.0) / ((LA * (1 - OB)) / (m * sm) * PI)
    tlng = (tlng * 180.0) / (LA / sm * math.cos(rlat) * PI)
    lat_gcj02 = lat_wgs84 + tlat
    lon_gcj02 = lon_wgs84 + tlng
    return [lon_gcj02, lat_gcj02]

def visual(locs, key_loc, output_file="total.html"): 
    # start in iflytek
    # map_ = folium.Map(location=[31.82877116479812, 117.14691878548942], zoom_start=12,
    map_ = folium.Map(location=[locs[0][1], locs[0][0]], zoom_start=12,
                      tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                      attr='default')

    for i, loc in enumerate(locs):
        color = COLORS[i % len(COLORS)]
        p_wgs84_lon = loc[0]
        p_wgs84_lat = loc[1]
        p_gc102_lon, p_gc102_lat = wgs84_gcj02(p_wgs84_lon, p_wgs84_lat)
        if p_gc102_lat is None or p_gc102_lon is None:
            continue
        folium.CircleMarker(location=[p_gc102_lat, p_gc102_lon],
                            radius=4, popup='popup', color=color, fill_color=color,
                            fill=True).add_to(map_)
    
    color = COLORS[0% len(COLORS)]
    p_wgs84_lon = key_loc[1]
    p_wgs84_lat = key_loc[0]
    p_gc102_lon, p_gc102_lat = wgs84_gcj02(p_wgs84_lon, p_wgs84_lat)
    if p_gc102_lat is None or p_gc102_lon is None:
        pass
    else:
        folium.CircleMarker(location=[p_gc102_lat, p_gc102_lon],
                        radius=10, popup='key_point', color=color, fill_color=color,
                        fill=True).add_to(map_)

    map_.save(output_file)
    
def draw_map_jiekou(seg, dist, src_data, key_loc, destination_path):
    logger.info("completed seg: {}".format(seg))
    global FLAG
    locs = []
    for item in src_data:
        lat, lon = float(item["latitude"]), float(item["longitude"])
        locs.append((lon, lat))
    save_path = os.path.join(
        destination_path,
        "data",
        f"map_distThreshold_{dist_threshold}"
    )
    if FLAG:
        FLAG = False
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
    
    seg = os.path.join(save_path, seg + "_R_{:0.3f}.html".format(dist))
    logger.info("seg_path_map: {}".format(seg))
    visual(locs, key_loc, seg)
    

def calculate_distance(pa, pb):
    previous_longitude = float(pa["longitude"])
    previous_latitude = float(pa["latitude"])
    current_longitude = float(pb["longitude"])
    current_latitude = float(pb["latitude"])
    p1 = (previous_latitude, previous_longitude)
    p2 = (current_latitude, current_longitude)
    dist = haversine(p1, p2, unit=Unit.METERS)
    return dist

def to_fiflter_data(src_data, dist_threshold):
    index = 0
    len_gnss_data = len(src_data)
    result_data = [src_data[0]]
    flag = index < len_gnss_data
    while flag:
        if index + 50 >= len_gnss_data:
            flag = False
            continue
        dist = calculate_distance(src_data[index], src_data[index+50])
        if dist > dist_threshold:
            index += 50
            result_data.append(src_data[index])
        else:
            temp = 0
            while dist < dist_threshold:
                temp += 10
                if index + temp >= len_gnss_data:
                    flag = False
                    break
                dist = calculate_distance(src_data[index], src_data[index+temp])
            if flag:
                index += temp
                result_data.append(src_data[index])
    return result_data

def calculate_curvature_radius(src_data):
    curvature_radius = []
    for i in range(1, len(src_data)-1):
        lat1, lon1 = float(src_data[i-1]["latitude"]), float(src_data[i-1]["longitude"])
        lat2, lon2 = float(src_data[i]["latitude"]), float(src_data[i]["longitude"])
        lat3, lon3 = float(src_data[i+1]["latitude"]), float(src_data[i+1]["longitude"])
        p1 = (lat1, lon1)
        p2 = (lat2, lon2)
        p3 = (lat3, lon3)
        dist1 = haversine(p1, p2, unit=Unit.METERS)
        dist2 = haversine(p2, p3, unit=Unit.METERS)
        dist3 = haversine(p1, p3, unit=Unit.METERS)
        
        c = 0.5 * (dist1 + dist2 + dist3)
        area = math.sqrt(c * (c - dist1) * (c - dist2) * (c - dist3))
        # if area == 0.0: # dist1:6.5440290907293726, dist2:6.3016164610620775, dist3:12.84564555179145, 几乎是直线，所以面积是0
        #     logger.info(area)
        #     return None, None
        R = dist1 * dist2 * dist3 / (4 * (area+1e-6))
        loc = (lat2, lon2)
        curvature_radius.append([R,loc])
    curvature_radius = sorted(curvature_radius, key=lambda x:x[0])  # 从小到大排序
    # curvature = [1/item if item !=0 else -1 for item in curvature_radius ]
    # curvature  = sorted(curvature, reverse=True) # 从大到小排序
    R, loc = curvature_radius[0]
    K = 1/(R+1e-6)
    
    return K, R, loc

def cuvature_run(gnss_file, seg, destination_path, dist_threshold):
    # gnss_file = os.path.join(seg_path, date, seg, "gnss.json")
    # if gnss_file == '/data_cold2/origin_data/sihao_8j998/custom_seg/brlil_chedaoxiannanli/20231105/sihao_8j998_20231105-10-04-47_seg0/gnss.json':
    #     logger.info("-----")
    
    if not os.path.exists(gnss_file):
        logger.info("File not exists!")
        return -1, -1
    try:
        with open(gnss_file, "r", encoding="utf-8") as file:
            gnss = json.load(file)
    except Exception as e:
        logger.info("Caught an errot as Tpye of: {}".format(e))
        return -1, -1
        
    gnss_data = [gnss[key] for key in gnss.keys()]
    gnss_data = sorted(gnss_data, key=lambda x:x["utc_time"])
    fiflter_data = to_fiflter_data(gnss_data, dist_threshold)
    curvature, curvature_radius, key_loc = calculate_curvature_radius(fiflter_data)
    draw_map_jiekou(seg, curvature_radius, fiflter_data, key_loc, destination_path)

    return curvature, curvature_radius

def deal_with_one_car(annotation:str, destination_path, dataset_attribute, dist_threshold):
    car_name = annotation.split("/")[4]
    people_data = annotation.split("/")[5]
    for attribue in dataset_attribute:
        annotation_path = os.path.join(annotation, attribue)
        if not os.path.exists(annotation_path):
            continue
        for date in os.listdir(annotation_path):
            data_dict = OrderedDict()
            data_dict["seg_name"] = []
            data_dict["dataset"] = []
            data_dict["curvature"] = []
            data_dict["curvature_radius"] = []
            data_dict["resconstruct-error"] = []
            anno_seg_path = os.path.join(annotation_path, date)
            seg_list = sorted(os.listdir(anno_seg_path))
            for seg in seg_list:
                annotation_file = os.path.join(anno_seg_path, seg, "annotation.json")
                if os.path.exists(annotation_file):
                    with open(annotation_file, "r", encoding="utf-8") as file:
                        anno = json.load(file)
                    reconstruction_flag = anno["lane"]["image_type"]["reconstruction-error"]
                    gnss_file = os.path.join(
                        "/data_cold2/origin_data/",
                        car_name,
                        "custom_seg"
                        ,people_data, 
                        date, 
                        seg, 
                        "gnss.json"
                    )
                    if not os.path.exists(gnss_file):
                        logger.info(FileExistsError)
                        logger.info(gnss_file)
                        return
                    if seg == "sihao_0fx60_20240128-10-07-42_seg0":
                        logger.info("debug")
                    curvature, curvature_radius = cuvature_run(gnss_file, seg, destination_path, dist_threshold)
                    data_dict["seg_name"].append(seg)
                    data_dict["dataset"].append(attribue)
                    data_dict["resconstruct-error"].append(reconstruction_flag)
                    data_dict["curvature"].append(curvature)
                    data_dict["curvature_radius"].append(curvature_radius)
            # statistics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data",f"curvature_statistics_distThreshold_{dist_threshold}")
            statistics_path = os.path.join(
                destination_path,
                "data",
                f"curvature_statistics_distThreshold_{dist_threshold}"
            )
            if not os.path.exists(statistics_path):
                os.makedirs(statistics_path)
            target_file = os.path.join(statistics_path, car_name + "_" + date + "_curvature.csv")
            # with open(target_file, "w", newline="") as file:
            #     writer = csv.writer(file)
            #     writer.writerows(data_dict)
            dframe = pd.DataFrame(data_dict)
            pd.set_option('display.max_colwidth', 25)
            dframe.to_csv(target_file, index=False, encoding='utf8')
            logger.info("===========================")
            logger.info("saved: {}".format(target_file))
            
        #     break
        # break
                
# src:      
# "/data_autodrive/auto/custom/sihao_8j998/brlil_chedaoxiannanli/clip_submit"
# "/data_autodrive/auto/custom/sihao_0fx60/frwang_dawandao/clip_submit/"

def select_suspicious_radius(destination_path, radius):
    data_path = os.path.join(destination_path, "data")
    map_paths = []
    for item in os.listdir(data_path):
        if item.startswith("map"):
            map_paths.append(os.path.join(data_path, item))
    for path in map_paths:
        for item in os.listdir(path):
            if not item.endswith(".html"):
                continue
            try :
                R = float(os.path.splitext(item)[0].split("_")[-1])
            except:
                logger.info(os.path.join(path, item))
            
            if R < radius:
                src_path = os.path.join(path, item)
                dst_path = os.path.join(path, f"AAA_threshold_{radius}suspicious_R")
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)

                shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""计算已标注的弯道数据的统计信息,输出为 :
                      1、包括seg_name, 数据集属性、曲率、曲率半径、reconstruct-erros,
                         5个字段的统计表格
                      2、以每个段名可视化的弯道数据,特别标记了最急的转弯点
                      3、可视化数据中的子文件夹,是从所有的可视化数据中拷贝了转弯半径小的可疑数据"""
    )
    parser.add_argument(
        '-s',
        "--source_path",
        type=str, 
        required=True,
        help="例如:/data_autodrive/auto/custom/sihao_8j998/brlil_chedaoxiannanli/clip_submit"
    )
    parser.add_argument(
        '-d',
        "--destination_path",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="输出结果保存路径,默认是当前路径"
    )
    parser.add_argument(
        '-t',
        "--threshold",
        type=int,
        default=10,
        help="采样点的最小间距,经测试设置10或者12时,计算结果比较合理"
    )
    parser.add_argument(
        '-r',
        "--radius",
        type=int,
        default=15,
        help="设置转弯半径的阈值,筛选出最小弯道半径小于该值的所有段数据"
    )
    args = parser.parse_args()  # 解析命令行参数
    dataset = ["annotation_train", "annotation_test"]
    start = time.time()
    logger.info("Input data:")
    logger.info("source_path:{}".format(args.source_path))
    logger.info("destination_path: {}".format(args.destination_path))
    logger.info("dist_threshold: {}".format(args.threshold))
    logger.info("radius: {}".format(args.radius))
    logger.info("start to calculate...")
    time.sleep(10)
    source_path = args.source_path
    destination_path = args.destination_path
    dist_threshold = args.threshold
    radius = args.radius
    deal_with_one_car(source_path, destination_path, dataset, dist_threshold)
    select_suspicious_radius(destination_path, radius)
    end = time.time()
    logger.info("time cost {}s".format(end - start))

    
    