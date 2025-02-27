import os, sys
import json
from shapely.geometry import Point, box
# from shapely.geometry.box import box
from shapely.geometry.polygon import Polygon
import folium

import numpy as np
import csv
from loguru import logger
import math

pi = 3.141592653589793234  # π
r_pi = pi * 3000.0 / 180.0
la = 6378245.0  # 长半轴
ob = 0.00669342162296594323  # 扁率

# 经纬度计算功能类
def transformlat(lon, lat):
    r = (
        -100.0
        + 2.0 * lon
        + 3.0 * lat
        + 0.2 * lat * lat
        + 0.1 * lon * lat
        + 0.2 * math.sqrt(math.fabs(lon))
    )
    r += (20.0 * math.sin(6.0 * lon * pi) + 20.0 * math.sin(2.0 * lon * pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(lat * pi) + 40.0 * math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    r += (
        (160.0 * math.sin(lat / 12.0 * pi) + 320 * math.sin(lat * pi / 30.0))
        * 2.0
        / 3.0
    )
    return r


def transformlng(lon, lat):
    r = (
        300.0
        + lon
        + 2.0 * lat
        + 0.1 * lon * lon
        + 0.1 * lon * lat
        + 0.1 * math.sqrt(math.fabs(lon))
    )
    r += (20.0 * math.sin(6.0 * lon * pi) + 20.0 * math.sin(2.0 * lon * pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(lon * pi) + 40.0 * math.sin(lon / 3.0 * pi)) * 2.0 / 3.0
    r += (
        (150.0 * math.sin(lon / 12.0 * pi) + 300.0 * math.sin(lon / 30.0 * pi))
        * 2.0
        / 3.0
    )
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
    rlat = lat_wgs84 / 180.0 * pi
    m = math.sin(rlat)
    m = 1 - ob * m * m
    sm = math.sqrt(m)
    tlat = (tlat * 180.0) / ((la * (1 - ob)) / (m * sm) * pi)
    tlng = (tlng * 180.0) / (la / sm * math.cos(rlat) * pi)
    lat_gcj02 = lat_wgs84 + tlat
    lon_gcj02 = lon_wgs84 + tlng
    return [lon_gcj02, lat_gcj02]


def decode_gnss(gnss_file):
    ret = {}
    with open(gnss_file) as fp:
        rd = csv.reader(fp)
        header = rd.__next__()
        time_idx = 0 if "utc_time" in header[0] else None
        longi_idx = 3 if "longitude" in header[3] else None
        lati_idx = 4 if "latitude" in header[4] else None

        for msg in rd:
            if msg[time_idx] == "na" or msg[longi_idx] == "na" or msg[lati_idx] == "na":
                continue
            time_key = int(float(msg[time_idx]) * 1000)
            longi_val = float(msg[longi_idx])
            lati_val = float(msg[lati_idx])
            if judge_China(longi_val, lati_val):
                continue
            ret[time_key] = [longi_val, lati_val]

    return ret


def gen_region_by_clips(clip_list, data_root):
    region_pts = []
    region_gcj_pts = []

    for clip in clip_list:
        clip_path = os.path.join(data_root, clip)
        gnss_csv = os.path.join(clip_path, "gnss.csv")
        if not os.path.exists(gnss_csv):
            logger.error("gnss.csv not found in {}".format(clip_path))
            continue
        gnss_info = decode_gnss(gnss_csv)
        tss = list(gnss_info.keys())
        _pts = []
        _gcj_pts = []
        for i, ts in enumerate(tss):
            # 每隔5秒取一个点，绘制GNSS轨迹,GNSS 100HZ
            if i % 500 != 0:
                continue

            if tss[i] not in gnss_info:
                continue
            lon_wgs84, lat_wgs84 = gnss_info[tss[i]]
            _pts.append([lat_wgs84, lon_wgs84])
            lon_gcj02, lat_gcj02 = wgs84_gcj02(lon_wgs84, lat_wgs84)
            _gcj_pts.append([lat_gcj02, lon_gcj02])

        region_pts.extend(_pts)
        region_gcj_pts.extend(_gcj_pts)

    return region_pts, region_gcj_pts


# def expand_regions(region_pts, expand_ratio=1.2):
#     region_poly = Polygon(region_pts)
#     region_poly = region_poly.buffer(expand_ratio * region_poly.area ** 0.5)
#     return region_poly.exterior.coords

def expand_bbox(region_pts, expand_value=0.01):
    region_poly = Polygon(pts)
    region_bbox = list(region_poly.bounds)
    min_lat = region_bbox[0] - expand_value
    max_lat = region_bbox[2] + expand_value
    min_lon = region_bbox[1] - expand_value
    max_lon = region_bbox[3] + expand_value
    # print(region_bbox)
    bbox_gcj02 = []
    bbox_wgs84 = [ # [lon, lat]
        [min_lon, min_lat],
        [min_lon, max_lat],
        [max_lon, max_lat],
        [max_lon, min_lat]
    ]
    print(bbox_wgs84)
    for i in range(4):
        lon_wgs84, lat_wgs84 = bbox_wgs84[i]
        lon_gcj02, lat_gcj02 = wgs84_gcj02(lon_wgs84, lat_wgs84)
        bbox_gcj02.append([lat_gcj02, lon_gcj02])
    return bbox_gcj02

if __name__ == "__main__":
    data_root = "/data_cold2/origin_data/sihao_1482/custom_frame/frwang_lukou_huainan_test_set/20240618_gnss/"
    test_region_key = "huainan_city"
    clip_list = os.listdir(data_root)

    visual_map = folium.Map(location=[31.82877116479812, 117.14691878548942], zoom_start=8,
                        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                        attr='default')

    pts, gcj_pts = gen_region_by_clips(clip_list, data_root)
    folium.PolyLine(
        gcj_pts,
        color="#FFFF00",
        weight=3,
        opacity=0.5,
        tooltip="Raw_{}".format(test_region_key)
    ).add_to(visual_map)
    
    expand_bbox_gcj02 = list(expand_bbox(pts, 0.001))
    folium.Polygon(
        expand_bbox_gcj02,
        color="#00FF00",
        weight=3,
        opacity=0.5,
        tooltip="Expand_{}".format(test_region_key)
    ).add_to(visual_map)

    visual_map.save("./output.html")
