import folium
import json
import os, sys
# from utils.gnss_coord_op import wgs84_gcj02
import numpy as np
import math

pi = 3.141592653589793234  # π
r_pi = pi * 3000.0 / 180.0
la = 6378245.0  # 长半轴
ob = 0.00669342162296594323  # 扁率
multi_info_file = "multi_info.json"

colors = ['#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
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
    r += (20.0 * math.sin(6.0 * lon * pi) + 20.0 * math.sin(2.0 * lon * pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(lat * pi) + 40.0 * math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    r += (160.0 * math.sin(lat / 12.0 * pi) + 320 * math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return r

def transformlng(lon, lat):
    r = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * math.sqrt(math.fabs(lon))
    r += (20.0 * math.sin(6.0 * lon * pi) + 20.0 * math.sin(2.0 * lon * pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(lon * pi) + 40.0 * math.sin(lon / 3.0 * pi)) * 2.0 / 3.0
    r += (150.0 * math.sin(lon / 12.0 * pi) + 300.0 * math.sin(lon / 30.0 * pi)) * 2.0 / 3.0
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

# draw wgs84 pts in map
def visual(locs, output_file="total.html"): 
    # start in iflytek
    map_ = folium.Map(location=[31.82877116479812, 117.14691878548942], zoom_start=12,
                      tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                      attr='default')

    for i, loc in enumerate(locs):
        color = colors[i % len(colors)]
        p_wgs84_lon = loc[0]
        p_wgs84_lat = loc[1]
        p_gc102_lon, p_gc102_lat = wgs84_gcj02(p_wgs84_lon, p_wgs84_lat)
        if p_gc102_lat is None or p_gc102_lon is None:
            continue
        folium.CircleMarker(location=[p_gc102_lat, p_gc102_lon],
                            radius=4, popup='popup', color=color, fill_color=color,
                            fill=True).add_to(map_)
    map_.save(output_file)

def main(multi_path):
    multi_file = os.path.join(multi_path, multi_info_file)
    if not os.path.exists(multi_file):
        return 

    def func_handle_clip(clip_path):
        gnss_file = os.path.join(clip_path, "gnss.json")
        if not os.path.exists(gnss_file):
            return None

        locations = []
        gnss = json.load(open(gnss_file, "r"))
        gnss_tss = list(gnss.keys())
        for i, ts in enumerate(gnss_tss):
            # get location every 2 second
            if i % 200 != 0:
                continue

            locations.append(
                [float(gnss[ts]["longitude"]), float(gnss[ts]["latitude"])]
            )
        return locations

    multi_info = json.load(open(multi_file, "r"))
    coll_key = list(multi_info.keys())[0]
    clips = multi_info[coll_key]['clips_path']
    locations = []
    for clip in clips:
        _tmp_locations = func_handle_clip(clip)
        locations.extend(_tmp_locations)
    
    output_html = os.path.join(multi_path, "location.html")
    visual(locations, output_html)


if __name__ == "__main__":
    multi_path = "/data_cold2/origin_data/sihao_19cp2/custom_coll/frwang_chengshilukou/20240305/20240305-10-34-52-R9VW8SOI"
    if len(sys.argv) > 1:
        multi_path = sys.argv[1]

    if os.path.exists(os.path.join(multi_path, multi_info_file)):
        main(multi_path)
    else:
        colls = os.listdir(multi_path)
        colls.sort()
        for coll in colls:
            coll_path = os.path.join(multi_path, coll)
            main(coll_path)
