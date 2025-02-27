import os, sys
import argparse
import csv
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import math
from sklearn import metrics
import folium
from scipy.spatial.distance import pdist, squareform
from overpy import Overpass
import json
from requests import Session

colors = [
    "#DC143C",
    "#FFB6C1",
    "#DB7093",
    "#C71585",
    "#8B008B",
    "#4B0082",
    "#7B68EE",
    "#0000FF",
    "#B0C4DE",
    "#708090",
    "#00BFFF",
    "#5F9EA0",
    "#00FFFF",
    "#7FFFAA",
    "#008000",
    "#FFFF00",
    "#808000",
    "#FFD700",
    "#FFA500",
    "#FF6347",
    "#DC143C",
    "#FFB6C1",
    "#DB7093",
    "#C71585",
    "#8B008B",
    "#4B0082",
    "#7B68EE",
    "#0000FF",
    "#B0C4DE",
    "#708090",
    "#00BFFF",
    "#5F9EA0",
    "#00FFFF",
    "#7FFFAA",
    "#008000",
    "#FFFF00",
    "#808000",
    "#FFD700",
    "#FFA500",
    "#FF6347",
    "#DC143C",
    "#FFB6C1",
    "#DB7093",
    "#C71585",
    "#8B008B",
    "#4B0082",
    "#7B68EE",
    "#0000FF",
    "#B0C4DE",
    "#708090",
    "#00BFFF",
    "#5F9EA0",
    "#00FFFF",
    "#7FFFAA",
    "#008000",
    "#FFFF00",
    "#808000",
    "#FFD700",
    "#FFA500",
    "#FF6347",
    "#DC143C",
    "#FFB6C1",
    "#DB7093",
    "#C71585",
    "#8B008B",
    "#4B0082",
    "#7B68EE",
    "#0000FF",
    "#B0C4DE",
    "#708090",
    "#00BFFF",
    "#5F9EA0",
    "#00FFFF",
    "#7FFFAA",
    "#008000",
    "#FFFF00",
    "#808000",
    "#FFD700",
    "#FFA500",
    "#FF6347",
    "#DC143C",
    "#FFB6C1",
    "#DB7093",
    "#C71585",
    "#8B008B",
    "#4B0082",
    "#7B68EE",
    "#0000FF",
    "#B0C4DE",
    "#708090",
    "#00BFFF",
    "#5F9EA0",
    "#00FFFF",
    "#7FFFAA",
    "#008000",
    "#FFFF00",
    "#808000",
    "#FFD700",
    "#FFA500",
    "#FF6347",
    "#DC143C",
    "#FFB6C1",
    "#DB7093",
    "#C71585",
    "#8B008B",
    "#4B0082",
    "#7B68EE",
    "#0000FF",
    "#B0C4DE",
    "#708090",
    "#00BFFF",
    "#5F9EA0",
    "#00FFFF",
    "#7FFFAA",
    "#008000",
    "#FFFF00",
    "#808000",
    "#FFD700",
    "#FFA500",
    "#FF6347",
    "#000000",
]  # 121 num

pi = 3.141592653589793234  # π
r_pi = pi * 3000.0 / 180.0
la = 6378245.0  # 长半轴
ob = 0.00669342162296594323  # 扁率

def parse_args():
    parser = argparse.ArgumentParser(description="Init work node")
    # clip decoder params
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/data_autodrive/自动驾驶/hangbai/Traffic_Light/annotations",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--ext",
        "-j",
        type=str,
        default="json",
    )
    args = parser.parse_args()
    return args


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
# lon为wgs84的经度
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

class IflyAutoRegeo:
    """
    通过智能汽车的事业部封装的接口进行地理信息逆向解析，返回高德地图逆向地址信息
    1. 获取高德地图逆向地址信息
    2. 获取高德地图道路信息
    3. 获取高德地图道路路口信息
    """

    def __init__(self, search_radius=100):
        self.openid = "iflyznjs"
        self.search_radius = search_radius
        self.url = "https://apisix-pre-in.iflytekauto.cn/ocp-common/v1/map/location/regeo"
        self.headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "AutoDrive/1.0.0",
        }
        self.did = "rdg_cv2"
        self.sess = Session()
        self.sess.headers.update(self.headers)

    def __call__(self, pt:list):
        if len(pt) != 2:
            return None
        
        wgs_lat = pt[0]
        wgs_lon = pt[1]
        gcj_lon, gcj_lat = wgs84_gcj02(wgs_lon, wgs_lat)
        if judge_China(gcj_lon, gcj_lat):  # 判断是否在国内
            return None
        
        query = {
            "longitude": gcj_lon,
            "latitude": gcj_lat,
            "radius": self.search_radius,
            "openId": self.openid,
            "dId": self.did,
            "extensions": "all",
            "roadlevel": 0
        }

        response = self.sess.get(self.url, params=query)
        status = response.status_code
        if status == 200:
            data = response.json()
            if data['code'] == 0:
                res = data['data']
                return res
        
        return None

def get_ways_name(wgs84_lon_lst:list, wgs84_lat_lst:list):
    wgs84_lon_arr = np.array(wgs84_lon_lst)
    wgs84_lat_arr = np.array(wgs84_lat_lst)

    wgs84_lst = [np.min(wgs84_lat_lst), np.min(wgs84_lon_lst), np.max(wgs84_lat_lst), np.max(wgs84_lon_lst)]
    lat_0 = wgs84_lst[0]
    lon_0 = wgs84_lst[1]
    lat_1 = wgs84_lst[2]
    lon_1 = wgs84_lst[3]
    query_str = "[out:json][timeout:30];way[highway]({},{},{},{});(._;>;);out;".format(lat_0, lon_0, lat_1, lon_1)

    api = Overpass()
    result = api.query(query_str)
    ret = {}
    for way in result.ways:
        if way.tags.get('highway'):
            name = way.tags.get('name:en')
            highway = way.tags.get('highway')
            if name not in ret:
                ret[name] = {
                    "highway": highway,
                    "name:en": name,
                    "name:zh": way.tags.get('name'),
                    "infos": way,
                    "count": 1
                }
            else:
                ret[name]['count'] += 1
    return ret

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


def add_test_region(visual_map):
    hefei_xunfei_lukou_region = [
        [ 31.844300409999995, 117.10376349,],
        [ 31.844300409999995, 117.18323005109997,],
        [ 31.81946599029999, 117.18323005109997,],
        [ 31.81946599029999, 117.10376349,],
        [ 31.844300409999995, 117.10376349,],
    ]

    wuhu_chery_lukou_region = [
        [
            31.46002497060376,
            118.35641914505207,
        ],
        [
            31.476279672235215,
            118.36732885412142,
        ],
        [
            31.475772058488097,
            118.38981702511694,
        ],
        [
            31.47046960093194,
            118.40864767957389,
        ],
        [
            31.46297035999998,
            118.4105233376,
        ],
        [
            31.39821327209999,
            118.4105233376,
        ],
        [
            31.399135246724644,
            118.3578712864522,
        ],
        [
            31.46002497060376,
            118.35641914505207,
        ],
    ]

    visual_map = folium.Polygon(
        hefei_xunfei_lukou_region,
        color="#FF0000",
        weight=5,
        popup="合肥讯飞周边路口",
    ).add_to(visual_map)
    return visual_map


def visual_csv(csv_file, clip_id, visual_map=None, color='#00FF00'):
    """
    visualize csv file
    """
    gnss_info = decode_gnss(csv_file)
    # clip_id = os.path.dirname(csv_file).split('/')[-1]
    if gnss_info is None:
        return

    tss = list(gnss_info.keys())
    pts = []
    for i, ts in enumerate(tss):
        # 每隔0.5秒取一个点，绘制GNSS轨迹,GNSS 100HZ
        if i % 50 != 0:
            continue        
        
        if tss[i] not in gnss_info:
            continue
        lon_wgs84, lat_wgs84 = gnss_info[tss[i]]
        lon_gcj02, lat_gcj02 = wgs84_gcj02(lon_wgs84, lat_wgs84)
        pts.append([lat_gcj02, lon_gcj02])

    if visual_map is None:
        visual_map = folium.Map(location=pts[0], zoom_start=8,
                        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                        attr='default')
    
    folium.PolyLine(
        pts,
        color=color,
        weight=3,
        opacity=0.5,
        tooltip="{}".format(clip_id)
    ).add_to(visual_map)
    # visual_map.save(output_file)
    return visual_map

def visual_json(json_file, seg_id, visual_map=None, color='#00FF00'):
    if not os.path.exists(json_file):
        return

    pts = []
    with open(json_file) as fp:
        gnss_info = json.load(fp)
        prev_ts = None
        for k, v in gnss_info.items():
            time_key = int(k)
            if prev_ts is None:
                prev_ts = time_key
            else:
                if time_key - prev_ts < 500:
                    continue
            prev_ts = time_key
            long_val = float(v['longitude'])
            lat_val = float(v['latitude'])
            if judge_China(long_val, lat_val):
                continue
            lon_gcj02, lat_gcj02 = wgs84_gcj02(long_val, lat_val)
            pts.append([lat_gcj02, lon_gcj02])
    
    if visual_map is None:
        visual_map = folium.Map(location=pts[0], zoom_start=8,
                        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                        attr='default')
    
    folium.PolyLine(
        pts,
        color=color,
        weight=3,
        opacity=0.5,
        tooltip="{}".format(seg_id)
    ).add_to(visual_map)
    return visual_map

def visual_clip_dir(clip_dir, output_file, ext):
    root = clip_dir
    clips = os.listdir(root)
    files = []
    for clip in clips:
        if not os.path.isdir(os.path.join(root, clip)):
            continue 
        if ext == 'json':
            json_file = os.path.join(root, clip, "gnss.json")
            if os.path.exists(json_file):
                files.append((json_file, clip))
                print("visual json file: {}".format(json_file))
        else:                            
            csv_file = os.path.join(root, clip, "gnss.csv")
            if os.path.exists(csv_file):
                files.append((csv_file, clip))
                print("visual csv file: {}".format(csv_file))
    visual_map = None
    for i, info in enumerate(files):
        if ext == 'json':
            f, clip_id = info
            visual_map = visual_json(f, clip_id, visual_map, colors[i % 100])
        else:
            visual_map = visual_csv(f, clip_id, visual_map, colors[i % 100])
    visual_map = add_test_region(visual_map)
    visual_map.save(output_file)

def visual(pts, visual_map=None):
    """
    创建或更新一张地图，用折线图可视化给定的路径点。

    参数:
    - pts: 路径点的列表，每个点是一个经纬度对,为gcj02坐标系下的[lat,lon]列表。
    - visual_map: 用于可视化地图的对象。如果为None,则创建一张新地图。

    返回:
    - visual_map: 更新后的地图对象。
    """
    # 如果没有提供visual_map，则创建一张新地图
    if visual_map is None:
        # 初始化地图，使用提供的路径点作为初始位置
        visual_map = folium.Map(location=pts[0], zoom_start=8,
                        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                        attr='default')
    
    # 在地图上添加折线图层，用以显示路径点
    folium.PolyLine(
        pts,
        color=colors[1],  # 使用预定义的颜色
        weight=3,  # 线条宽度
        opacity=0.5,  # 线条透明度
    ).add_to(visual_map)
    
    # 返回更新后的地图对象
    return visual_map

def main():
    seg_roots = "/data_cold2/ripples/"
    cars = [
        "sihao_1482",
        "sihao_y7862",
        "sihao_7xx65",
        "sihao_27en6",
        "sihao_47466"
    ]
    for car in cars:
        seg_roots_ = os.path.join(seg_roots, car, "custom_seg/frwang_zhixinglukou")
        if not os.path.exists(seg_roots_):
            print("seg root not exist: {}".format(seg_roots_))
            continue
        dates = os.listdir(seg_roots_)
        for date in dates:
            if not os.path.isdir(os.path.join(seg_roots_, date)):
                continue

            seg_root = os.path.join(seg_roots_, date)
            output_file = f"hefei_{car}_{date}.html"
            visual_clip_dir(seg_root, output_file, "json")


def main_with_args():
    args = parse_args()
    query = args.input
    name = os.path.basename(query)
    output_file=f"{args.output}_{name}.html"
    if os.path.isfile(query):
        print("visual csv file: {}".format(query))
        if args.ext == 'json':
            visual_map = visual_json(query, name)
        else:
            visual_map = visual_csv(query)
        visual_map.save(output_file)
    else:
        if os.path.isdir(query):
            visual_clip_dir(query, output_file, args.ext)
        else:
            print("csv/json file or dir not exist: {}".format(query))
    # visual_csv(csv_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        main()
    else:
        main_with_args()
