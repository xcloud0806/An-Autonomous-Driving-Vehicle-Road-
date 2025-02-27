import requests
import os
import json
from requests import Session
import time
import math

ABK_DB_IP="10.103.136.26"
ABK_PORT="8084"
DB_IP="172.30.12.140"
PORT="8191"
SAVE_URL="http://{}:{}/api/dataset/v1/segData/save".format(DB_IP, PORT)
QUERY_URL="http://{}:{}/api/dataset/v1/segData/query".format(DB_IP, PORT)
UPDATE_URL="http://{}:{}/api/dataset/v1/segData/update".format(DB_IP, PORT)

HTTP_HEADER = {
    "User-Agent": "AutoDrive/1.0.0",
    "Content-Type": "application/json; charset=utf-8",
}

# 设置常量
pi = 3.141592653589793234  # π
r_pi = pi * 3000.0 / 180.0
la = 6378245.0  # 长半轴
ob = 0.00669342162296594323  # 扁率


def judge_China(lon, lat):
    if lon < 70 or lon > 140:
        return True
    if lat < 0 or lat > 55:
        return True
    return False


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


# wgs84 -> gcj02
# lng为wgs84的经度
# lat为wgs84的纬度
# 返回值为转换后坐标的列表形式，[经度, 纬度]
def wgs84_gcj02(lon_wgs84, lat_wgs84):
    if judge_China(lon_wgs84, lat_wgs84):  # 判断是否在国内
        return [lon_wgs84, lat_wgs84]
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


gnss_submit_keys = [
    "utc_time",
    "longitude",
    "latitude",
    "altitude",
    "speed",
    "pitch",
    "roll",
    "yaw",
    "gyrox",
    "gyroy",
    "gyroz",
]


def gen_gnss_info(seg_path, meta):
    gnss_list = []
    gcj02_list = []
    gnss_file = os.path.join(seg_path, "gnss.json")
    gnss_json = open(gnss_file)
    gnss = json.load(gnss_json)

    frames = meta["frames"]
    for i, f in enumerate(frames):
        if i % 10 != 0:
            continue

        gnss_ts = str(f["gnss"])
        gnss_info = gnss[gnss_ts]
        lat = float(gnss_info["latitude"])
        lon = float(gnss_info["longitude"])
        gc102 = wgs84_gcj02(lon, lat)
        gcj02_list.append(
            {"utcTime": gnss_ts, "latitude": gc102[1], "longitude": gc102[0]}
        )
        gnss_submit = {}
        for k in gnss_submit_keys:
            gnss_submit[k] = gnss_info[k]
        gnss_list.append(gnss_submit)
    return gnss_list, gcj02_list


def try_add_seg_payload(payload_json):
    try_cnt = 0
    while True:
        try:
            session = Session()
            session.headers.update(HTTP_HEADER)
            res = session.post(SAVE_URL, data=payload_json)
            if res.status_code == 200:
                response = res.text
                rc = json.loads(response)
                # rc_status = rc["rc"]
                rc_status = rc['code']
                return rc_status, response
            else:
                print(
                    f"Try add {payload_json}, but return {res.status_code} - {res.text}"
                )
                if try_cnt < 10:
                    time.sleep(60)
                    try_cnt += 1
                    continue
                else:
                    return 99, ""
        except ConnectionError as e:
            print(f"\tExecption [{e}] occered!")
            if try_cnt < 10:
                time.sleep(60)
                try_cnt += 1
                continue
            else:
                return 99, ""


def try_update_seg_payload(payload_json):
    try_cnt = 0
    while True:
        try:
            session = Session()
            session.headers.update(HTTP_HEADER)
            res = session.post(UPDATE_URL, data=payload_json)
            if res.status_code == 200:
                response = res.text
                rc = json.loads(response)
                # rc_status = rc["rc"]
                rc_status = rc['code']
                return rc_status, response
            else:
                print(
                    f"Try add {payload_json}, but return {res.status_code} - {res.text}"
                )
                if try_cnt < 10:
                    time.sleep(60)
                    try_cnt += 1
                    continue
                else:
                    return 99, ""
        except ConnectionError as e:
            print(f"\tExecption [{e}] occered!")
            if try_cnt < 10:
                time.sleep(60)
                try_cnt += 1
                continue
            else:
                return 99, ""


def db_add_seg(seg_path, lane3d_anno_path, obstacle_anno_path):
    meta_file = os.path.join(seg_path, "meta.json")
    meta_json = open(meta_file)
    meta = json.load(meta_json)

    calib = meta["calibration"]
    calib_date = calib["date"]
    car = meta["car"]

    payload = {}

    payload["id"] = meta["seg_uid"]
    payload["segPath"] = seg_path
    payload["metaFilePath"] = meta_file
    payload["vehicleDriverFilePath"] = os.path.join(seg_path, "vehicle.json")
    payload["gnssFilePath"] = os.path.join(seg_path, "gnss.json")
    payload["preAnnotationPath"] = ""
    payload["correctPcdFilePath"] = os.path.join(seg_path, "correct_pointclouds")
    payload["calibrationUuid"] = car + "_" + calib_date
    payload["calibrationDateVersion"] = calib_date
    payload["calibrationCar"] = car
    payload["collectionDataDate"] = meta["date"]

    payload_meta = {}
    payload_meta["distance"] = meta["distance"]
    payload_meta["timeInterval"] = meta["time_interval"]
    payload_meta["date"] = meta["date"]
    payload_meta["car"] = meta["car"]
    payload_meta["dataSystem"] = meta["data_system"]
    payload_meta["framesPath"] = meta["frames_path"]
    payload_meta["cameras"] = meta["cameras"]
    payload_meta["dataTags"] = meta["data_tags"]
    payload["segDataMeta"] = payload_meta

    payload_paths = {}
    payload_paths["lane3dAnnoDataPath"] = lane3d_anno_path
    payload_paths["obstacle3dAnnoDataPath"] = obstacle_anno_path
    payload["pathMap"] = payload_paths

    gnss_list, gcj02_list = gen_gnss_info(seg_path, meta)
    payload["gnssList"] = gnss_list
    payload["gcj02List"] = gcj02_list

    payload_json = json.dumps(payload)
    # res = requests.request("POST", SAVE_URL, headers=HTTP_HEADER, data=payload_json)
    # print(res.text)
    ret, response = try_add_seg_payload(payload_json)
    print(response)


def query_seg(segids: list):
    payload = {}
    payload["idList"] = segids
    payload_json = json.dumps(payload)
    res = requests.request("POST", QUERY_URL, headers=HTTP_HEADER, data=payload_json)
    # print(res.text)
    response = res.text
    rc = json.loads(response)
    # rc_status = rc["rc"]
    rc_status = rc['code']
    if rc_status == 0:
        # res = rc["result"]
        res = rc['data']
        if len(res) > 0:
            return len(res), res

    return 0, []


def db_update_seg(seg_path, lane3d_anno_path, obstacle_anno_path):
    meta_file = os.path.join(seg_path, "meta.json")
    # meta_json = open(meta_file)  # 需要手动关闭，否则会造成资源泄露
    # meta = json.load(meta_json)
    with open(meta_file, "r") as jf:
        meta = json.load(jf)
    segid = meta["seg_uid"]

    cnt, res = query_seg([segid])
    if cnt == 0:
        db_add_seg(seg_path, lane3d_anno_path, obstacle_anno_path)
    else:
        calib = meta["calibration"]
        calib_date = calib["date"]
        car = meta["car"]
        payload = {}
        payload["segDataId"] = meta["seg_uid"]
        payload["preAnnotationPath"] = ""

        payload_paths = {}
        payload_paths["lane3dAnnoDataPath"] = lane3d_anno_path
        payload_paths["obstacle3dAnnoDataPath"] = obstacle_anno_path
        payload["pathMap"] = payload_paths

        payload_meta = {}
        payload_meta["dataTags"] = meta["data_tags"]
        payload["segDataMetaQuery"] = payload_meta

        payload_json = json.dumps(payload)
        ret, res = try_update_seg_payload(payload_json)
        print(res)


def test_one_seg():
    seg_path = "/data_cold/origin_data/sihao_0fx60/custom_seg/shuangmu_test_frwang/training_sets/20230327/sihao_0fx60_20230327-16-05-39_seg0/"
    lane_path = "/data_autodrive/auto/custom/sihao_0fx60/clip_lane/training_sets/20230327/sihao_0fx60_20230327-16-05-39_seg0"
    obstacle_path = "/data_autodrive/auto/custom/sihao_0fx60/clip_obstacle/training_sets/20230327/sihao_0fx60_20230327-16-05-39_seg0"
    db_add_seg(seg_path, lane_path, obstacle_path)
    cnt, res = query_seg(["aion_d77052_20230504-13-07-55_seg8"])


if __name__ == "__main__":
    test_one_seg()
