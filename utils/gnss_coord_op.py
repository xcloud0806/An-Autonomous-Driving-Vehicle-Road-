# -*- coding: utf-8 -*-
import math

# 设置常量
pi = 3.141592653589793234  # π
r_pi = pi * 3000.0 / 180.0
la = 6378245.0  # 长半轴
ob = 0.00669342162296594323  # 扁率

# 简单判断坐标点是否在中国
# 不在的话返回True
# 在的话返回False
def judge_China(lon, lat):
    if lon < 70 or lon > 140:
        return True
    if lat < 0 or lat > 55:
        return True
    return False

# gcj02 -> bd09
# lng为gcj02的经度
# lat为gcj02的纬度
# 返回值为转换后坐标的列表形式，[经度, 纬度]
def gcj02_bd09(lon_gcj02, lat_gcj02):
    b = math.sqrt(lon_gcj02 * lon_gcj02 + lat_gcj02 * lat_gcj02) + 0.00002 * math.sin(lat_gcj02 * r_pi)
    o = math.atan2(lat_gcj02, lon_gcj02) + 0.000003 * math.cos(lon_gcj02 * r_pi)
    lon_bd09 = b * math.cos(o) + 0.0065
    lat_bd09 = b * math.sin(o) + 0.006
    return [lon_bd09, lat_bd09]

# bd09 -> gcj02
# lng为bd09的经度
# lat为bd09的纬度
# 返回值为转换后坐标的列表形式，[经度, 纬度]
def bd09_gcj02(lon_bd09, lat_bd09):
    m = lon_bd09 - 0.0065
    n = lat_bd09 - 0.006
    c = math.sqrt(m * m + n * n) - 0.00002 * math.sin(n * r_pi)
    o = math.atan2(n, m) - 0.000003 * math.cos(m * r_pi)
    lon_gcj02 = c * math.cos(o)
    lat_gcj02 = c * math.sin(o)
    return [lon_gcj02, lat_gcj02]

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

# gcj02 -> wgs84
# lng为gcj02的经度
# lat为gcj02的纬度
# 返回值为转换后坐标的列表形式，[经度, 纬度]
def gcj02_wgs84(lon_gcj02, lat_gcj02):
    if judge_China(lon_gcj02, lat_gcj02):
        return [lon_gcj02, lat_gcj02]
    tlat = transformlat(lon_gcj02 - 105.0, lat_gcj02 - 35.0)
    tlng = transformlng(lon_gcj02 - 105.0, lat_gcj02 - 35.0)
    rlat = lat_gcj02 / 180.0 * pi
    m = math.sin(rlat)
    m = 1 - ob * m * m
    sm = math.sqrt(m)
    tlat = (tlat * 180.0) / ((la * (1 - ob)) / (m * sm) * pi)
    tlng = (tlng * 180.0) / (la / sm * math.cos(rlat) * pi)
    lat_wgs84 = 2 * lat_gcj02 - (lat_gcj02 + tlat)
    lon_wgs84 = 2 * lon_gcj02 - (lon_gcj02 + tlng)
    return [lon_wgs84, lat_wgs84]

# wgs84 -> bd09
# lng为wgs84的经度
# lat为wgs84的纬度
# 返回值为转换后坐标的列表形式，[经度, 纬度]
def wgs84_bd09(lon_wgs84, lat_wgs84):
    # 先把wgs84坐标系的坐标转换为火星坐标系
    tmpList_gcj02 = wgs84_gcj02(lon_wgs84, lat_wgs84)
    # 然后把火星坐标系的坐标转换为百度坐标系
    return gcj02_bd09(tmpList_gcj02[0], tmpList_gcj02[1])

# bd09 -> wgs84
# lng为bd09的经度
# lat为bd09的纬度
# 返回值为转换后坐标的列表形式，[经度, 纬度]
def bd09_wgs84(lon_bd09, lat_bd09):
    # 先把百度坐标系的经纬度转换为火星坐标系
    tmpList_gcj02 = bd09_gcj02(lon_bd09, lat_bd09)
    # 然后把火星坐标系的坐标转换为百度坐标系
    return gcj02_wgs84(tmpList_gcj02[0], tmpList_gcj02[1])

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