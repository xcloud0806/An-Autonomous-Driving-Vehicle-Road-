import os
import argparse
import csv
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import cv2
import PIL.Image as Image
import folium
from gnss_visual_utils import visual, wgs84_gcj02, judge_China
from haversine import haversine, Unit
from collections import Counter

MDC_CARS = ['10034', '48160', '04228', '18047']
INNO_CARS = ['48160', '36gl1']
HPP_CARS = ['23gc9', '48160','72kx6']
MISS_THRESHOLD = 10 # unit is millseconds

VISUAL_FLAG = False

def fig_to_numpy(fig):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2) # 转换为 RGBA
    image = Image.frombytes("RGBA", (w, h), buf.tobytes()) # 得到 Image RGBA图像对象
    image = np.asarray(image) # 转换为numpy array rgba四通道数组
    rgb_image = image[:, :, :3] # 转换为rgb图像
    return rgb_image[:,:,::-1]

def parse_args():
    parser = argparse.ArgumentParser(description="Init work node")
    # clip decoder params
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/data_autodrive/自动驾驶/hangbai/Traffic_Light/annotations",
    )
    parser.add_argument("--car", "-c", type=str, default="chery_e0y_10034")
    args = parser.parse_args()
    return args

class FrameRawData:
    def __init__(self, frame_path, car_name) -> None:
        self.frame_path = frame_path
        self.car_name = car_name
        self.sensors = []
        self.frame_data_ts = {}
        self.lidar_name = 'lidar'
        self.inno_lidar_name = 'inno_lidar'
        self.use_h265 = False
        self.use_mdc = False
        self.hpp = False
        self.has_inno = False
        self.gnss_csv = "gnss.csv"
        self.veh_csv = "vehicle.csv"
        self.table = {}
        self.valid_flag = True
        self.start_idx = 0
        self.clip_id = os.path.basename(frame_path)

    def init_data(self):
        self.sensors = [
            "ofilm_surround_front_120_8M",
            "ofilm_surround_rear_100_2M",
            "ofilm_surround_front_left_100_2M",
            "ofilm_surround_front_right_100_2M",
            "ofilm_surround_rear_left_100_2M",
            "ofilm_surround_rear_right_100_2M",
        ]
        # if "10034" in car_name or "48160" in car_name:
        for _mdc_car in MDC_CARS:
            if _mdc_car in self.car_name:
                self.sensors = [
                    "ofilm_surround_front_120_8M",
                    "ofilm_surround_front_left_100_2M",
                    "ofilm_surround_front_right_100_2M",
                    "ofilm_surround_rear_left_100_2M",
                    "ofilm_surround_rear_right_100_2M",
                ]
                self.use_h265 = True
                self.use_mdc = True
        for _hpp_car in HPP_CARS:
            if _hpp_car in self.car_name:
                self.sensors.extend(
                    [
                        "ofilm_around_front_190_3M",
                        "ofilm_around_rear_190_3M",
                        "ofilm_around_left_190_3M",
                        "ofilm_around_right_190_3M",
                    ]
                )
                self.hpp = True
        for _inno_car in INNO_CARS:
            if _inno_car in self.car_name:
                self.has_inno = True

        self.load_data_timestamps()

    def path_exist_check(self):
        paths = [
            os.path.join(self.frame_path, self.gnss_csv),
            os.path.join(self.frame_path, self.veh_csv),
            os.path.join(self.frame_path, self.lidar_name)
        ]
        self.table['miss_paths'] = []
        if self.has_inno:
            paths.append(os.path.join(self.frame_path, self.inno_lidar_name))
        paths.extend([os.path.join(self.frame_path, sensor) for sensor in self.sensors])
        for p in paths:
            if not os.path.exists(p):
                logger.error(f"{p} not exists")
                self.valid_flag = False
                self.table['miss_paths'].append(p)

    def load_data_timestamps(self):
        lidars = os.listdir(os.path.join(self.frame_path, self.lidar_name))
        lidars.sort()
        lidar_ts = [int(l.split(".")[0]) for l in lidars]
        lidar_ts_arr = np.array(lidar_ts)
        self.lidar_num = len(lidars)
        self.frame_data_ts['lidar'] = lidar_ts_arr
        self.table['mismatch_sensor_num'] = []
        if self.has_inno:
            innos = os.listdir(os.path.join(self.frame_path, self.inno_lidar_name))
            innos.sort()
            inno_ts = [int(l.split(".")[0]) for l in innos]
            s_num = len(innos)
            if s_num < self.lidar_num - 4:
                logger.error(f"inno {s_num} mismatch lidar {self.lidar_num}")
                self.valid_flag = False
                self.table['mismatch_sensor_num'].append((self.inno_lidar_name, s_num, self.lidar_num))
            inno_ts_arr = np.array(inno_ts)
            self.frame_data_ts['inno'] = inno_ts_arr

        def check_lidar_timestamp_correct(lidar_ts_arr, key='Lidar'):
            a_arr = lidar_ts_arr[:-1]
            b_arr = lidar_ts_arr[1:]
            judge_arr = np.logical_and(np.abs(a_arr - b_arr) < 105, np.abs(a_arr - b_arr) > 95)
            if not np.all(judge_arr):
                logger.error(f"{key} timestamp not correct")
                poss = np.where(np.logical_not(judge_arr))[0].tolist()
                for p in poss:
                    logger.error(f"frame<{p}> {a_arr[p]} vs {b_arr[p]}")
                return False
            return True

        if not check_lidar_timestamp_correct(lidar_ts_arr):
            self.valid_flag = False
            self.table['mismatch_sensor_num'].append((self.lidar_name, self.lidar_num, self.lidar_num))
        if self.has_inno and not check_lidar_timestamp_correct(inno_ts_arr, 'Inno'):
            self.valid_flag = False
            self.table['mismatch_sensor_num'].append((self.inno_lidar_name, s_num, self.lidar_num))

        start_ts = 0
        set_limit_sensor = ""
        for sensor in self.sensors:
            sensor_path = os.path.join(self.frame_path, sensor)
            sensor_files = os.listdir(sensor_path)
            sensor_files.sort()
            sensor_ts = [int(l.split(".")[0]) for l in sensor_files]           
            if sensor_ts[0] > start_ts:
                start_ts = sensor_ts[0]
                set_limit_sensor = sensor
            sensor_ts_arr = np.array(sensor_ts)
            self.frame_data_ts[sensor] = sensor_ts_arr

        for idx, ts in enumerate(lidar_ts):
            if ts > start_ts:
                self.start_idx = idx
                break
        logger.info(f"Lost check start_idx: [{self.start_idx} - {lidar_ts[self.start_idx]}] with [{set_limit_sensor} - {start_ts}]")

    def check_timestamp(self):
        lidar_ts_arr = self.frame_data_ts[self.lidar_name]
        lidar_base_ts = lidar_ts_arr[self.start_idx:] # skip head frames for skip image lost frame
        lidar_base_num = len(lidar_base_ts)
        total_miss_cnt = 0
        for s in self.sensors:            
            images_ts_arr = self.frame_data_ts[s]
            img_num = images_ts_arr.shape[0]
            pre_stat = True
            miss_cnt = 0
            for idx, ts in enumerate(lidar_base_ts.tolist()):
                miss_time_threshold = MISS_THRESHOLD
                if 'around' in s:
                    miss_time_threshold = 15
                match_idx = np.abs(np.abs(images_ts_arr - ts -25)).argmin()
                if np.abs(images_ts_arr[match_idx] - ts - 25) > miss_time_threshold:
                    logger.error(f"frame<{s}.{idx}> {images_ts_arr[match_idx]} vs {ts}")
                    miss_cnt += 1
                    if not pre_stat:
                        logger.error(f"frame<{s}.{idx}&{idx-1}> continuously lost.") # 连续丢了2帧数据
                        self.table['mismatch_sensor_num'].append((s, miss_cnt, img_num))
                        self.valid_flag = False
                    else:
                        pre_stat = False
                else:
                    pre_stat = True

            if miss_cnt > 5:
                logger.error(f"{s} miss {miss_cnt} frames")
                self.valid_flag = False

            total_miss_cnt += miss_cnt

        if total_miss_cnt > (len(self.sensors) * 2):
            logger.error(f"all {len(self.sensors)} cameras total miss {total_miss_cnt} frames in {lidar_base_num} lidar frames")
            self.valid_flag = False

    def check_gnss_location(self, gnss_infos, pair_gnss_ts):
        gnss_info_ts = [int(l) for l in gnss_infos['utc_time'] if l != 'na']
        gnss_info_lat = [float(l) for l in gnss_infos['latitude'] ]
        gnss_info_lon = [float(l) for l in gnss_infos['longitude'] ]
        gnss_info_status = [int(l) for l in gnss_infos['gps_status'] ]

        gnss_dict = {}
        for idx, ts in enumerate(gnss_info_ts):
            gnss_dict[ts] = {
                'lat': gnss_info_lat[idx],
                'lon': gnss_info_lon[idx],
                'status': gnss_info_status[idx]
            }
        
        pre_status_lst = []
        pts = []
        pre_lat = gnss_dict[pair_gnss_ts[0]].get('lat')
        pre_lon = gnss_dict[pair_gnss_ts[0]].get('lon')
        pre_dist = 0 # unit meters
        for idx, ts in enumerate(pair_gnss_ts):
            _info = gnss_dict.get(ts, None)
            pre_status_lst.append(_info['status'])
            if len(pre_status_lst) > 10:
                pre_status_lst.pop(0)
            c = Counter(pre_status_lst)
            s_ok = c.get(42, 0) + c.get(52, 0)
            if len(pre_status_lst) > 8 and s_ok < 2:
                logger.warning(f"{ts} gnss status {_info['status']} not ok")
                # self.valid_flag = False

            if idx < 1:
                continue

            curr_lat = _info['lat']
            curr_lon = _info['lon']
            lon_gcj02, lat_gcj02 = wgs84_gcj02(curr_lon, curr_lat)
            pts.append([lat_gcj02, lon_gcj02])
            if judge_China(lon_gcj02, lat_gcj02):
                logger.error(f"{ts} {curr_lat} {curr_lon} not in China")
                self.valid_flag = False
            pa = (pre_lat, pre_lon)
            pb = (curr_lat, curr_lon)
            dist = haversine(pa, pb, unit=Unit.METERS)
            if pre_dist == 0:
                pre_dist = dist
            else:                
                if abs(dist - pre_dist) > 20: # unit meters
                    logger.error(f"{ts} {dist} {pre_dist} location change [{abs(dist - pre_dist)}m]too much")
                    self.valid_flag = False
                pre_dist = dist
        return pts

    def check_gnss_vehicle(self):
        # gnss_df = pd.read_csv(gnss_csv)
        def parse_csv(csv_file):
            df = pd.read_csv(csv_file)
            col_lists = [df[col].tolist() for col in df.columns]
            result = dict(zip(df.columns, col_lists))
            return result

        self.table['gnss_veh_check'] = []
        gnss_csv = os.path.join(self.frame_path, self.gnss_csv)
        gnss_info = parse_csv(gnss_csv)
        vehi_csv = os.path.join(self.frame_path, self.veh_csv)
        vehi_info = parse_csv(vehi_csv)

        lidar_base_ts = self.frame_data_ts[self.lidar_name][self.start_idx:] # skip head frames for skip image lost frame
        valid_lidar_num = lidar_base_ts.shape[0]
        start_ts = lidar_base_ts[0]
        end_ts = lidar_base_ts[-1]

        gnss_info_ts = [int(l) for l in gnss_info['utc_time'] if l != 'na']
        vehi_info_ts = [int(l) for l in vehi_info['utc_time'] if l != 'na']
        gnss_info_spd = [float(l) * 3.6 for l in gnss_info['speed'] if l != 'na'] # unit KPH
        vehi_info_spd = [float(l)       for l in vehi_info['vehicle_spd'] if l != 'na'] # unit KPH

        valid_gnss_st_idx = 0
        valid_gnss_ed_idx = len(gnss_info_ts)
        for idx, ts in enumerate(gnss_info_ts):
            if ts <= start_ts:
                valid_gnss_st_idx = idx
            if ts >= end_ts:
                valid_gnss_ed_idx = idx
                break
        gnss_valid_ts = np.array(gnss_info_ts[valid_gnss_st_idx:valid_gnss_ed_idx])
        gnss_valid_spd = np.array(gnss_info_spd[valid_gnss_st_idx:valid_gnss_ed_idx])
        gnss_valid_num = gnss_valid_ts.shape[0]

        valid_veh_st_idx = 0
        valid_veh_ed_idx = len(vehi_info_ts)
        for idx, ts in enumerate(vehi_info_ts):
            if ts <= start_ts:
                valid_veh_st_idx = idx
            if ts >= end_ts:
                valid_veh_ed_idx = idx
                break
        vehi_valid_ts = np.array(vehi_info_ts[valid_veh_st_idx:valid_veh_ed_idx])
        vehi_valid_spd = np.array(vehi_info_spd[valid_veh_st_idx:valid_veh_ed_idx])
        vehi_valid_num = vehi_valid_ts.shape[0]

        if gnss_valid_num < (valid_lidar_num * 10) - 10: # GNSS is 100HZ
            logger.error(f"gnss valid num {gnss_valid_num} less than lidar num {valid_lidar_num}")
            self.valid_flag = False
            self.table['gnss_veh_check'].append((gnss_valid_num, vehi_valid_num, valid_lidar_num))
        if vehi_valid_num < (valid_lidar_num * 5) - 10: # Vehicle is 50HZ
            logger.error(f"vehicle valid num {vehi_valid_num} less than lidar num {valid_lidar_num}")
            self.valid_flag = False
            self.table['gnss_veh_check'].append((gnss_valid_num, vehi_valid_num, valid_lidar_num))

        # 通过时间戳同步来获取每一帧雷达的GNSS和VEHICLE的同步帧
        plot_gnss_spd = []
        plot_vehi_spd = []
        for idx, ts in enumerate(lidar_base_ts.tolist()):
            match_idx = np.abs(np.abs(gnss_valid_ts - ts)).argmin()
            gnss_spd = gnss_valid_spd[match_idx]
            gnss_ts = gnss_valid_ts[match_idx]
            plot_gnss_spd.append(gnss_spd)

            match_idx = np.abs(np.abs(vehi_valid_ts - ts)).argmin()
            vehi_spd = vehi_valid_spd[match_idx]
            vehi_ts = vehi_valid_ts[match_idx]
            plot_vehi_spd.append(vehi_spd)

            spd_diff = abs(gnss_spd - vehi_spd)
            if (gnss_spd > 20 and spd_diff > 0.05 * gnss_spd) or (
                gnss_spd > 8 and spd_diff > 0.1 * gnss_spd
            ):  # when speed diff > 5% gnss, check failed
                logger.error(f"frame<{idx}> {ts} GNSS:{gnss_ts} {gnss_spd} VEHICLE:{vehi_ts} {vehi_spd}")
                self.table['gnss_veh_check'].append((gnss_valid_num, vehi_valid_num, valid_lidar_num))
                self.valid_flag = False
        
        pts = self.check_gnss_location(gnss_info, gnss_valid_ts)

        if len(pts) > 10 and VISUAL_FLAG:
            visual_map = visual(pts)
            visual_map.save(f"{self.car_name}_{self.clip_id}_gnss_location.html")

        if VISUAL_FLAG:
            # 通过matplotlib 绘制速度曲线
            dpi_value = 200
            fig_scale = 8 / dpi_value
            fig = plt.figure(dpi=dpi_value, figsize=(len(plot_gnss_spd) * fig_scale, 300 * fig_scale))
            ax = fig.add_subplot(111)
            # ax.set_xlabel('index', fontsize=12)
            ax.set_ylabel('speed(KPH)', fontsize=20)
            # ax.set_aspect(1)
            x_axes = np.arange(len(plot_gnss_spd))
            ax.plot(x_axes, plot_gnss_spd, 'ro--', label='GNSS', linewidth=1)
            ax.plot(x_axes, plot_vehi_spd, 'bo--', label='VEHICLE', linewidth=1)
            ax.legend(fontsize=20)
            ax.grid(axis='y', alpha=0.5, linestyle='--')
            fig_img = fig_to_numpy(fig)
            plt.close(fig)
            cv2.imwrite(f"{self.car_name}_{self.clip_id}_gnss_vehicle_check.jpg", fig_img)

def check_valid(frame_path, car_name):
    inst = FrameRawData(frame_path, car_name)
    inst.path_exist_check()
    if not inst.valid_flag:
        return False, inst.table
    inst.init_data()
    if not inst.valid_flag:
        return False, inst.table
    inst.check_gnss_vehicle()
    if not inst.valid_flag:
        return False, inst.table
    inst.check_timestamp()
    return inst.valid_flag, inst.table

if __name__ == "__main__":
    logger.add(f"lost_frame_check.log", rotation="50 MB", level="INFO")
    args = parse_args()

    query = args.input
    name = os.path.basename(query)
    
    clips = os.listdir(query)
    clips = [c for c in clips if os.path.isdir(os.path.join(query, c))]
    total = len(clips)
    pass_cnt = 0
    fail_cnt = 0
    fail_lst = {}
    for clip in clips:
        if not os.path.isdir(os.path.join(query, clip)):
            continue     
        logger.info(f"checking {clip}")
        ret, table = check_valid(os.path.join(query, clip), args.car)        
        if ret:
            logger.info(f"{clip} check pass")
            pass_cnt += 1
        else:
            logger.error(f"{clip} check failed")                        
            fail_cnt += 1
            fail_lst[clip] = table
    df = pd.DataFrame.from_dict(fail_lst, orient='index')
    logger.info(f"{name} total: {total}, pass: {pass_cnt}, fail: {fail_cnt}")
    logger.warning(f"FAILED: \n{df}")
