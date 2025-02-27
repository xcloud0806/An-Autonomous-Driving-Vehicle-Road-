import numpy as np
from copy import deepcopy
from haversine import haversine, Unit
import csv
import os
from utils import cam_position, bpearl_list, inno_list

gnss_key_offset = 5
vehicle_key_offset = 10
speed_key = "speed"
veh_speed_key = "vehicle_spd"
status_key = "gps_status"
lati_key = "latitude"
long_key = "longitude"

class Segment:
    def __init__(self):
        self.gnss = []
        self.vehicle = []
        self.lidar = []
        self.match = {}
        self.raw = {}
        self.enable_cams = []
        self.enable_bpearls = []
        self.enable_innos = []

        self.seg_vehicle = {}
        self.seg_gnss = {}
        
        self.distance = 0.0
        self.time_interval = 0.0
        self.related_seg = ""
        self.gnss_info_lost = False
        self.vehicle_info_lost = False
    def reset(self):
        self.gnss.clear()
        self.vehicle.clear()
        self.lidar.clear()
        self.match.clear()
        self.raw.clear()
    
    def fill(self, gnss_info, match_info, raw_info, vehicle_info, enable_cams, sensor_idx:dict, enable_bpearls=None, enable_innos=None):
        key_ts = self.lidar
        enbale_vech = True
        if len(vehicle_info) < len(match_info):
            print("Vehicle Info Disabled!")
            enbale_vech = False
        self.enable_cams = enable_cams
        for cam in enable_cams:
            self.match[cam] = []
            self.raw[cam] = []
        
        if enable_bpearls is not None:
            self.enable_bpearls = enable_bpearls
            for bpearl in enable_bpearls:
                self.match[bpearl] = []
                self.raw[bpearl] = []

        if enable_innos is not None:
            self.enable_innos = enable_innos
            for inno in enable_innos:
                self.match[inno] = []
                self.raw[inno] = []

        gnss_ts_arr = np.array(list(gnss_info.keys()))
        vech_ts_arr = np.array(list(vehicle_info.keys()))

        min_gnss_ts = 0
        max_gnss_ts = key_ts[0]
        if gnss_ts_arr[-1] - key_ts[0] <= 0:
            self.gnss_info_lost = True
            print("ERROR: in this seg, Gnss Info is lost!")
            return
        
        min_vech_ts = 0
        max_vech_ts = key_ts[0]
        if vech_ts_arr[-1] - key_ts[0] <= 0:
            self.vehicle_info_lost = True
            print("ERROR: in this seg, vehicle Info is lost!")
            return
                
        for i, pts in enumerate(key_ts):                    
            gnss_idx = abs(gnss_ts_arr - pts).argmin()
            gnss_ts = gnss_ts_arr[gnss_idx]
            # status = int(gnss_info[gnss_ts][status_key])
            self.gnss.append(gnss_ts)
            if min_gnss_ts == 0:
                min_gnss_ts = gnss_ts
            if gnss_ts > max_gnss_ts:
                max_gnss_ts = gnss_ts
            # if abs(gnss_ts - pts) > gnss_key_offset:
            #     self.gnss.append(None)
            if enbale_vech:
                vehi_idx = abs(vech_ts_arr - pts).argmin()
                vehi_ts = vech_ts_arr[vehi_idx]            
                self.vehicle.append(vehi_ts)            
                if min_vech_ts == 0:
                    min_vech_ts = vehi_ts
                if vehi_ts > max_vech_ts:
                    max_vech_ts = vehi_ts                    
            # if abs(vehi_ts - pts) > vehicle_key_offset:
            #     self.vehicle.append(None)

            for cam in enable_cams:
                idx = sensor_idx[cam] - 1 
                cam_ts = match_info[pts][idx]
                self.match[cam].append(cam_ts)
                raw_cam_ts = raw_info[pts][idx]
                self.raw[cam].append(raw_cam_ts)
            
            if enable_bpearls is not None:
                for bpearl in enable_bpearls:
                    idx = sensor_idx[bpearl] - 1 
                    cam_ts = match_info[pts][idx]
                    self.match[bpearl].append(cam_ts)
                    self.raw[bpearl].append(cam_ts)
            
            if enable_innos is not None:
                for inno in enable_innos:
                    idx = sensor_idx[inno] - 1 
                    cam_ts = match_info[pts][idx]
                    self.match[inno].append(cam_ts)
                    self.raw[inno].append(cam_ts)
            
            if "sin_radar" in sensor_idx:
                idx = sensor_idx["sin_radar"] - 1
                cam_ts = match_info[pts][idx]
                if "sin_radar" not in self.match:
                    self.match["sin_radar"] = []
                self.match["sin_radar"].append(cam_ts)
                if "sin_radar" not in self.raw:
                    self.raw["sin_radar"] = []
                self.raw['sin_radar'].append(cam_ts)
        
        # print("lidar {}F <-> {}/{}F".format(len(self.lidar), enable_cams[1], len(self.match[enable_cams[1]])))
        
        for _k, _v in gnss_info.items():
            ts = _k
            if ts >= min_gnss_ts and ts <= max_gnss_ts:
                self.seg_gnss[ts] = _v
        
        if enbale_vech:
            for _k, _v in vehicle_info.items():
                ts = _k
                if ts >= min_vech_ts and ts <= max_vech_ts:
                    self.seg_vehicle[ts] = _v

class SegTool:
    def __init__(self, gnss_info: dict, veh_info:dict, match_info: dict, 
                 distance=600, time_interval=60, seg_mode=None) -> None:
        self.gnss_info = gnss_info
        self.veh_info = veh_info
        self.lidar_key = list(match_info.keys())
        self.length = len(self.lidar_key)
        self.lidar_key.sort()
        self.set_dist = distance
        self.set_intr = time_interval
        self.segs = {}
        self.seg_mode = seg_mode
    def try_cut_seg(self):
        speeds = {}
        gnss_ts_arr = np.array(list(self.gnss_info.keys()))
        veh_ts_arr = np.array(list(self.veh_info.keys()))
        for key in self.lidar_key:
            speeds[key] = None
            gnss_idx = abs(gnss_ts_arr - key).argmin()
            gnss_ts = gnss_ts_arr[gnss_idx]
            # just not confirm gnss status
            # status = int(self.gnss_info[key + offset][status_key])                    
            # if status == 2 or status == 42 or status == 52: # OK 2 in zeer and 42/52 in iflytek
            speed = self.gnss_info[gnss_ts][speed_key]
            if speed == 'na':
                continue
            speed = float(speed)
            if speed == 0.0:
                veh_idx = abs(veh_ts_arr - key).argmin()
                veh_ts = veh_ts_arr[veh_idx]
                str_spd = self.veh_info[veh_ts][veh_speed_key]
                if str_spd == 'na' or str_spd == '':
                    continue
                veh_speed = float(str_spd)
                if veh_speed != 0.0:
                    speed = veh_speed
            speeds[key] = speed           
        print("Speeds cal over.")

        seg_lidar = []
        distance = 0.0
        time_interval = 0.0        
        t_idx = 10 # start from the fifth second, because clip data usually loses frames on the head
        base_t = self.lidar_key[t_idx]
        while True:            
            if base_t not in speeds:
                t_idx += 1
                if t_idx > len(self.lidar_key) - 1:
                    break
                base_t = self.lidar_key[t_idx]
                continue
            elif speeds[base_t] is None:
                t_idx += 1
                if t_idx > len(self.lidar_key) - 1:
                    break
                base_t = self.lidar_key[t_idx]
                continue
            elif speeds[base_t] < 0.01:
                t_idx += 1
                if t_idx > len(self.lidar_key) - 1:
                    break
                base_t = self.lidar_key[t_idx]
                continue
            else:
                break
        t_idx += 1

        if t_idx > (self.length - 1):
            return self.segs
        
        if self.set_dist == 0 and self.set_intr == 0:
            # return self.segs
            while True:
                try:
                    lidar_t = self.lidar_key[t_idx]
                    if lidar_t - base_t > 998: # eval distance every second                        
                        gnss_a = abs(gnss_ts_arr - base_t).argmin()
                        gnss_t_a = gnss_ts_arr[gnss_a]
                        long_a = float(self.gnss_info[gnss_t_a][long_key])
                        lati_a = float(self.gnss_info[gnss_t_a][lati_key])
                        gnss_b = abs(gnss_ts_arr - lidar_t).argmin()
                        gnss_t_b = gnss_ts_arr[gnss_b]
                        long_b = float(self.gnss_info[gnss_t_b][long_key])
                        lati_b = float(self.gnss_info[gnss_t_b][lati_key])
                        pa = (lati_a, long_a)
                        pb = (lati_b, long_b)
                        dist = haversine(pa, pb, unit=Unit.METERS)
                        distance += dist
                        base_t = lidar_t
                    seg_lidar.append(lidar_t)                    
                    time_interval += (lidar_t - base_t)
                    t_idx += 1
                    if t_idx > (self.length - 1):
                        seg = Segment()
                        seg.lidar = deepcopy(seg_lidar)
                        seg.distance = distance
                        seg.time_interval = time_interval
                        self.segs[seg.lidar[0]] = seg
                        seg_lidar.clear()
                        distance = 0
                        time_interval = 0
                        break
                except IndexError:
                    print("Index {} outof {}, curr seg length {}".format(t_idx, self.length, len(seg_lidar)))
                    return self.segs
                except:
                    raise Exception
        elif self.set_dist == 0 and self.set_intr >= 0:
            print("Set interval is {} sec".format(self.set_intr))
            while True:
                try:
                    lidar_t = self.lidar_key[t_idx]
                    if lidar_t - base_t > 998: # eval seg every second
                        if time_interval > (self.set_intr * 1000): 
                            seg = Segment()
                            seg.lidar = deepcopy(seg_lidar)
                            seg.distance = distance
                            seg.time_interval = time_interval
                            self.segs[seg.lidar[0]] = seg
                            seg_lidar.clear()
                            distance = 0.0
                            time_interval = 0.0
                        else:
                            time_interval += (lidar_t - base_t)   
                            seg_lidar.append(lidar_t)
                        base_t = lidar_t                            
                    else:
                        seg_lidar.append(lidar_t)
                
                    t_idx += 1
                    if t_idx > (self.length - 1):
                        break
                except IndexError:
                    print("Index {} outof {}, curr seg length {}".format(t_idx, self.length, len(seg_lidar)))
                    return self.segs
                except:
                    raise Exception
        else:
            while True: 
                try:
                    lidar_t = self.lidar_key[t_idx]
                    if lidar_t - base_t > 998: # eval distance every second                        
                        gnss_a = abs(gnss_ts_arr - base_t).argmin()
                        gnss_t_a = gnss_ts_arr[gnss_a]
                        long_a = float(self.gnss_info[gnss_t_a][long_key])
                        lati_a = float(self.gnss_info[gnss_t_a][lati_key])
                        gnss_b = abs(gnss_ts_arr - lidar_t).argmin()
                        gnss_t_b = gnss_ts_arr[gnss_b]
                        long_b = float(self.gnss_info[gnss_t_b][long_key])
                        lati_b = float(self.gnss_info[gnss_t_b][lati_key])
                        pa = (lati_a, long_a)
                        pb = (lati_b, long_b)
                        dist = haversine(pa, pb, unit=Unit.METERS)
                        if dist > self.set_dist: # skip abnormal positions
                            print(f"cacl a larger distance ->{dist}<- between two gnss point [A{pa}, B{pb}].")
                            t_idx += 1
                            continue
                        distance += dist
                        time_interval += (lidar_t - base_t)
                        if distance > self.set_dist:
                            seg_lidar.append(lidar_t)
                            seg = Segment()
                            seg.lidar = deepcopy(seg_lidar)
                            seg.distance = distance
                            seg.time_interval = time_interval
                            self.segs[seg.lidar[0]] = seg
                            # print(list(self.segs.keys()))
                            seg_lidar.clear()
                            distance = 0.0
                            time_interval = 0.0
                        else:
                            # 增加在最大距离判断时限制分段的最大时长
                            if self.set_intr > 0 and time_interval > (self.set_intr * 1000) :
                                seg = Segment()
                                seg.lidar = deepcopy(seg_lidar)
                                seg.distance = distance
                                seg.time_interval = time_interval
                                self.segs[seg.lidar[0]] = seg
                                seg_lidar.clear()
                                distance = 0.0
                                time_interval = 0.0

                            seg_lidar.append(lidar_t)
                        base_t = lidar_t
                    else:
                        seg_lidar.append(lidar_t)

                    t_idx += 1
                    if t_idx > (self.length - 1) :
                        if len(seg_lidar) > 5:
                            seg = Segment()
                            seg.lidar = deepcopy(seg_lidar)
                            seg.distance = distance
                            seg.time_interval = time_interval
                            self.segs[seg.lidar[0]] = seg
                        break   
                except IndexError:
                    print("Index {} outof {}, curr seg length {}".format(t_idx, self.length, len(seg_lidar)))
                    return self.segs
                except:
                    raise Exception

        return self.segs

    def try_cut_seg_luce_mode(self):
        """
        cut segs by luce mode
        """
        speeds = {}
        gnss_ts_arr = np.array(list(self.gnss_info.keys()))
        veh_ts_arr = np.array(list(self.veh_info.keys()))
        for key in self.lidar_key:
            speeds[key] = None
            gnss_idx = abs(gnss_ts_arr - key).argmin()
            gnss_ts = gnss_ts_arr[gnss_idx]
            # just not confirm gnss status
            # status = int(self.gnss_info[key + offset][status_key])                    
            # if status == 2 or status == 42 or status == 52: # OK 2 in zeer and 42/52 in iflytek
            speed = self.gnss_info[gnss_ts][speed_key]
            if speed == 'na':
                continue
            speed = float(speed)
            if speed == 0.0:
                veh_idx = abs(veh_ts_arr - key).argmin()
                veh_ts = veh_ts_arr[veh_idx]
                str_spd = self.veh_info[veh_ts][veh_speed_key]
                if str_spd == 'na' or str_spd == '':
                    continue
                veh_speed = float(str_spd)
                if veh_speed != 0.0:
                    speed = veh_speed
            speeds[key] = speed           
        print("Speeds cal over.")
        
        seg_lidar = []
        distance = 0.0
        time_interval = 0.0        
        t_idx = 10 # start from the fifth second, because clip data usually loses frames on the head
        base_t = self.lidar_key[t_idx]

        while True:
            try:
                lidar_t = self.lidar_key[t_idx]
                if lidar_t - base_t > 998: # eval distance every second                        
                    gnss_a = abs(gnss_ts_arr - base_t).argmin()
                    gnss_t_a = gnss_ts_arr[gnss_a]
                    long_a = float(self.gnss_info[gnss_t_a][long_key])
                    lati_a = float(self.gnss_info[gnss_t_a][lati_key])
                    gnss_b = abs(gnss_ts_arr - lidar_t).argmin()
                    gnss_t_b = gnss_ts_arr[gnss_b]
                    long_b = float(self.gnss_info[gnss_t_b][long_key])
                    lati_b = float(self.gnss_info[gnss_t_b][lati_key])
                    pa = (lati_a, long_a)
                    pb = (lati_b, long_b)
                    dist = haversine(pa, pb, unit=Unit.METERS)
                    distance += dist
                    base_t = lidar_t
                seg_lidar.append(lidar_t)                    
                time_interval += (lidar_t - base_t)
                t_idx += 1
                if t_idx > (self.length - 1):
                    seg = Segment()
                    seg.lidar = deepcopy(seg_lidar)
                    seg.distance = distance
                    seg.time_interval = time_interval
                    self.segs[seg.lidar[0]] = seg
                    seg_lidar.clear()
                    distance = 0
                    time_interval = 0
                    break
            except IndexError:
                print("Index {} outof {}, curr seg length {}".format(t_idx, self.length, len(seg_lidar)))
                return self.segs
            except:
                raise Exception
        return self.segs

    def try_cut_seg_with_spec_timestamp(self, timestamp:list):
        """
        timestamp: list of timestamp [ [st, et, distance, day_seg_name], ...]
        cut segs by timestamp list
        """
        gnss_ts_arr = np.array(list(self.gnss_info.keys()))

        for t_idx in range(len(timestamp)):
            st = int(timestamp[t_idx][0])
            et = int(timestamp[t_idx][1])
            distance = timestamp[t_idx][2]
            day_seg_name = timestamp[t_idx][3]
            print ("st: {}, et: {}, distance: {}, day_seg_name: {}".format(st, et, distance, day_seg_name))
            if st > et:
                print("timestamp error: start time > end time")
                continue
            if st < gnss_ts_arr[0]:
                print("timestamp error: start time < 0")
                continue
            if distance < 250:
                print("timestamp error: distance < 250")
                continue
            
            try:
                lidar_idx = 0
                seg_lidar = []
                time_interval = float(et - st) / 1000
                while True:
                    lidar_ts = self.lidar_key[lidar_idx]
                    if lidar_ts < st :
                        lidar_idx += 1
                        continue
                    if lidar_ts > et:
                        break
                    seg_lidar.append(lidar_ts)
                    lidar_idx += 1

                if len(seg_lidar) * 100 < (et - st - 1000):
                    print("timestamp error: lidar data not enough")
                    continue

                seg = Segment()
                seg.lidar = deepcopy(seg_lidar)
                seg.distance = distance
                seg.time_interval = time_interval
                seg.related_seg = day_seg_name
                self.segs[seg.lidar[0]] = seg
            except IndexError:
                print("Index {} outof {}, curr seg length {}".format(t_idx, self.length, len(seg_lidar)))
                # return self.segs
            except Exception as e:
                print("An unexpected error occurred: %s", str(e))
        return self.segs

    def try_cut_seg_hpp_mode(self):
        speeds = {}
        gnss_ts_arr = np.array(list(self.gnss_info.keys()))
        veh_ts_arr = np.array(list(self.veh_info.keys()))
        for key in self.lidar_key:
            speeds[key] = None
            gnss_idx = abs(gnss_ts_arr - key).argmin()
            gnss_ts = gnss_ts_arr[gnss_idx]
            # just not confirm gnss status
            # status = int(self.gnss_info[key + offset][status_key])                    
            # if status == 2 or status == 42 or status == 52: # OK 2 in zeer and 42/52 in iflytek
            speed = self.gnss_info[gnss_ts][speed_key]
            if speed == 'na':
                continue
            speed = float(speed)
            if speed == 0.0:
                veh_idx = abs(veh_ts_arr - key).argmin()
                veh_ts = veh_ts_arr[veh_idx]
                str_spd = self.veh_info[veh_ts][veh_speed_key]
                if str_spd == 'na' or str_spd == '':
                    continue
                veh_speed = float(str_spd)
                if veh_speed != 0.0:
                    speed = veh_speed
            speeds[key] = speed           
        print("Speeds cal over.")

        seg_lidar = []
        distance = 0.0
        time_interval = 0.0        
        t_idx = 50 # start from the fifth second, because clip data usually loses frames on the head
        if self.seg_mode == "hpp_luce":
            t_idx = 20

        base_t = self.lidar_key[t_idx]
        if self.seg_mode == "hpp":
            while True:
                if base_t not in speeds:
                    t_idx += 1
                    if t_idx > len(self.lidar_key) - 1:
                        break
                    base_t = self.lidar_key[t_idx]
                    continue
                elif speeds[base_t] is None:
                    t_idx += 1
                    if t_idx > len(self.lidar_key) - 1:
                        break
                    base_t = self.lidar_key[t_idx]
                    continue
                elif speeds[base_t] < 0.01:
                    t_idx += 1
                    if t_idx > len(self.lidar_key) - 1:
                        break
                    base_t = self.lidar_key[t_idx]
                    continue
                else:
                    break
            t_idx += 1

        if t_idx > (self.length - 1):
            return self.segs
        
        if self.set_dist == 0 and self.set_intr == 0:
            # return self.segs
            while True:
                try:
                    lidar_t = self.lidar_key[t_idx]
                    if lidar_t - base_t > 998: # eval distance every second   
                        veh_a = abs(veh_ts_arr - base_t).argmin()
                        veh_b = abs(veh_ts_arr - lidar_t).argmin()
                        veh_t_a = veh_ts_arr[veh_a]
                        veh_t_b = veh_ts_arr[veh_b]
                        veh_a_speed = float(self.veh_info[veh_t_a][veh_speed_key])
                        veh_b_speed = float(self.veh_info[veh_t_b][veh_speed_key])
                        dist = 0.5*(veh_a_speed + veh_b_speed)*(veh_t_b - veh_t_a)/3600 # unit: m
                        distance += dist
                        base_t = lidar_t
                    seg_lidar.append(lidar_t)                    
                    time_interval += (lidar_t - base_t)
                    t_idx += 1
                    if t_idx > (self.length - 1):
                        seg = Segment()
                        seg.lidar = deepcopy(seg_lidar)
                        seg.distance = distance
                        seg.time_interval = time_interval
                        self.segs[seg.lidar[0]] = seg
                        seg_lidar.clear()
                        distance = 0
                        time_interval = 0
                        break
                except IndexError:
                    print("Index {} outof {}, curr seg length {}".format(t_idx, self.length, len(seg_lidar)))
                    return self.segs
                except:
                    raise Exception
        elif self.set_dist == 0 and self.set_intr >= 0:
            print("Set interval is {} sec".format(self.set_intr))
            while True:
                try:
                    lidar_t = self.lidar_key[t_idx]
                    if lidar_t - base_t > 998: # eval seg every second
                        if time_interval > (self.set_intr * 1000): 
                            seg = Segment()
                            seg.lidar = deepcopy(seg_lidar)
                            seg.distance = distance
                            seg.time_interval = time_interval
                            self.segs[seg.lidar[0]] = seg
                            seg_lidar.clear()
                            distance = 0.0
                            time_interval = 0.0
                        else:
                            time_interval += (lidar_t - base_t)   
                            seg_lidar.append(lidar_t)
                        base_t = lidar_t                   
                    else:
                        seg_lidar.append(lidar_t)
                
                    t_idx += 1
                    if t_idx > (self.length - 1):
                        break
                except IndexError:
                    print("Index {} outof {}, curr seg length {}".format(t_idx, self.length, len(seg_lidar)))
                    return self.segs
                except:
                    raise Exception
        else:
            while True: 
                try:
                    lidar_t = self.lidar_key[t_idx]
                    if lidar_t - base_t > 998: # eval distance every second      
                        veh_a = abs(veh_ts_arr - base_t).argmin()
                        veh_b = abs(veh_ts_arr - lidar_t).argmin()
                        veh_t_a = veh_ts_arr[veh_a]
                        veh_t_b = veh_ts_arr[veh_b]
                        veh_a_speed = float(self.veh_info[veh_t_a][veh_speed_key])
                        veh_b_speed = float(self.veh_info[veh_t_b][veh_speed_key])
                        dist = 0.5*(veh_a_speed + veh_b_speed)*(veh_t_b - veh_t_a)/3600 # unit: m
                        if dist > self.set_dist: # skip abnormal positions
                            # print(f"cacl a larger distance ->{dist}<- between two gnss point [A{pa}, B{pb}].")
                            print(f"cacl a larger distance ->{dist}<- between two time point [A{veh_a}, B{veh_b}].")
                            t_idx += 1
                            continue
                        distance += dist
                        time_interval += (lidar_t - base_t)
                        if distance > self.set_dist:
                            seg_lidar.append(lidar_t)
                            seg = Segment()
                            seg.lidar = deepcopy(seg_lidar)
                            seg.distance = distance
                            seg.time_interval = time_interval
                            self.segs[seg.lidar[0]] = seg
                            # print(list(self.segs.keys()))
                            seg_lidar.clear()
                            distance = 0.0
                            time_interval = 0.0
                        else:
                            # 增加在最大距离判断时限制分段的最大时长
                            if self.set_intr > 0 and time_interval > (self.set_intr * 1000) :
                                seg = Segment()
                                seg.lidar = deepcopy(seg_lidar)
                                seg.distance = distance
                                seg.time_interval = time_interval
                                self.segs[seg.lidar[0]] = seg
                                seg_lidar.clear()
                                distance = 0.0
                                time_interval = 0.0

                            seg_lidar.append(lidar_t)
                        base_t = lidar_t
                    else:
                        seg_lidar.append(lidar_t)

                    t_idx += 1
                    if t_idx > (self.length - 1) :
                        if len(seg_lidar) > 5:
                            seg = Segment()
                            seg.lidar = deepcopy(seg_lidar)
                            seg.distance = distance
                            seg.time_interval = time_interval
                            self.segs[seg.lidar[0]] = seg
                        break   
                except IndexError:
                    print("Index {} outof {}, curr seg length {}".format(t_idx, self.length, len(seg_lidar)))
                    return self.segs
                except:
                    raise Exception

        return self.segs

    def try_cut_seg_aeb_mode(self):
        speeds = {}
        gnss_ts_arr = np.array(list(self.gnss_info.keys()))
        veh_ts_arr = np.array(list(self.veh_info.keys()))
        for key in self.lidar_key:
            speeds[key] = None
            gnss_idx = abs(gnss_ts_arr - key).argmin()
            gnss_ts = gnss_ts_arr[gnss_idx]
            # just not confirm gnss status
            # status = int(self.gnss_info[key + offset][status_key])                    
            # if status == 2 or status == 42 or status == 52: # OK 2 in zeer and 42/52 in iflytek
            speed = self.gnss_info[gnss_ts][speed_key]
            if speed == 'na':
                continue
            speed = float(speed)
            if speed == 0.0:
                veh_idx = abs(veh_ts_arr - key).argmin()
                veh_ts = veh_ts_arr[veh_idx]
                str_spd = self.veh_info[veh_ts][veh_speed_key]
                if str_spd == 'na' or str_spd == '':
                    continue
                veh_speed = float(str_spd)
                if veh_speed != 0.0:
                    speed = veh_speed / 3.6 # m/s
            speeds[key] = speed           
        print("Speeds cal over.")

        seg_lidar = []
        distance = 0.0
        time_interval = 0.0        
        t_idx = 0 # aeb模式有多少数据用多少数据
        base_t = self.lidar_key[t_idx]
        while True:            
            if base_t not in speeds:
                t_idx += 1
                if t_idx > len(self.lidar_key) - 1:
                    break
                base_t = self.lidar_key[t_idx]
                continue
            elif speeds[base_t] is None:
                t_idx += 1
                if t_idx > len(self.lidar_key) - 1:
                    break
                base_t = self.lidar_key[t_idx]
                continue
            elif speeds[base_t] < 0.01:
                t_idx += 1
                if t_idx > len(self.lidar_key) - 1:
                    break
                base_t = self.lidar_key[t_idx]
                continue
            else:
                break
        t_idx += 1

        if t_idx > (self.length - 1):
            return self.segs
        
        if self.set_dist == 0 and self.set_intr == 0:
            while True:
                try:
                    lidar_t = self.lidar_key[t_idx]
                    if lidar_t - base_t > 998: # eval distance every second                        
                        gnss_a = abs(gnss_ts_arr - base_t).argmin()
                        gnss_t_a = gnss_ts_arr[gnss_a]
                        long_a = float(self.gnss_info[gnss_t_a][long_key])
                        lati_a = float(self.gnss_info[gnss_t_a][lati_key])
                        gnss_b = abs(gnss_ts_arr - lidar_t).argmin()
                        gnss_t_b = gnss_ts_arr[gnss_b]
                        long_b = float(self.gnss_info[gnss_t_b][long_key])
                        lati_b = float(self.gnss_info[gnss_t_b][lati_key])
                        pa = (lati_a, long_a)
                        pb = (lati_b, long_b)
                        dist = haversine(pa, pb, unit=Unit.METERS)
                        distance += dist
                        base_t = lidar_t
                    seg_lidar.append(lidar_t)                    
                    time_interval += (lidar_t - base_t)
                    t_idx += 1
                    if t_idx > (self.length - 1):
                        seg = Segment()
                        seg.lidar = deepcopy(seg_lidar)
                        seg.distance = distance
                        seg.time_interval = time_interval
                        self.segs[seg.lidar[0]] = seg
                        seg_lidar.clear()
                        distance = 0
                        time_interval = 0
                        break
                except IndexError:
                    print("Index {} outof {}, curr seg length {}".format(t_idx, self.length, len(seg_lidar)))
                    return self.segs
                except:
                    raise Exception
        else:
            print("try_cut_seg_aeb_mode: Index {} outof {}, curr seg length {}".format(t_idx, self.length, len(seg_lidar)))


        return self.segs

class IlfyFrameToSeg:
    def __init__(self, source, seg_path, distance=600, time_interval=60) -> None:
        self.source = source
        self.target = seg_path
        self.distance = distance
        self.time_int = time_interval
        self.clip = source.split('/')[-1]
        if self.clip == '':
            self.clip = source.split('/')[-2]

        self.segs = {}

        self.sensors_idx = {}  # sensor_name : match_header_idx
        self.match_info = {}  # key: lidar time, value: cams' time, 10fps
        self.raw_info = {}  # key: lidar time, value: cams' time, 10fps
        self.match_csv = os.path.join(self.source, "matches.csv")
        self.raw_csv = os.path.join(self.source, "raw.csv")

        self.cams = []
        for item in cam_position:
            _data_path = os.path.join(self.source, item)
            # print(_data_path)
            if os.path.exists(_data_path):                
                self.cams.append(item)
        print("Enable cams {}".format(self.cams))
        self.bpearl_lidars = []
        for item in bpearl_list:
            _data_path = os.path.join(self.source, item)
            # print(_data_path)
            if os.path.exists(_data_path):                
                self.bpearl_lidars.append(item)
        self.inno_lidars = []
        for item in inno_list:
            _data_path = os.path.join(self.source, item)
            if os.path.exists(_data_path):                
                self.inno_lidars.append(item)

        self.gnss_csv = os.path.join(
            self.source, "gnss.csv")
        self.gnss_info = {}  # 100fps

        self.vehicle_csv = os.path.join(
            self.source, "vehicle.csv")
        self.vehicle_info = {}  # 50fps
        self.seg_mode = None
    def parse_match_info(self):
        if not os.path.exists(self.match_csv):
            return 1
        with open(self.match_csv, encoding='utf-8') as fp:
            reader = csv.reader(fp)
            header = next(reader)
            for idx, sensor in enumerate(header):
                self.sensors_idx[sensor] = idx

            for row in reader:
                key = int(float(row[0]))
                val = [int(float(item)) for item in row[1:]]
                self.match_info[int(key)] = val
        if len(self.match_info) < 10:
            print(f"{self.clip} match NONE frame.")
            return 1
        return 0
    
    def parse_raw_info(self):
        if not os.path.exists(self.raw_csv):
            return 1
        with open(self.raw_csv, encoding='utf-8') as fp:
            reader = csv.reader(fp)
            header = next(reader)            

            for row in reader:
                key = int(float(row[0]))
                val = [int(float(item)) for item in row[1:]]
                self.raw_info[int(key)] = val
        return 0

    def parse_gnss_info(self):
        if not os.path.exists(self.gnss_csv):
            return 1

        with open(self.gnss_csv, encoding='utf-8') as fp:
            reader = csv.reader(fp)
            header = next(reader)
            key_idx = {}
            for idx, key in enumerate(header):
                key_idx[key] = idx
            time_key = "utc_time"
            time_id = key_idx[time_key]

            for row in reader:
                item = {}                
                k = row[time_id]
                if k == 'na' or k == time_key:
                    continue
                for _i, v in enumerate(row):
                    if _i > (len(header)-1):
                        continue
                    _k = header[_i]
                    item[_k] = v
                self.gnss_info[int(k)] = item

        return 0

    def parse_vehicle_info(self):
        if not os.path.exists(self.vehicle_csv):
            return 1

        with open(self.vehicle_csv, encoding='utf-8') as fp:
            reader = csv.reader(fp)
            header = next(reader)
            key_idx = {}
            for idx, key in enumerate(header):
                key_idx[key] = idx
            time_key = "utc_time"
            time_id = key_idx[time_key]

            for row in reader:
                item = {}
                k = row[time_id]
                if k == 'na' or k == time_key:
                    continue
                for _i, v in enumerate(row):
                    if _i > (len(header)-1):
                        continue
                    _k = header[_i]
                    item[_k] = v
                self.vehicle_info[int(k)] = item

        return 0

    def cut_seg(self):
        tool = SegTool(
            self.gnss_info, self.vehicle_info, self.match_info, self.distance, self.time_int)
        self.segs = tool.try_cut_seg()        
    
    def cut_seg_hpp(self):
        tool = SegTool(self.gnss_info, self.vehicle_info, self.match_info,
                        self.distance, self.time_int, self.seg_mode)
        self.segs = tool.try_cut_seg_hpp_mode()     

    def cut_seg_luce(self):
        tool = SegTool(
            self.gnss_info, self.vehicle_info, self.match_info, self.distance, self.time_int)
        self.segs = tool.try_cut_seg_luce_mode()

    def cut_seg_by_day_segs(self, spec_timestamp):
        if spec_timestamp is None or len(spec_timestamp) == 0:
            return

        tool = SegTool(
            self.gnss_info, self.vehicle_info, self.raw_info, self.distance, self.time_int)
        
        self.segs = tool.try_cut_seg_with_spec_timestamp(spec_timestamp)

    def cut_seg_aeb(self):
        tool = SegTool(
            self.gnss_info, self.vehicle_info, self.match_info, self.distance, self.time_int)
        self.segs = tool.try_cut_seg_aeb_mode()

    def __call__(self, spec_timestamp=None, seg_mode=None):
        ret = 0
        self.seg_mode = seg_mode
        ret = self.parse_match_info()
        if ret > 0:
            print("match info parse error!")
            return 1
        
        ret = self.parse_raw_info()
        if ret > 0:
            print("raw info parse error!")
            return 1

        ret = self.parse_gnss_info()
        if ret > 0:
            print("gnss info parse error!")
            return 1

        ret = self.parse_vehicle_info()
        if ret > 0:
            print("vehicle info parse error!")
            return 1
        
        enable_cams = []
        for cam in self.cams:
            if cam in self.sensors_idx:
                enable_cams.append(cam)
        enable_bpearls = []
        for bpearl in self.bpearl_lidars:
            if bpearl in self.sensors_idx:
                enable_bpearls.append(bpearl)
        enable_innos = []
        for inno in self.inno_lidars:
            if inno in self.sensors_idx:
                enable_innos.append(inno)

        if seg_mode=="hpp_luce" or seg_mode=="hpp":
            self.cut_seg_hpp()

        elif seg_mode=="luce":
            self.cut_seg_luce()

        elif seg_mode=="aeb":
            self.cut_seg_aeb()

        elif spec_timestamp is None:
            self.cut_seg()        

        else:
            self.cut_seg_by_day_segs(spec_timestamp)
        
        for ts, seg in self.segs.items():
            seg.fill(self.gnss_info, self.match_info, self.raw_info, self.vehicle_info, enable_cams, self.sensors_idx, enable_bpearls, enable_innos)
        return 0

DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

def export_sync_info(seg: Segment, img_ext='jpg'):
    frames = []
    miss_cnt = 0
    enable_cams = seg.enable_cams
    enable_bpearls = seg.enable_bpearls
    enable_innos = seg.enable_innos
    for i, ts in enumerate(seg.lidar):
        sync = {}
        sync['lidar'] = {
            "timestamp": ts,
            "pose": DEFAULT_POSE_MATRIX,
            "path": "lidar/{}.pcd".format(ts),
            "undistort_path": ""
        }
        sync['gnss'] = seg.gnss[i]
        if len(seg.vehicle) > i:
            sync['vehicle'] = seg.vehicle[i]
        sync['images'] = {}
        sync['bpearls'] = {}
        sync['innos'] = {}
        sync['4d_radar'] = {}
        for cam in seg.match.keys():
            cam_ts = seg.match[cam]            
            if cam in enable_cams:
                if cam_ts[i] == 0:
                    if i>15: #前1.5s丢帧不算
                        miss_cnt += 1
                    continue
                sync['images'][cam] = {
                    "timestamp": cam_ts[i],
                    "pose": DEFAULT_POSE_MATRIX,
                    "path": "{}/{}.{}".format(cam, cam_ts[i], img_ext)
                }
            elif cam in enable_bpearls:
                if cam_ts[i] == 0:
                    if i>15: #前1.5s丢帧不算
                        miss_cnt += 1
                    continue
                sync['bpearls'][cam] = {
                    "timestamp": cam_ts[i],
                    "pose": DEFAULT_POSE_MATRIX,
                    "path": "{}/{}.pcd".format(cam, cam_ts[i])
                }
            elif cam in enable_innos:
                if cam_ts[i] == 0:
                    if i>15: #前1.5s丢帧不算
                        miss_cnt += 1
                    continue
                sync['innos'][cam] = {
                    "timestamp": cam_ts[i],
                    "pose": DEFAULT_POSE_MATRIX,
                    "path": "{}/{}.pcd".format(cam, cam_ts[i])
                }
            else:
                if cam == 'sin_radar':
                    sync['4d_radar'][cam] = {
                        "timestamp": cam_ts[i],
                        "pose": DEFAULT_POSE_MATRIX,
                        "path": "{}/{}.pcd".format(cam, cam_ts[i])
                    }

        frames.append(sync)
    return frames, miss_cnt

def export_raw_info(seg: Segment, img_ext='jpg', seg_mode=None):
    frames = []
    enable_cams = seg.enable_cams
    enable_bpearls = seg.enable_bpearls
    enable_innos = seg.enable_innos
    for i, ts in enumerate(seg.lidar):
        raw = {}
        raw['lidar'] = {
            "timestamp": ts,
            "pose": DEFAULT_POSE_MATRIX,
            "path": "lidar/{}.pcd".format(ts),
            "undistort_path": ""
        }
        raw['gnss'] = seg.gnss[i]
        if len(seg.vehicle) > i:
            raw['vehicle'] = seg.vehicle[i]
        raw['images'] = {}
        raw['bpearls'] = {}
        for cam in seg.raw.keys():
            cam_ts = seg.raw[cam]
            if seg_mode == "hpp":
                if cam in ["bpearl_lidar_front", "bpearl_lidar_rear", "bpearl_lidar_left", "bpearl_lidar_right"]:
                    continue
            if cam_ts[i] == 0:
                # miss_cnt += 1
                continue
            if cam in enable_cams:
                raw['images'][cam] = {
                    "timestamp": cam_ts[i],
                    "pose": DEFAULT_POSE_MATRIX,
                    "path": "{}/{}.{}".format(cam, cam_ts[i], img_ext)
                }

            elif cam in enable_bpearls:
                raw['bpearls'][cam] = {
                    "timestamp": cam_ts[i],
                    "pose": DEFAULT_POSE_MATRIX,
                    "path": "{}/{}.pcd".format(cam, cam_ts[i])
                }

        frames.append(raw)
    return frames, 0

def handle_ifly_frame(source, seg_path, car_name, distance=700, time_interval=60, spec_time_list=None, seg_mode=None) -> list:
    inst = IlfyFrameToSeg(source, seg_path, distance, time_interval)
    ret = inst(spec_time_list, seg_mode)
    if ret > 0:
        return []

    clip = inst.clip
    date = clip[:8]

    segs = list(inst.segs.values())
    ret = []

    img_ext = 'jpg'
    if len(segs) == 0:
        return ret
    for cam in segs[0].enable_cams:
        img_dir = os.path.join(source, cam)
        if not os.path.isdir(img_dir):
            continue
        img_ext = os.listdir(img_dir)[0].split(".")[-1]
        break

    for i, seg in enumerate(segs):
        if seg.gnss_info_lost or seg.vehicle_info_lost:
            continue
        gnss = seg.seg_gnss
        vehicle = seg.seg_vehicle
        frames, miss_cnt = export_sync_info(seg, img_ext)
        raws, raw_miss_cnt = export_raw_info(seg, img_ext, seg_mode)
        cameras = seg.enable_cams
        seg_name = "{}_{}_seg{}".format(car_name, clip, i)
        if len(vehicle) == 0:
            print(f"{seg_name} vehicle disable.")

        seg_dict = {}
        
        seg_dict['distance'] = seg.distance
        seg_dict['time_interval'] = seg.time_interval
        seg_dict['related_seg'] = seg.related_seg
        seg_dict['seg_uid'] = seg_name
        seg_dict['date'] = date
        seg_dict['frames_path'] = source
        seg_dict['cameras'] = cameras
        seg_dict['frames'] = frames
        seg_dict['raws'] = raws
        seg_dict['lost_image_num'] = miss_cnt
        seg_dict['raw_pair_lost_image_num'] = raw_miss_cnt

        ret.append((seg_dict, gnss, vehicle))
    return ret

