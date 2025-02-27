import os
import sys
import json
import pandas as pd
import csv
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool

@dataclass
class VehicleReport:
    timestamp: int = 0
    fl_whl_spd: int = 0
    fr_whl_spd: int = 0
    rl_whl_spd: int = 0
    rr_whl_spd: int = 0

    fl_whl_direc: int = 0
    fr_whl_direc: int = 0
    rl_whl_direc: int = 0
    rr_whl_direc: int = 0

    fl_whl_spd_pulse: int = 0
    fr_whl_spd_pulse: int = 0
    rl_whl_spd_pulse: int = 0
    rr_whl_spd_pulse: int = 0

    acc_pedal: float = 0.0
    brake_pedal: float = 0.0
    steering_angle: float = 0.0
    lat_acc: float = 0.0
    lon_acc: float = 0.0
    yaw_rate: float = 0.0
    vehicle_spd: float = 0.0
    gear: int = 0
    odo: float = 0.0


class VehicleReportParse:

    def __init__(self):
        self.last_fl_wheel_speed_pulse: int = 0
        self.last_fr_wheel_speed_pulse: int = 0
        self.last_rl_wheel_speed_pulse: int = 0
        self.last_rr_wheel_speed_pulse: int = 0
        self.last_fl_whl_pulse: int = 0
        self.last_fr_whl_pulse: int = 0
        self.last_rl_whl_pulse: int = 0
        self.last_rr_whl_pulse: int = 0
        self.max_whl_pulse: int = 2046
        self.data_arr = []
        self.vehicle_txt_path = ""

    def parse_vehicle_report(self ,json_data):
        vr = VehicleReport()
        # timestamp 位移,只保留13位
        vr.timestamp = int( json_data["header"]["stamp"] * 1000 )
        vr.fl_whl_spd = int(json_data["fl_wheel_speed"] * 3.6)
        vr.fr_whl_spd = int(json_data["fr_wheel_speed"] * 3.6)
        vr.rl_whl_spd = int(json_data["rl_wheel_speed"] * 3.6)
        vr.rr_whl_spd = int(json_data["rr_wheel_speed"] * 3.6)
        if (json_data["fl_wheel_speed_direction"]==0):
            vr.fl_whl_direc = 1
        elif (json_data["fl_wheel_speed_direction"]==1):
            vr.fl_whl_direc = -1
        else:
            vr.fl_whl_direc = 0
        
        if (json_data["fr_wheel_speed_direction"]==0):
            vr.fr_whl_direc = 1
        elif (json_data["fr_wheel_speed_direction"]==1):
            vr.fr_whl_direc = -1
        else:
            vr.fr_whl_direc = 0

        if (json_data["rl_wheel_speed_direction"]==0):
            vr.rl_whl_direc = 1
        elif (json_data["rl_wheel_speed_direction"]==1):
            vr.rl_whl_direc = -1
        else:
            vr.rl_whl_direc = 0
        
        if (json_data["rr_wheel_speed_direction"]==0):
            vr.rr_whl_direc = 1
        elif (json_data["rr_wheel_speed_direction"]==1):
            vr.rr_whl_direc = -1
        else:
            vr.rr_whl_direc = 0

        vr.fl_whl_spd_pulse = self.last_fl_whl_pulse + ((abs(json_data["fl_wheel_speed_pulse"] - self.last_fl_wheel_speed_pulse) + self.max_whl_pulse)%  self.max_whl_pulse)

        vr.fr_whl_spd_pulse = self.last_fr_whl_pulse + ((abs(json_data["fr_wheel_speed_pulse"] - self.last_fr_wheel_speed_pulse) + self.max_whl_pulse)%  self.max_whl_pulse)

        vr.rl_whl_spd_pulse = self.last_rl_whl_pulse + ((abs(json_data["rl_wheel_speed_pulse"] - self.last_rl_wheel_speed_pulse) + self.max_whl_pulse)%  self.max_whl_pulse)

        vr.rr_whl_spd_pulse = self.last_rr_whl_pulse + ((abs(json_data["rr_wheel_speed_pulse"] - self.last_rr_wheel_speed_pulse) + self.max_whl_pulse)%  self.max_whl_pulse)


        self.last_fl_wheel_speed_pulse = json_data["fl_wheel_speed_pulse"]
        self.last_fr_wheel_speed_pulse = json_data["fr_wheel_speed_pulse"]
        self.last_rl_wheel_speed_pulse = json_data["rl_wheel_speed_pulse"]
        self.last_rr_wheel_speed_pulse = json_data["rr_wheel_speed_pulse"]

        self.last_fl_whl_pulse = vr.fl_whl_spd_pulse
        self.last_fr_whl_pulse = vr.fr_whl_spd_pulse
        self.last_rl_whl_pulse = vr.rl_whl_spd_pulse
        self.last_rr_whl_pulse = vr.rr_whl_spd_pulse

        vr.acc_pedal = json_data["accelerator_pedal_pos"]
        vr.brake_pedal = json_data["brake_pedal_pos"]
        vr.steering_angle = json_data["steering_wheel_angle"]*180/3.141592
        vr.lat_acc = json_data["lat_acceleration"]
        vr.lon_acc = json_data["long_acceleration"]
        vr.yaw_rate = json_data["yaw_rate"]
        vr.gear = json_data["shift_lever_state"]+1
        vr.vehicle_spd = json_data["vehicle_speed"]*3.6
        vr.odo = json_data["total_odometer"]


        return vr
    
    def parse(self, vehicle_txt_path):
        self.vehicle_txt_path = vehicle_txt_path


        pdj=[]
        if os.path.exists(vehicle_txt_path):
            with open(vehicle_txt_path, 'r') as f:
                for line in f:
                    pdj.append(json.loads(line.strip())["msg"])


        for i,j in enumerate(pdj):
            self.data_arr.append(self.parse_vehicle_report(j))
    
    def dumpCsv(self,output_path="vehicle_report.csv"):
        


        csvHeader="utc_time,fl_whl_spd_e_sum,fr_whl_spd_e_sum,rl_whl_spd_e_sum,rr_whl_spd_e_sum,fl_whl_direc,fr_whl_direc,rl_whl_direc,rr_whl_direc,fl_whl_spd,fr_whl_spd,rl_whl_spd,rr_whl_spd,acc_pedal,brake_pedal,steering_angle,lat_acc,lon_acc,yaw_rate,vehicle_spd,gear,odo"
        csvHeaderList=csvHeader.split(",")
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csvHeaderList)
            for i in self.data_arr:
                writer.writerow([i.timestamp,i.fl_whl_spd_pulse,i.fr_whl_spd_pulse,i.rl_whl_spd_pulse,i.rr_whl_spd_pulse,i.fl_whl_direc,i.fr_whl_direc,i.rl_whl_direc,i.rr_whl_direc,i.fl_whl_spd,i.fr_whl_spd,i.rl_whl_spd,i.rr_whl_spd,i.acc_pedal,i.brake_pedal,i.steering_angle,i.lat_acc,i.lon_acc,i.yaw_rate,i.vehicle_spd,i.gear,i.odo])

def export_vehicle_txt_csv(clip, idx):
    print(f"No.{idx} parsing:", clip)
    vehicle_txt = str(clip / "debug_info" / "iflytek_vehicle_service.txt")
    output_csv = str(clip / "vehicle.csv")
    vrp = VehicleReportParse()
    vrp.parse(vehicle_txt)
    vrp.dumpCsv(output_csv)

def main():
    sin_radar_root = Path("/data_cold/abu_zone/autoparse/chery_tiggo9_32694/custom_frame/tguo_radar_data/")
    p = Pool(16)
    idx = 0
    # for date in sin_radar_root.iterdir():
    date = sin_radar_root / "20240926"
    if date.is_dir():
        for clip in date.iterdir():
            idx += 1
            export_vehicle_txt_csv(clip, idx)
            # p.apply_async(export_vehicle_txt_csv, args=(clip , idx,))
    p.close()
    p.join()

if __name__ == "__main__":
    main()
    
