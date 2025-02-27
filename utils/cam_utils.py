cam_position = [
    "surround_front_30_8M",    "surround_front_60_8M",    "surround_front_120_8M", "surround_rear_120_8M",
    "surround_rear_left_120_2M",
    "surround_rear_right_120_2M",
    "surround_front_left_120_2M",
    "surround_front_right_120_2M",
    "around_left_2M", "around_right_2M", "around_front_2M", "around_rear_2M",
    "ofilm_camera_front_100_2M", "ofilm_camera_front_120_8M",
    "ofilm_surround_front_30_8M", "ofilm_surround_front_120_8M", "ofilm_surround_rear_100_2M",
    "ofilm_surround_front_left_100_2M",        "ofilm_surround_front_right_100_2M",
    "ofilm_surround_rear_left_100_2M",        "ofilm_surround_rear_right_100_2M",
    "ofilm_around_front_190_1M", "ofilm_around_rear_190_1M", "ofilm_around_left_190_1M", "ofilm_around_right_190_1M",
    "ofilm_around_front_190_3M", "ofilm_around_rear_190_3M", "ofilm_around_left_190_3M", "ofilm_around_right_190_3M",
    "surround_front_A_120_8M", "surround_front_B_120_8M"
]

bpearl_list = [
    "bpearl_lidar_front",
    "bpearl_lidar_rear",
    "bpearl_lidar_left",
    "bpearl_lidar_right"
]

inno_list = [
    "inno_lidar"
]

tag_list = {
    "light": ["BackLight", "Evening", "Night"],
    "weather": ["Sunny", "Rainy", "Cloudy", "Snow"],
    "changjing": ["PassingBridge", "PassingTunel", "PassingRamp"
                  "CrossRoad", "UpHill", "DownHill",
                  "PrintOnGround", "SpecialLane", "BrokenLane"],
    "roadtype": ["CityRoad", "Highway", "ExpressRoad"],
    "time": ["Morning", "Afternoon", "Evening"]
}

vehicle_keys = ["utc_time", "fl_whl_spd_e_sum", "fr_whl_spd_e_sum", "rl_whl_spd_e_sum", "rr_whl_spd_e_sum",
                "yaw_rate", "fl_whl_direc", "fr_whl_direc", "steering_angle", "rl_whl_direc", "rr_whl_direc", "gear_position"]
zeer_vehicle_key_map = {
    "UtcTime" : "utc_time",
    "Yaw rate[°/s]" : "yaw_rate",
    "Steer angle[°]" : "steering_angle",
    "Gear position" : "gear_position",
    "Speed[KPH]" : "vehicle_spd",
    "Wheel speed (fl)[KPH]": "fl_whl_spd_e_sum",
    "Wheel speed (fr)[KPH]": "fr_whl_spd_e_sum",
    "Wheel speed (rl)[KPH]": "rl_whl_spd_e_sum",
    "Wheel speed (rr)[KPH]": "rr_whl_spd_e_sum"
}

gnss_keys = [ "utc_time","longitude","latitude","altitude","speed","pitch","roll","yaw",
"accx","accy","accz","gyrox","gyroy","gyroz",
"temperature","ve","vn","vu","sat_cnt1","sat_cnt2",
"gps_status"]
zeer_gnss_key_map = {
    "UtcTime" : "utc_time",
    "Location mode": "gps_status",
    "Longitude[°]": "longitude",
    "Latitude[°]": "latitude",
    "Altitude[m]": "altitude",
    "Speed[KPH]": "speed",
    "Orientation[°]": "yaw",
    "Pitch angle[°]": "pitch",
    "Roll angle[°]": "roll",
    "Acceleration-x[m/s2]": "accx",
    "Acceleration-y[m/s2]": "accy",
    "Acceleration-z[m/s2]": "accz",
    "Yaw rate[°/s]": "gyroz",
    "Pitch rate[°/s]": "gyroy",
    "Roll rate[°/s]": "gyrox"
}

gnss_status_map = {
    "52" : "2", # RTK 
    "42" : "3", # RTK float
    "00" : "0", # not work
}