import json

CAM_DICT_KEY = "idx2cam"
CAM_KEY = "camera_names"
OFFSET_KEY = "offset_map"
LIDAR_KEY = "lidar_name"
OUTPUT_KEY = "output_cameras_name"
LIDAR_TYPE_KEY = "lidar_type"
CHANNEL_KEY = "lidar_msg_channel"
CAR_KEY = "car_name"
DC_KEY = "dc_system_version"
RECON_CAM_KEY = "map_campera_name"
RECON_CAM_ADD_KEY = "map_camera_name_add"
RECON_CAM_LST_KEY = "reconstruct_camera_list"
DC_TYPE_KEY = "dc_type"
SENSOR_INFO_KEY = "sensor_infos"
VEH_INFO_KEY = "veh_info"
GNSS_INFO_KEY = "gnss_info"
BPEARL_LIDAR_KEY = "bpearl_lidar_info"
INNO_LIDAR_KEY = "inno_lidar_info"
VISION_SLOT_KEY = "vision_slot_key"
VISION_SLOT_TIMEINT = "vision_slot_timeinterval"

class CarMeta:
    idx2cam = {}
    cameras = []
    offset_map = {}
    lidar_name = ""
    output_cameras_name = {}
    lidar_type = ""
    lidar_msg_channel = 1
    car_name = ""
    dc_system_version = ""
    dc_type = "fdc"
    offset_detail = {}
    map_campera_name = ""
    map_camera_name_add = ""
    vision_slot = ""
    vision_slot_timeinterval = 0
    reconstruct_camera_list = []
    other_sensors_info = {}
    bpearl_lidars = []
    bpearl_lidar_type = ""
    inno_lidars = []
    bpearl_lidar_type = ""
    radars = []
    radar_type = ""

    def from_json(self, json_file: str):
        with open(json_file, "r") as fp:
            config = json.load(fp)            
            self.dc_system_version = config[DC_KEY]
            if self.dc_system_version == 'zeer':
                self.from_json_zeer(config)
            elif 'iflytek' in self.dc_system_version:
                self.from_json_iflytek(config)
    
    def from_json_iflytek(self, json_config: dict):        
        self.cameras = json_config[CAM_KEY]
        self.offset_detail = json_config[OFFSET_KEY]
        for k,v in self.offset_detail.items():
            base, n = v[0], v[1]
            offset = base + 50 * n
            self.offset_map[k] = offset
        self.lidar_name = json_config[LIDAR_KEY]
        self.lidar_type = json_config[LIDAR_TYPE_KEY]
        self.dc_system_version = json_config[DC_KEY]
        self.dc_type = json_config.get(DC_TYPE_KEY, "fdc")
        self.lidar_msg_channel = json_config[CHANNEL_KEY]
        self.car_name = json_config[CAR_KEY]
        if RECON_CAM_LST_KEY in json_config:
            self.reconstruct_camera_list = json_config[RECON_CAM_LST_KEY]
        else:
            self.reconstruct_camera_list = ['surround_rear_120_8M',
                   'surround_front_120_8M', "surround_front_60_8M"]
        if RECON_CAM_KEY in json_config:
            self.map_camera_name = json_config[RECON_CAM_KEY]
        else:
            self.map_camera_name = "surround_rear_120_8M"
        if RECON_CAM_ADD_KEY in json_config:
            self.map_camera_name_add = json_config[RECON_CAM_ADD_KEY]
        else:
            self.map_camera_name_add = "surround_front_60_8M"
        
        if SENSOR_INFO_KEY in json_config:
            self.other_sensors_info = json_config[SENSOR_INFO_KEY]
            if BPEARL_LIDAR_KEY in self.other_sensors_info:
                if self.other_sensors_info[BPEARL_LIDAR_KEY]['enable'] == 'true':
                    self.bpearl_lidars = self.other_sensors_info[BPEARL_LIDAR_KEY]['positions']
                    self.bpearl_lidar_type = self.other_sensors_info[BPEARL_LIDAR_KEY]['lidar_type']
                    
            if INNO_LIDAR_KEY in self.other_sensors_info:
                if self.other_sensors_info[INNO_LIDAR_KEY]['enable'] == 'true':
                    self.inno_lidars = self.other_sensors_info[INNO_LIDAR_KEY]['positions']
                    self.inno_lidar_type = self.other_sensors_info[INNO_LIDAR_KEY]['lidar_type']

            if "4d_radar" in self.other_sensors_info:
                if self.other_sensors_info["4d_radar"]['enable'] == 'true':
                    self.radars = self.other_sensors_info["4d_radar"]['positions']
                    self.radar_type = self.other_sensors_info["4d_radar"]['lidar_type']

        if VISION_SLOT_KEY in json_config:
            if json_config[VISION_SLOT_KEY] != "null" and json_config[VISION_SLOT_KEY] != "None" and \
                json_config[VISION_SLOT_KEY] != "":
                self.vision_slot = json_config[VISION_SLOT_KEY]
                self.vision_slot_timeinterval = int(json_config[VISION_SLOT_TIMEINT])

if __name__ == '__main__':
    p = '/data_autodrive/auto/calibration/sihao_2xx71/20230220/car_meta.json'
    inst = CarMeta()
    inst.from_json(p)