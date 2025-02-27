import os
import ujson
from loguru import logger
import shutil

json_file_root = "/data_autodrive/users/brli/dev_raw_data/ynchen_recon_fix/hpp_cfg/json/"
cfg_file_root =    "/data_autodrive/users/brli/dev_raw_data/ynchen_recon_fix/hpp_cfg/cfg/"
selected_json = "/data_autodrive/users/brli/dev_raw_data/ynchen_recon_fix/hpp_cfg/need_expand_area.json"
with open(selected_json, "r") as fp:
    selected_segs = ujson.load(fp)
    
# night_segs_json = ""
# with open(night_segs_json, "r") as fp:
#     night_segs = ujson.load(fp)

def check_seg_valid(seg_meta:dict, clip_lane:str, clip_lane_valid:list):
    segid = seg_meta['seg_uid']
    if 'key_frames' not in seg_meta:
        logger.warning(f"{segid} skip. Because no [key_frame] field.")
        return False
    sig_frames = seg_meta.get('key_frames', 0)
    if len(sig_frames) <= 10:
        logger.warning(f"{segid} skip. Because too few key frame.")
        return False
    sig_frames_lost = seg_meta.get('key_frames_lost', 0)
    if sig_frames_lost > 2:
        logger.warning(f"{segid} skip. Because too many key frame lost. [{sig_frames_lost}]")
        return False

    if clip_lane_valid is not None:
        if segid not in clip_lane_valid:
            logger.warning(f"{segid} skip. Because not in clip_lane_valid LIST.")
            return False
    return True
    
def handle_date(car, date, config_file=f"{json_file_root}/chery_24029-20240922.json"):
    specs = selected_segs[car].get(date, [])
    if len(specs) == 0:
        logger.warning(f"{date} skip. Because no specs.")
        return

    if not os.path.exists(config_file):
        logger.error(f"{config_file} not exists")
        return

    with open(config_file, "r") as fp:
        config = ujson.load(fp)

    clip_lane = config['annotation']['clip_lane']
    clip_lane = clip_lane.lower()
    dst_clip_lane = clip_lane.replace("ynchen_hpp_10v", "ynchen_hpp_10v_50m")
    clip_obs = config['annotation']['clip_obstacle']    
    clip_obs = clip_obs.lower()
    dst_clip_obs = clip_obs.replace("ynchen_hpp_10v", "ynchen_hpp_10v_50m")
    os.makedirs(dst_clip_lane, exist_ok=True)
    os.makedirs(dst_clip_obs, exist_ok=True)
    segment_path = config['preprocess']['segment_path']
    segment_path = segment_path.lower()
    clip_lane_check = clip_lane.replace("clip_lane", "clip_lane_check")
    clip_lane_valid_lst = None
    if os.path.exists(clip_lane_check):     
        clip_lane_valid_lst = list()   
        for rgbs in os.listdir(clip_lane_check):
            seg_id, _ = os.path.splitext(rgbs)
            clip_lane_valid_lst.append(seg_id)

    segs = os.listdir(segment_path)
    segs.sort()
    total_seg_cnt = len(segs)
    valid_cnt = 0
    spec_cnt = len(specs)
    for idx, segid in enumerate(segs):
        meta_file = os.path.join(segment_path, segid, "meta.json")
        if not os.path.exists(meta_file):
            continue
        if segid not in specs:
            continue
        
        meta_json = open(meta_file, "r")
        meta = ujson.load(meta_json)
        valid = check_seg_valid(meta, clip_lane, clip_lane_valid_lst)        
        if valid:
            clip_seg_lane = os.path.join(clip_lane, segid)
            clip_seg_obs = os.path.join(clip_obs, segid)
            dst_valid_lane = clip_seg_lane.replace("ynchen_hpp_10v", "ynchen_hpp_10v_50m")
            dst_valid_obs = clip_seg_obs.replace("ynchen_hpp_10v", "ynchen_hpp_10v_50m")
            if os.path.exists(clip_seg_lane) and os.path.exists(clip_seg_obs):
                logger.info(f"...[{idx+1}/{total_seg_cnt}]{car}.{date}.{segid} ===> **valid**")
                shutil.move(clip_seg_lane, dst_valid_lane)
                shutil.move(clip_seg_obs, dst_valid_obs) 
                valid_cnt += 1
                continue           
        
        logger.info(f"...[{idx+1}/{total_seg_cnt}]{car}.{date}.{segid} ===> invalid!")
    logger.info(f"{car}.{date} spec_cnt={spec_cnt}, valid_cnt={valid_cnt}")

def main_with_cfgs():
    logger.add("logs/handle_hpp_50m.log", rotation="50 MB")
    # handle_date("sihao_96tj0", "20240904", config_file=os.path.join(cfg_file_root, "sihao_96tj0-20240904.cfg"))
    # skip_list = [
    #     "chery_24029-20240902.cfg",
    #     "chery_24029-20240903.cfg",
    #     "chery_24029-20240904.cfg",
    #     "chery_48160-20240825.cfg",
    #     "chery_48160-20240827.cfg",
    #     "chery_24029-20240922.json"
    # ]
    
    cfgs = os.listdir(cfg_file_root)
    cfgs.sort()
    for cfg in cfgs:
        if cfg in skip_list:
            continue
        cfg_file = os.path.join(cfg_file_root, cfg)
        infos, _ = os.path.splitext(cfg)
        car, date = infos.split('-')
        handle_date(car, date, config_file=cfg_file)
    
    jsons = os.listdir(json_file_root)
    jsons.sort()
    for cfg in jsons:
        if cfg in skip_list:
            continue
        cfg_file = os.path.join(json_file_root, cfg)
        infos, _ = os.path.splitext(cfg)
        car, date = infos.split('-')
        handle_date(car, date, config_file=cfg_file)

def parse_all_cfgs():
    def parse_cfg(car, date, cfg_file):
        with open(cfg_file, "r") as fp:
            config = ujson.load(fp)
        clip_lane = config['annotation']['clip_lane']
        clip_lane = clip_lane.lower()
        
        clip_obs = config['annotation']['clip_obstacle']    
        clip_obs = clip_obs.lower()
        if "ynchen_hpp_10v" in clip_lane:
            dst_clip_lane = clip_lane.replace("ynchen_hpp_10v", "ynchen_hpp_10v_50m")
            dst_clip_obs = clip_obs.replace("ynchen_hpp_10v", "ynchen_hpp_10v_50m")
        elif "ztwen_apa_hpp_sec" in clip_lane:
            dst_clip_lane = clip_lane.replace("ztwen_apa_hpp_sec", "ztwen_apa_hpp_sec_50m")
            dst_clip_obs = clip_obs.replace("ztwen_apa_hpp_sec", "ztwen_apa_hpp_sec_50m")
        
        os.makedirs(dst_clip_lane, exist_ok=True)
        os.makedirs(dst_clip_obs, exist_ok=True)   
        segment_path = config['preprocess']['segment_path']
        segment_path = segment_path.lower() 
        logger.info(f"{segment_path}#")
        return {
            "clip_lane": clip_lane,
            "clip_obs": clip_obs,
            "segment_path": segment_path,
            "clip_lane_check": clip_lane.replace("clip_lane", "clip_lane_check"),
            "dst_clip_lane": dst_clip_lane,
            "dst_clip_obs": dst_clip_obs
        }
    
    all_cfg_info = dict()

    cfgs = os.listdir(cfg_file_root)
    cfgs.sort()
    for cfg in cfgs:
        cfg_file = os.path.join(cfg_file_root, cfg)
        infos, _ = os.path.splitext(cfg)
        car, date = infos.split('-')
        key = f"{car}.{date}"
        cfg_info = parse_cfg(car, date, cfg_file)
        all_cfg_info[key] = cfg_info
    
    jsons = os.listdir(json_file_root)
    jsons.sort()
    for cfg in jsons:
        cfg_file = os.path.join(json_file_root, cfg)
        infos, _ = os.path.splitext(cfg)
        car, date = infos.split('-')
        key = f"{car}.{date}"
        cfg_info = parse_cfg(car, date, cfg_file)
        all_cfg_info[key] = cfg_info
    return all_cfg_info

def main_with_specs():
    logger.add("logs/handle_hpp_50m.log", rotation="50 MB")
    conf_infos = parse_all_cfgs()
    cars = [
        "sihao_23gc9",
        "sihao_96tj0",
        "sihao_19cp2",
        "sihao_72kx6",
        "sihao_47466",
        "sihao_7xx65",
        "chery_24029",
        "chery_48160"
    ]
    for car in cars:
        car_specs = selected_segs[car]
        for date, segids in car_specs.items():
            conf_key = f"{car}.{date}"
            if conf_key not in conf_infos:
                logger.critical(f"{conf_key} lost config file.")
                continue
            conf_info = conf_infos[conf_key]
            clip_lane_check = conf_info["clip_lane_check"]
            clip_lane_valid_lst = None
            if os.path.exists(clip_lane_check):     
                clip_lane_valid_lst = list()   
                for rgbs in os.listdir(clip_lane_check):
                    _seg_id, _ = os.path.splitext(rgbs)
                    clip_lane_valid_lst.append(_seg_id)
            
            prev_date = int(date) - 1
            prev_conf_key = f"{car}.{prev_date}"
            prev_conf_info = conf_infos.get(prev_conf_key, None)
            if prev_conf_info is not None:                
                prev_clip_lane_check = prev_conf_info["clip_lane_check"]
                if os.path.exists(prev_clip_lane_check):
                    for rgbs in os.listdir(prev_clip_lane_check):
                        _seg_id, _ = os.path.splitext(rgbs)
                        clip_lane_valid_lst.append(_seg_id)
                    
            for seg_id in segids:
                segment_path = conf_info["segment_path"]
                meta_file = os.path.join(segment_path, seg_id, "meta.json")
                if not os.path.exists(meta_file):
                    if prev_conf_info is not None:
                        prev_segment_path = prev_conf_info["segment_path"]
                        meta_file = os.path.join(prev_segment_path, seg_id, "meta.json")
                        if os.path.exists(meta_file):                            
                            curr_conf_info = prev_conf_info
                    else:
                        logger.error(f"{seg_id} miss meta file.")
                        continue
                else:
                    curr_conf_info = conf_info                
                
                clip_lane = curr_conf_info["clip_lane"]
                clip_obs = curr_conf_info["clip_obs"]
                clip_seg_lane = os.path.join(clip_lane, seg_id)
                clip_seg_obs = os.path.join(clip_obs, seg_id)
                if os.path.exists(clip_seg_lane) and os.path.exists(clip_seg_obs):
                    dst_clip_lane = curr_conf_info["dst_clip_lane"]
                    dst_clip_obs = curr_conf_info["dst_clip_obs"]
                    meta_json = open(meta_file, "r")
                    meta = ujson.load(meta_json)
                    valid = check_seg_valid(meta, clip_lane, clip_lane_valid_lst)
                    if valid:                        
                        dst_valid_lane = os.path.join(dst_clip_lane, seg_id)
                        dst_valid_obs = os.path.join(dst_clip_obs, seg_id)                    
                        logger.info(f"...{car}.{date}.{seg_id} // {meta_file} ===> **valid**")
                        shutil.move(clip_seg_lane, dst_valid_lane)
                        shutil.move(clip_seg_obs, dst_valid_obs) 
                        continue 

if __name__ == "__main__":
    main_with_specs()