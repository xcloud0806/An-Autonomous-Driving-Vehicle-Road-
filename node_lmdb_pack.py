from multiprocessing import Pool
from tqdm import tqdm
from utils import lmdb_helper, prepare_infos,verify_error_clips
from loguru import logger
from multiprocessing import Event
import os, sys
import json
import shutil
import numpy as np

import tarfile  
import hashlib
import time

from pathlib import Path
def get_file_size_using_pathlib(file_path):
    """
    使用 pathlib 模块的 Path 类获取文件大小
    """
    try:
        path = Path(file_path)
        size = path.stat().st_size
        return size
    except Exception as e:
        logger.error(f"获取文件大小出现错误: {e}")
        return None
def copy_fast(file_src,file_dst,force_copy=False):
    if force_copy:
        shutil.copy(file_src, file_dst)
    else:
        if os.path.exists(file_dst) :
            size_src =  get_file_size_using_pathlib(file_src)
            size_trt =  get_file_size_using_pathlib(file_dst)
            if size_trt is not None and size_src == size_trt:
                pass
            else:
                shutil.copy(file_src, file_dst)
        else:
            shutil.copy(file_src, file_dst)
def get_folder_size_using_pathlib(folder_path):
    total_size = 0
    path = Path(folder_path)
    for file in path.rglob('*'):
        if file.is_file():
            total_size += file.stat().st_size
    if path.is_file():
        total_size += path.stat().st_size
    return total_size
def delete_non_empty_directory(directory_path):
    try:
        if not os.path.exists(directory_path):
            logger.info(f"文件不存在: {directory_path}")
        elif os.path.isdir(directory_path):
            shutil.rmtree(directory_path)
            logger.info(f"已成功删除目录: {directory_path}")
        elif os.path.isfile(directory_path):
            os.remove(directory_path)
            logger.info(f"已成功删除文件: {directory_path}")
        else:
            logger.error(f"delete error {directory_path}")
            raise ValueError
    except Exception as e:
        logger.error(f"删除目录时出现错误: {e}")
        raise e
def save_json(data,path):
    data_str = json.dumps(data)
    with open(path,"w") as f:
        f.write(data_str)

ANNO_INFO_JSON = "annotation.json"
ANNO_INFO_CALIB_KEY = "calib"
ANNO_INFO_INFO_KEY = "clip_info"
ANNO_INFO_LANE_KEY = "lane"
ANNO_INFO_OBSTACLE_KEY = "obstacle"
ANNO_INFO_OBSTACLE_STATIC_KEY = "obstacle_static"
ANNO_INFO_PAIR_KEY = "pair_list"
ANNO_INFO_RAW_PAIR_KEY = "raw_pair_list"
ANNO_INFO_POSE_KEY = "pose_list"
TEST_ROADS_GNSS = "test_roads_gnss_info.json"
DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

LUCE_DEBUG_FILES = [
    "iflytek_camera_perception_debug_info.txt",
    "iflytek_camera_perception_lane_lines.txt",
    "iflytek_camera_perception_objects.txt",
    "iflytek_sensor_pbox_gnss.txt",
    "iflytek_sensor_pbox_imu.txt",
    "iflytek_sensor_gnss.txt",
    "iflytek_sensor_imu.txt",
    "mobileye_camera_perception_lane_lines.txt",
    "mobileye_camera_perception_objects.txt",
    "iflytek_fusion_objects.txt",
    "iflytek_fusion_road_fusion.txt",
    "iflytek_planning_debug_info.txt",
    "iflytek_radar_fm_perception_info.txt",
    "iflytek_radar_rl_perception_info.txt",
    "iflytek_radar_rr_perception_info.txt",
    "iflytek_camera_perception_lane_lines_debug_info.txt",
    "iflytek_camera_perception_lane_topo.txt",
    "iflytek_camera_perception_lane_topo_debug_info.txt",
    "iflytek_prediction_prediction_result.txt",
    "iflytek_camera_perception_traffic_sign_recognition.txt",
]

RADAR_FILES = [
    "iflytek_radar_fm_perception_info.txt",
    "iflytek_radar_rl_perception_info.txt",
    "iflytek_radar_rr_perception_info.txt",
]

def error_callback(e, error_event):  
    # 捕获子进程中的异常并设置事件  
    logger.error(f"Error occurred: {e}")
    error_event.set()  # 通知主进程有异常发生  

def calculate_sha256(file_path):  
    """计算文件的 SHA-256 哈希值"""  
    hash_sha256 = hashlib.sha256()  
    with open(file_path, "rb") as f:  
        for chunk in iter(lambda: f.read(4096), b""):  
            hash_sha256.update(chunk)  
    return hash_sha256.hexdigest()  

def read_file(pcd_img_path):
    try:
        with open(pcd_img_path, "rb") as f:
            file_bytes = f.read()
            if f.__sizeof__()<2:
                logger.error(f"file_bytes 字节数过小: {e}")
                raise ValueError
            return file_bytes
    except Exception as e:
        logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
        raise e   

def compress_directory_with_gzip(source_dir, output_tar_gz):  
    """  
    使用 tar.gz 压缩指定目录，并生成校验文件。  
    :param source_dir: 要压缩的目录路径  
    :param output_tar_gz: 输出的 tar.gz 文件路径  
    """  
    if not os.path.exists(source_dir):  
        logger.error(f"错误：目录 {source_dir} 不存在！")
        raise ValueError(f"错误：目录 {source_dir} 不存在！")  

    try:  
        # 创建 tar.gz 文件
        with tarfile.open(output_tar_gz, "w:gz") as tar:  
            sha256_data = []  
            for root, _, files in os.walk(source_dir):  
                for file in files:  
                    file_path = os.path.join(root, file)  
                    arcname = os.path.relpath(file_path, source_dir)  # 相对路径  
                    tar.add(file_path, arcname)  
                    # 计算 SHA-256 并记录  
                    file_sha256 = calculate_sha256(file_path)  
                    sha256_data.append(f"{arcname} {file_sha256}")
            
            # 写入校验文件  
            checksum_file = "checksum.sha256"  
            with open(checksum_file, "w") as f:  
                f.write("\n".join(sha256_data))  
            tar.add(checksum_file, checksum_file)  # 将校验文件添加到 tar.gz 包中  
            delete_non_empty_directory(checksum_file)  # 删除临时校验文件  

        logger.info(f"压缩完成！输出文件：{output_tar_gz}")  
    except Exception as e:  
        logger.error(f"压缩过程中出现错误：{e}")
        delete_non_empty_directory(output_tar_gz)
        raise e

def find_closest_timestamps(timestamp_list, target):  
    """找到与目标时间戳最近的值"""  
    closest = min(timestamp_list, key=lambda x: abs(x - target))  
    return closest 
def compress_sinradar_with_gzip(source_dir, output_tar_gz, meta):  
    """  
    使用 tar.gz 压缩指定目录，并生成校验文件。  
    :param source_dir: 要压缩的目录路径  
    :param output_tar_gz: 输出的 tar.gz 文件路径  
    """  
    if not os.path.exists(source_dir):  
        logger.error(f"错误：目录 {source_dir} 不存在！")
        raise ValueError(f"目录 {source_dir} 不存在！")

    try:  
        frames = meta["frames"]
        first_frame = frames[0]
        last_frame = frames[-1]

        first_lidar_ts = first_frame['lidar']['timestamp']
        last_lidar_ts = last_frame['lidar']['timestamp']
        sinradar_ts_list = os.listdir(source_dir)
        # sinradar_ts_list.sort()
        sinradar_ts_list = [int(filename[:-4]) for filename in sinradar_ts_list if filename.endswith('.pcd')] 

        sinradar_first_ts = find_closest_timestamps(sinradar_ts_list, first_lidar_ts)  
        sinradar_last_ts = find_closest_timestamps(sinradar_ts_list, last_lidar_ts) 

        # 创建 tar.gz 文件
        with tarfile.open(output_tar_gz, "w:gz") as tar:  
            sha256_data = []  
            for root, _, files in os.walk(source_dir):  
                for file in files:
                    if int(file[:-4]) < sinradar_first_ts or int(file[:-4]) > sinradar_last_ts:
                        continue
                    file_path = os.path.join(root, file)  
                    arcname = os.path.relpath(file_path, source_dir)  # 相对路径  
                    tar.add(file_path, arcname)  
                    # 计算 SHA-256 并记录  
                    file_sha256 = calculate_sha256(file_path)  
                    sha256_data.append(f"{arcname} {file_sha256}")
            
            # 写入校验文件  
            checksum_file = "checksum.sha256"  
            with open(checksum_file, "w") as f:  
                f.write("\n".join(sha256_data))  
            tar.add(checksum_file, checksum_file)  # 将校验文件添加到 tar.gz 包中  
            delete_non_empty_directory(checksum_file)  # 删除临时校验文件  
            logger.info(f"{output_tar_gz} total has file {len(sha256_data)}")
        logger.info(f"压缩完成！输出文件：{output_tar_gz}")  
    except Exception as e:
        logger.error(f"错误：目录 {source_dir} 不存在！")
        delete_non_empty_directory(output_tar_gz)
        raise e 


def write_lmdb(submit_path, lmdb_bytes, frame_cnt, key_list, total_cnt, lmdb_name ):
        try:
            cache_size = 50 * 1024 * 1024 * 1024
            _lmdb_data_path = submit_path + f"/{lmdb_name}"
            if os.path.exists(_lmdb_data_path):
                os.system(f"rm -rf {_lmdb_data_path}")
            lmdb_handle = lmdb_helper.LmdbHelper(_lmdb_data_path, tmp_size=cache_size)
            lmdb_info = {}    

            lmdb_handle.write_datas(lmdb_bytes)
            lmdb_size, lmdb_hash = lmdb_handle.cacl_hash(submit_path +  f"/{lmdb_name}")
            keys = lmdb_handle.get_all_keys()
            print(f"{submit_path} total_cnt:{total_cnt}, keys len:{len(keys)}")
            lmdb_info["lmdb_size"] = lmdb_size
            lmdb_info["lmdb_hash"] = lmdb_hash
            lmdb_info["frame_cnt"] = frame_cnt
            lmdb_info["total_cnt"] = total_cnt
            lmdb_info["key_list"] = key_list
            with open(submit_path + f"/{lmdb_name}_info.json", "w") as f:
                json.dump(lmdb_info, f)
        except Exception as e:
            logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
            delete_non_empty_directory(_lmdb_data_path)
            raise e
def dump_numpy(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

    
def gen_lmdb_bytes_xingche(meta, frame_cnt, start_idx, end_idx,enable_bpearls,enable_cams):
    frames = meta["frames"]
    total_cnt = 0
    key_list = []
    seg_frame_path = meta["frames_path"]
    seg_copy_frames_limits = {}
    lmdb_bytes = {}
    for i, f in enumerate(frames):
        # 拷贝雷达数据
        if i < start_idx or i >= end_idx:
            continue
        lidar_src = f['lidar']['path']
        if not os.path.exists(lidar_src):
            lidar_src = os.path.join(seg_frame_path, f["lidar"]["path"])
        if 'lidar' not in frame_cnt:
            frame_cnt['lidar'] = 0
        lidar_bytes = read_file(lidar_src)
        lidar_key = f"lidar_{f['lidar']['timestamp']}"
        lmdb_bytes[lidar_key] = lidar_bytes
        frame_cnt["lidar"] += 1
        total_cnt += 1
        key_list.append(lidar_key)
        images = f["images"]
        cam_keys = list(images.keys())
        bpearls = {}
        bpearl_keys = []
        innos = {}
        radars = {}
        if len(enable_bpearls) > 0:
            if "bpearls" in f:
                bpearls = f["bpearls"]
                bpearl_keys = list(bpearls.keys())
            if "innos" in f:
                innos = f["innos"]
                bpearl_keys.extend(list(innos.keys()))
            if "4d_radar" in f:
                radars = f["4d_radar"]
                bpearl_keys.extend(list(radars.keys()))
        for cam in enable_cams:
            if cam not in cam_keys:
                continue
            if cam not in frame_cnt:
                frame_cnt[cam] = 0
            if cam not in seg_copy_frames_limits:
                seg_copy_frames_limits[cam] = []
            img_path = images[cam]["path"]
            img_pre, ext = os.path.splitext(img_path)
            img_ts = images[cam]["timestamp"]
            if img_ts == 0:
                continue
            seg_copy_frames_limits[cam].append(img_ts)
        for b in enable_bpearls:
            if b not in bpearl_keys:
                continue
            if b in bpearls:
                b_path = bpearls[b]["path"]
                b_ts = bpearls[b]['timestamp']
                if b_ts == 0:
                    continue
                b_key = f"{b}_{bpearls[b]['timestamp']}"
            elif b in innos:
                b_path = innos[b]["path"]
                b_ts = innos[b]['timestamp']
                if b_ts == 0:
                    continue
                b_key = f"{b}_{innos[b]['timestamp']}"
            # elif b in radars:
            #     b_path = radars[b]["path"]
            #     b_ts = radars[b]['timestamp']
            #     if b_ts == 0:
            #         continue
            #     b_key = f"{b}_{radars[b]['timestamp']}"
            else:
                continue
            b_src = os.path.join(seg_frame_path, b_path)
            b_bytes = read_file(b_src)                
            lmdb_bytes[b_key] = b_bytes
            key_list.append(b_key)
            total_cnt += 1
    if 'sin_radar' in bpearl_keys:
        sin_root = os.path.join(seg_frame_path, 'sin_radar')
        lidar_ts_min = int(frames[0]['lidar']['timestamp'])
        lidar_ts_max = int(frames[-1]['lidar']['timestamp'])
        frame_cnt["sin_radar"] = 0
        sin_radar_files = os.listdir(sin_root)
        sin_radar_files.sort()
        for sin_radar_f in sin_radar_files:
            sin_ts = int(os.path.splitext(sin_radar_f)[0])
            if sin_ts < lidar_ts_max and sin_ts > lidar_ts_min:
                sin_src = os.path.join(sin_root, sin_radar_f)
                sin_bytes = read_file(sin_src)
                sin_key = f"sin_radar_{sin_ts}"
                lmdb_bytes[sin_key] = sin_bytes
                frame_cnt["sin_radar"] += 1
                total_cnt += 1
                key_list.append(sin_key)
    for cam in enable_cams:
        if cam not in seg_copy_frames_limits.keys():
            continue
        cam_ts_min = seg_copy_frames_limits[cam][0] - 1
        cam_ts_max = seg_copy_frames_limits[cam][-1] + 1
        cam_frame_path = os.path.join(seg_frame_path, cam)
        cam_frames = os.listdir(cam_frame_path)
        cam_frames.sort()
        for i, f in enumerate(cam_frames):
            img_pre, ext = os.path.splitext(f)
            img_ts = int(img_pre)
            if img_ts < cam_ts_max and img_ts > cam_ts_min:
                img_src = os.path.join(seg_frame_path, f"{cam}/{f}")
                img_bytes = read_file(img_src)
                img_key = f"{cam}_{img_ts}"
                lmdb_bytes[img_key] = img_bytes
                frame_cnt[cam] += 1
                total_cnt += 1
                key_list.append(img_key)
    return lmdb_bytes, total_cnt, key_list, frame_cnt


def pack_and_verify(frame_cnt,_lmdb_data_path,
                    lmdb_name,submit_path,
                    meta,start_idx,frame_total_cnt,
                    enable_bpearls,enable_cams):
    #打包前校验
    if os.path.exists(_lmdb_data_path):
        # 验证是否可读
        try:
            verify_error_clips.read_single_clips(submit_path+"/clip_info.json",200) # 是否可读校验
        except:
            delete_non_empty_directory(_lmdb_data_path) 
        # 验证字节大小
        lmdb_size = get_folder_size_using_pathlib(_lmdb_data_path)
        if lmdb_size<=1024*1024*1: #<=1Mb 字节过小重打包
            lmdb_bytes, total_cnt, key_list, frame_cnt = gen_lmdb_bytes_xingche(meta, frame_cnt, start_idx, frame_total_cnt,enable_bpearls,enable_cams)
            write_lmdb(submit_path, lmdb_bytes, frame_cnt, key_list, total_cnt, lmdb_name)
        else:
            pass
    else:
        lmdb_bytes, total_cnt, key_list, frame_cnt = gen_lmdb_bytes_xingche(meta, frame_cnt, start_idx, frame_total_cnt,enable_bpearls,enable_cams)
        write_lmdb(submit_path, lmdb_bytes, frame_cnt, key_list, total_cnt, lmdb_name)
    # 打包后校验
    ## 验证字节大小
    lmdb_size = get_folder_size_using_pathlib(_lmdb_data_path)
    if lmdb_size<=1024*1024*1: #<=1Mb
        logger.error(f"lmdb_size <=1MB")
        raise ValueError
    if lmdb_name == "lmdb":
        ## 验证是否可读
        pack_right = False
        try:
            verify_error_clips.read_single_clips(submit_path+"/clip_info.json",200) # 是否可读校验
            pack_right=True
        except:
            # 不可读清除目录
            pack_right=False
            delete_non_empty_directory(_lmdb_data_path)
            logger.error(f"lmdb cannot read")
            raise ValueError 



def gen_seg_lmdb(
    meta: dict,
    enable_cams: list,
    enable_bpearls: list,
    submit_path: str,
    seg_frame_path:str,
    max_frame=1200
):
    submit_data_path = submit_path
    sin_radar_path = os.path.join(seg_frame_path, "sin_radar")
    sin_radar_tar_path = os.path.join(submit_data_path, "sin_radar.tar.gz")
    logger.info(f">>>> Prepare tar {sin_radar_path}")
    if os.path.exists(sin_radar_tar_path):
        delete_non_empty_directory(sin_radar_tar_path)
    if os.path.exists(sin_radar_path):
        compress_sinradar_with_gzip(sin_radar_path, sin_radar_tar_path, meta)
    vehicle_config_path = os.path.join(seg_frame_path, "vehicle_config")
    vehicle_config_tar_path = os.path.join(submit_data_path, "vehicle_config.tar.gz")
    logger.info(f">>>> Prepare tar {vehicle_config_path}")
    if os.path.exists(vehicle_config_tar_path):
        delete_non_empty_directory(vehicle_config_tar_path)
    if os.path.exists(vehicle_config_path):
        compress_directory_with_gzip(vehicle_config_path, vehicle_config_tar_path)

    
    
    logger.info(f"Prepare {meta['seg_uid']} to Pack {submit_path}")    
    
    frame_total_cnt = len(meta['frames'])
    if frame_total_cnt < (max_frame + 1):
        frame_cnt = {}
        lmdb_name = "lmdb"
        _lmdb_data_path = submit_path + f"/{lmdb_name}"
        start_idx=0
        pack_and_verify(frame_cnt,_lmdb_data_path,lmdb_name,submit_path,meta,start_idx,frame_total_cnt,enable_bpearls,enable_cams)
        # gen_lmdb_bytes_xingche(meta, frame_cnt, 0, frame_total_cnt,enable_bpearls,enable_cams)
    else:        
        total_lmdb_clip_cnt = int(frame_total_cnt / max_frame) + 1
        logger.info(f"\tcut {meta['seg_uid']} to {total_lmdb_clip_cnt} LMDB_PACKS.")
        for idx in range(total_lmdb_clip_cnt):
            _lmdb_name = f"lmdb_{idx}"
            logger.info(f"\t{_lmdb_name} start......")
            start_idx = idx * max_frame
            end_idx = (idx + 1) * max_frame if (idx + 1) * max_frame < frame_total_cnt else frame_total_cnt
            frame_cnt = {}
            _lmdb_data_path = submit_path + f"/{_lmdb_name}"
            pack_and_verify(frame_cnt,_lmdb_data_path,_lmdb_name,submit_path,meta,start_idx,end_idx,enable_bpearls,enable_cams)
        ## 分块的全打完再 验证是否可读
        pack_right = False
        try:
            verify_error_clips.read_single_clips(submit_path+"/clip_info.json",200) # 是否可读校验
            pack_right=True
        except:
            # 不可读清除目录
            pack_right=False
            delete_non_empty_directory(_lmdb_data_path)
            logger.error(f"{_lmdb_name} cannot read")
            raise ValueError 
        
    size_dict = {}
    [size_dict.update({m:get_folder_size_using_pathlib(os.path.join(submit_path,m))}) for m in os.listdir(submit_path)]
    file_size_path = os.path.join(submit_path,"filesize.json")
    save_json(size_dict,file_size_path)
    logger.info(f"Pack {meta['seg_uid']} to {submit_path} done")  

def gen_hpp_seg_lmdb(meta:dict, submit_path:str, enable_cams:list, enable_bpearls:list, max_frame:int=1000):
    def gen_lmdb_bytes(meta:dict, frame_cnt:dict, start_idx:int, end_idx:int):
        frames = meta["frames"]
        sig_frames = meta['key_frames']
        total_cnt = 0
        key_list = []
        seg_frame_path = meta["frames_path"]
        
        lmdb_bytes = {}
        for i in range(start_idx, end_idx):
            sig = sig_frames[i]
            frame_idx = sig['frame_idx']
            f = frames[frame_idx]
            lidar_src = f['lidar']['path']
            if not os.path.exists(lidar_src):
                lidar_src = os.path.join(seg_frame_path, f["lidar"]["path"])
            if 'lidar' not in frame_cnt:
                frame_cnt['lidar'] = 0
            lidar_bytes = read_file(lidar_src)
            lidar_key = f"lidar_{f['lidar']['timestamp']}"
            lmdb_bytes[lidar_key] = lidar_bytes
            frame_cnt["lidar"] += 1
            total_cnt += 1
            key_list.append(lidar_key)

            if len(enable_bpearls) > 0:
                bpearls = f["bpearls"]
                bpearl_keys = list(bpearls.keys())
                for b in bpearl_keys:
                    if b not in frame_cnt:
                        frame_cnt[b] = 0
                    if b not in bpearls:
                        continue
                    b_path = bpearls[b]["path"]
                    b_ts = bpearls[b]['timestamp']
                    if b_ts == 0:
                        continue
                    b_key = f"{b}_{bpearls[b]['timestamp']}"
                    b_src = os.path.join(seg_frame_path, b_path)
                    b_bytes = read_file(b_src)                
                    lmdb_bytes[b_key] = b_bytes
                    key_list.append(b_key)
                    total_cnt += 1
                    frame_cnt[b] += 1
            
            images = f["images"]
            for cam in enable_cams:
                if cam not in frame_cnt:
                    frame_cnt[cam] = 0
                if cam not in images:
                    continue
                img_path = images[cam]["path"]
                img_ts = images[cam]["timestamp"]
                img_src = os.path.join(seg_frame_path, img_path)
                img_bytes = read_file(img_src)
                img_key = f"{cam}_{img_ts}"
                lmdb_bytes[img_key] = img_bytes
                key_list.append(img_key)
                frame_cnt[cam] += 1
                total_cnt += 1
        return lmdb_bytes, total_cnt, key_list, frame_cnt

    logger.info(f"Prepare {meta['seg_uid']} to Pack {submit_path}")   
    func_write_lmdb = write_lmdb
    frame_cnt = {}
    frame_total_cnt = len(meta['key_frames'])
    lmdb_bytes, total_cnt, key_list, frame_cnt = gen_lmdb_bytes(meta, frame_cnt, 0, frame_total_cnt)
    func_write_lmdb(submit_path, lmdb_bytes, frame_cnt, key_list, total_cnt, "lmdb")

    logger.info(f"Pack {meta['seg_uid']} to {submit_path} done") 

def check_cnt_result(res:dict):
    if 'lidar' not in res:
        return False
    
    base_cnt = res['lidar']
    for key in list(res.keys()):
        if key == 'lidar':
            continue

        cnt = res[key]
        if abs(cnt - base_cnt) > 4:
            logger.warning(f"!!!!!! {key} - lidar check_cnt_result :{abs(cnt - base_cnt)} > 4")
            return False
    
    return True

def check_with_lmdb_info(seg_lmdb_path):
    keys = []
    lmdb_json_files = [item for item in os.listdir(seg_lmdb_path) if item.startswith('lmdb') and item.endswith('json')]
    for lmdb_json_file in lmdb_json_files:
        lmdb_json = os.path.join(seg_lmdb_path, lmdb_json_file)
        with open(lmdb_json, 'r') as fp:
            lmdb_info = json.load(fp)
            _keys = lmdb_info['key_list']
            keys.extend(_keys)
    info_json = os.path.join(seg_lmdb_path, "clip_info.json")
    with open(info_json, "r") as f:
        info = json.load(f)
    sensors =  info['calib']['sensors']
    found  = False
    for s in sensors:
        if "ofilm" in  s and "front" in s:
            found = True
    if not found:
        logger.error(f"There is no OFILM Cam in {seg_lmdb_path}.")
        return False
    
    pair_keys = list(info["pair_list"][5].keys())
    for s in sensors:
        if s not in pair_keys and s not in ["gnss", "vehicle"]:
            logger.error("{} in {} is not a valid sensor".format(s, seg_lmdb_path))
            return False

    pair_list =  info["pair_list"]
    pair_exist_cnt = {}
    for pair in pair_list:
        for key in pair.keys():            
            assert key in pair, "{} is missing".format(key)
            if "around" not in key:
                #assert key in sensors, "{} is not a valid sensor".format(key)
                if key not in sensors:
                    logger.error("{} is not a valid sensor".format(key))
                    pair_exist_cnt[key] = 0
                    continue
            else:
                continue
            assert pair[key] is not None, "{} is missing".format(pair[key])
            if key not in pair_exist_cnt:
                pair_exist_cnt[key] = 0

            lmdb_data_key = f"{key}_{pair[key]}"
            if lmdb_data_key in keys:
                pair_exist_cnt[key] += 1
    logger.info("{} Pair exist count: {}".format(seg_lmdb_path, pair_exist_cnt))
    return check_cnt_result(pair_exist_cnt)

def multi_process_error_callback(error):
    # get the current process
    process = os.getpid()
    # report the details of the current process
    print(f"Callback Process: {process}, Exeption {error}", flush=True)

def node_main(run_config):
    seg_config = run_config["preprocess"]
    seg_mode =  seg_config["seg_mode"]
    frames_path = seg_config['frames_path']
    tgt_seg_path = seg_config["segment_path"]
    car_name = seg_config['car']
    rec_cfg = run_config["reconstruction"]
    skip_reconstruct = False
    if rec_cfg['enable'] != "True":
        skip_reconstruct = True

    spec_clips = seg_config.get("spec_clips", None)    
    pre_anno_cfg = run_config['annotation']
    test_road_gnss_file = f"{pre_anno_cfg['test_gnss_info']}"
    
    odometry_mode = run_config["odometry"]["pattern"]

    deploy_cfg = run_config["deploy"]
    src_deploy_root = deploy_cfg["clip_submit_data"]
    subfix = deploy_cfg['data_subfix']
    anno_path = os.path.join(src_deploy_root, subfix)

    seg_root_path = tgt_seg_path
    if not os.path.exists(seg_root_path):
        logger.error(f"{seg_root_path} NOT Exist...")
        sys.exit(1)
    seg_names = os.listdir(seg_root_path) #TODO
    seg_names.sort()
    error_event = Event()
    num_thread=4
    pool = Pool(processes=num_thread)
    try:
        for segid in seg_names:
            if spec_clips is not None:
                go_on = False
                for clip in spec_clips:
                    if clip in segid:
                        go_on = True
                        break
                if not go_on:
                    continue

            seg_path = os.path.join(seg_root_path, segid)
            meta_file = os.path.join(seg_root_path, segid, "updated_meta.json")
            if not os.path.exists(meta_file):
                meta_file = os.path.join(seg_root_path, segid, "meta.json")
            if not os.path.exists(meta_file):
                logger.error(f"{meta_file} Not Exists.")
                raise ValueError(f"{meta_file} Not Exists.")

            reconstruct_path = os.path.join(seg_root_path, segid, "multi_reconstruct")
            if not os.path.exists(reconstruct_path):
                reconstruct_path = os.path.join(seg_root_path, segid, "reconstruct")

            rgb_file = []
            if os.path.exists(reconstruct_path):
                files = os.listdir(reconstruct_path)
                for f in files:
                    if f.endswith("jpg") or f.endswith("jpeg") or f.endswith("png"):
                        r_file = os.path.join(reconstruct_path, f)
                        rgb_file.append(r_file)
                    if f.endswith("npy") or f.endswith("npz"):
                        npy_file = os.path.join(reconstruct_path, f)
                        rgb_file.append(npy_file)

            meta_json = open(meta_file, "r")
            meta = json.load(meta_json)
            seg_frame_path = meta['frames_path']
            first_lidar_pose = np.array(meta['frames'][0]['lidar']['pose']).astype(np.float32)
            dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
            if (first_lidar_pose==dft_pose_matrix).all():
                logger.warning(f"{segid} not selected .")
                continue
            
            logger.info("Commit segment {}.".format(segid))
                
            enable_cams = meta["cameras"]
            enable_bpearls = []
            if "other_sensors_info" in meta:
                _info = meta["other_sensors_info"]
                if "bpearl_lidar_info" in _info:
                    if _info["bpearl_lidar_info"]["enable"] == "true":
                        bpearls = _info["bpearl_lidar_info"]["positions"]
                        bpearl_path_exist_count = 0
                        for bp in bpearls:
                            bpearl_path = os.path.join(seg_frame_path, bp)
                            if os.path.exists(bpearl_path):
                                bpearl_path_exist_count += 1
                            else:
                                logger.warning(f"warning car_meta.json enabled bpearl_lidar, actually {bpearl_path} not exist.")
                        if bpearl_path_exist_count == len(bpearls):
                            enable_bpearls = _info["bpearl_lidar_info"]["positions"]

                if "inno_lidar_info" in _info:
                    if _info["inno_lidar_info"]["enable"] == "true":
                        enable_bpearls.extend(_info["inno_lidar_info"]["positions"])
                if "4d_radar" in _info:
                    if _info["4d_radar"]["enable"] == "true":
                        enable_bpearls.extend(_info["4d_radar"]["positions"])

            meta_json.close()
            submit_data_path = os.path.join(anno_path, segid)
            def prepare_copy(file_name, rgb=False):
                file_src = os.path.join(seg_path, file_name)
                file_dst = os.path.join(submit_data_path, file_name)
                if rgb:
                    file_src = file_name
                    _f = os.path.basename(file_name)
                    file_dst = os.path.join(submit_data_path, _f)
                try:
                    copy_fast(file_src,file_dst)
                except Exception as e:
                    delete_non_empty_directory(file_dst)
                    raise e
            if not os.path.exists(submit_data_path):
                os.makedirs(submit_data_path, mode=0o775, exist_ok=True)
            clip_info = prepare_infos(meta, enable_cams, enable_bpearls, seg_path, test_road_gnss_file)
            if seg_mode == 'test' or seg_mode == 'luce' or seg_mode == 'hpp_luce' or seg_mode == 'aeb':
                clip_info['datasets'] = 'test'

            debug_info_path = os.path.join(seg_frame_path, "debug_info")
            for filename in os.listdir(debug_info_path):
                    file_src = os.path.join(debug_info_path, filename)
                    if os.path.exists(file_src) and filename.endswith('.txt'):
                        file_dst = os.path.join(submit_data_path, filename)
                        try:
                            copy_fast(file_src,file_dst)
                        except Exception as e:
                            delete_non_empty_directory(file_dst)
                            raise e

            try:
                with open(os.path.join(submit_data_path, "clip_info.json"), "w") as fp:
                    ss = json.dumps(clip_info, ensure_ascii=False, default=dump_numpy)
                    fp.write(ss)
                    
                prepare_copy("gnss.json")
                prepare_copy("vehicle.json")


                for r in rgb_file:
                    prepare_copy(r, True)
            except Exception as e:
                logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
                raise e
            
            pool.apply_async(
                gen_seg_lmdb,
                args=(
                    meta,
                    enable_cams,
                    enable_bpearls,
                    submit_data_path,
                    seg_frame_path,
                ),
                error_callback=lambda e: error_callback(e, error_event)
            )        
        pool.close()
        pool.join()

        while True:  
            if error_event.is_set():  # 如果检测到异常  
                logger.error("Terminating all processes due to an error.", file=sys.stderr)  
                pool.terminate()  # 终止所有子进程  
                sys.exit(1)  # 退出主程序 
            break  # 如果没有异常，跳出循环 

        logger.info(f">>>> {car_name}.{subfix} Prepare LMDB Done.")
    except Exception as e:  
        # 捕获主进程中的异常  
        logger.error(f"Error occurred in main process: {e}")  
        pool.terminate()  # 终止所有子进程  
        pool.join()  # 等待子进程终止  
        sys.exit(1)  # 退出主程序

def node_main_hpp(run_config) -> None:
    seg_config = run_config["preprocess"]
    seg_mode =  seg_config["seg_mode"]
    frames_path = seg_config['frames_path']
    tgt_seg_path = seg_config["segment_path"]
    car_name = seg_config['car']
    rec_cfg = run_config["reconstruction"]
    skip_reconstruct = False
    if rec_cfg['enable'] != "True":
        skip_reconstruct = True

    spec_clips = seg_config.get("spec_clips", None)
    pre_anno_cfg = run_config['annotation']
    test_road_gnss_file = f"{pre_anno_cfg['test_gnss_info']}"
    
    odometry_mode = run_config["odometry"]["pattern"]

    deploy_cfg = run_config["deploy"]
    src_deploy_root = deploy_cfg["clip_submit_data"]
    subfix = deploy_cfg['data_subfix']
    anno_path = os.path.join(src_deploy_root, subfix)

    clip_lane = pre_anno_cfg['clip_lane']
    clip_lane_check = clip_lane.replace("clip_lane", "clip_lane_check")
    specs = list()
    if os.path.exists(clip_lane_check):
        for rgbs in os.listdir(clip_lane_check):
            seg_id, _ = os.path.splitext(rgbs)
            specs.append(seg_id)

    seg_root_path = tgt_seg_path
    if not os.path.exists(seg_root_path):
        logger.error(f"{seg_root_path} NOT Exist...")
        sys.exit(1)
    seg_names = os.listdir(seg_root_path)
    seg_names.sort()

    pool = Pool(processes=4)
    error_event = Event()
    try:
        for segid in seg_names:
            if len(specs) > 0 and segid not in specs:
                continue

            if spec_clips is not None:
                go_on = False
                for clip in spec_clips:
                    if clip in segid:
                        go_on = True
                        break
                if not go_on:
                    continue

            seg_path = os.path.join(seg_root_path, segid)
            meta_file = os.path.join(seg_root_path, segid, "updated_meta.json")
            if not os.path.exists(meta_file):
                meta_file = os.path.join(seg_root_path, segid, "meta.json")
            if not os.path.exists(meta_file):
                continue

            meta_json = open(meta_file, "r")
            meta = json.load(meta_json)
            seg_frame_path = meta['frames_path']
            first_lidar_pose = np.array(meta['frames'][0]['lidar']['pose']).astype(np.float32)
            dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
            if (first_lidar_pose==dft_pose_matrix).all():
                logger.warning(f"{segid} not selected .")
                continue
            
            logger.info("Commit segment {}.".format(segid))
                
            enable_cams = meta["cameras"]
            enable_bpearls = []
            if "other_sensors_info" in meta:
                _info = meta["other_sensors_info"]
                if "bpearl_lidar_info" in _info:
                    if _info["bpearl_lidar_info"]["enable"] == "true":
                        bpearls = _info["bpearl_lidar_info"]["positions"]
                        bpearl_path_exist_count = 0
                        for bp in bpearls:
                            bpearl_path = os.path.join(seg_frame_path, bp)
                            if os.path.exists(bpearl_path):
                                bpearl_path_exist_count += 1
                            else:
                                logger.warning(f"warning car_meta.json enabled bpearl_lidar, actually {bpearl_path} not exist.")
                        if bpearl_path_exist_count == len(bpearls):
                            enable_bpearls = _info["bpearl_lidar_info"]["positions"]

            meta_json.close()
            submit_data_path = os.path.join(anno_path, segid)
            def prepare_copy(file_name, rgb=False):
                file_src = os.path.join(seg_path, file_name)
                file_dst = os.path.join(submit_data_path, file_name)
                if rgb:
                    file_src = file_name
                    _f = os.path.basename(file_name)
                    file_dst = os.path.join(submit_data_path, _f)

                try:
                    copy_fast(file_src,file_dst)
                except Exception as e:
                    delete_non_empty_directory(file_dst)
                    raise e

            if not os.path.exists(submit_data_path):
                os.makedirs(submit_data_path, mode=0o775, exist_ok=True)
            clip_info = prepare_infos(meta, enable_cams, enable_bpearls, seg_path, test_road_gnss_file)
            if seg_mode == 'test' or seg_mode == 'luce' or seg_mode == 'hpp_luce' or seg_mode == 'aeb':
                clip_info['datasets'] = 'test'
            
            if 'hpp' in seg_mode and 'key_frames' in meta:
                clip_info['key_frames'] = meta['key_frames']
            try:
                with open(os.path.join(submit_data_path, "clip_info.json"), "w") as fp:
                    ss = json.dumps(clip_info, ensure_ascii=False, default=dump_numpy)
                    fp.write(ss)
                prepare_copy("gnss.json")
                prepare_copy("vehicle.json")

                sin_radar_path = os.path.join(seg_frame_path, "sin_radar")
                sin_radar_tar_path = os.path.join(submit_data_path, "sin_radar.tar.gz")
                logger.info(f">>>> Prepare tar {sin_radar_path}")
                if os.path.exists(sin_radar_tar_path):
                    delete_non_empty_directory(sin_radar_tar_path)        
                if os.path.exists(sin_radar_path):
                    compress_sinradar_with_gzip(sin_radar_path, sin_radar_tar_path, meta)

                vehicle_config_path = os.path.join(seg_frame_path, "vehicle_config")
                vehicle_config_tar_path = os.path.join(submit_data_path, "vehicle_config.tar.gz")        
                logger.info(f">>>> Prepare tar {vehicle_config_path}")
                if os.path.exists(vehicle_config_tar_path):
                    delete_non_empty_directory(vehicle_config_tar_path)
                if os.path.exists(vehicle_config_path):
                    compress_directory_with_gzip(vehicle_config_path, vehicle_config_tar_path)
            except Exception as e:
                logger.error(f"Caught an exception of type {type(e).__name__}: {e}")
                raise e
            
            pool.apply_async(
                gen_hpp_seg_lmdb,
                args=(
                    meta,
                    submit_data_path,
                    enable_cams,
                    enable_bpearls,
                    2000
                ),
                error_callback=lambda e: error_callback(e, error_event)
            )     
        pool.close()
        pool.join()

        while True:  
            if error_event.is_set():  # 如果检测到异常  
                logger.error("Terminating all processes due to an error.")  
                pool.terminate()  # 终止所有子进程  
                sys.exit(1)  # 退出主程序  
            break  # 如果没有异常，跳出循环 

        logger.info(f">>>> {car_name}.{subfix} Prepare LMDB Done.")
    except Exception as e:  
        # 捕获主进程中的异常  
        logger.error(f"Error occurred in main process: {e}")  
        pool.terminate()  # 终止所有子进程  
        sys.exit(1)  # 退出主程序


if  __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        logger.error(f"{config_file} Not Exists.")
        sys.exit(1)

    work_tmp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_tmp_dir, "node_lmdb_pack.log"), rotation="50 MB")

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    seg_config = run_config["preprocess"]
    seg_mode =  seg_config["seg_mode"]
    if seg_mode == 'hpp' or seg_mode == 'hpp_luce':
        node_main_hpp(run_config)
    else:
        node_main(run_config)
    
    with open("SUCCESS.done", "w") as fp:
        fp.write("success")

    sys.exit(0)    
