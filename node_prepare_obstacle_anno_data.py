import os
import json
import shutil
import numpy as np
from utils import  load_calibration, load_bpearls, undistort, project_lidar2img, db_update_seg, gen_datasets
from multiprocessing.pool import Pool
from datetime import datetime
import time
import cv2
import math
from loguru import logger
import pandas as pd

curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_path)
import sys
sys.path.append(f"{curr_dir}/lib/python3.8/site_packages")
import pcd_iter as pcl

MAX_LOST_LIMIT = 2
INFO_FILE = "infos.json"
MAX_FRAMES = 100
PICK_INTERVAL = 5 # 10 * 0.5
DEFAULT_POSE_MATRIX = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
# 10034 is huawei camera test car with 10hz images
BYPASS_CARS = [
    "chery_10034",
    "chery_04228",
    "chery_18047",
    "chery_48160"
]

def parse_xlsx(excel_file):
    # sheets = ["21PT6", "72KX6", "04228"]
    ret = {}
    # df = pd.read_excel(excel_file, sheet_name=sheet, header=0)
    df = pd.read_excel(excel_file, sheet_name='72KX6', header=0)
    logger.info(f"Total {len(df)}")

    for idx, row in df.iterrows():
        src_path, _st, _end = row
        st_frame_idx = int(_st) * 10
        end_frame_idx = int(_end) * 10
        clip_id = os.path.basename(src_path)
        print(f"{idx}/{len(df)} {clip_id}")
        ret[clip_id] = [st_frame_idx, end_frame_idx, src_path]
    return ret

def prepare_obstacle(segment_path, dst_path, mode, seg_mode, boundary=None):
    def convert_frame(frame_idx, frame, meta, dst_path, cameras:list):
        frame_dir = meta['frames_path']
        bpearl_lidars = meta['bpearl_lidars']
        inno_lidars = meta['inno_lidars']
        bpearl_exts = load_bpearls(meta)
        
        frame_info = {}

        lidar = frame['lidar']
        pose = lidar['pose']

        images = frame['images']
        img_info = {}
        for cam in cameras:
            cam_calib = load_calibration(meta, cam)
            if cam in images:
                cam_img = images[cam]
                img_src = os.path.join(frame_dir, cam_img['path'])
                img_src_arr = cv2.imread(img_src) # type: ignore
                img_dst = os.path.join(dst_path, "image_frames", cam, "{}.jpg".format(frame_idx))
                img_dst_arr = undistort(cam_calib, img_src_arr)
                cv2.imwrite(img_dst, img_dst_arr) # type: ignore

                img_info[cam] = {}
                lidar_to_camera = np.concatenate((cam_calib["extrinsic"], np.array([[0,0,0,1]])))
                camera_to_world = np.matmul(np.array(pose), np.linalg.pinv(lidar_to_camera))
                img_info[cam]['pose'] = camera_to_world.tolist()
                img_info[cam]['timestamp'] = str(cam_img['timestamp'])
                img_info[cam]['frame_id'] = frame_idx
        frame_info['image_frames'] = img_info

        pc_info = {}
        pc_info['pose'] = pose
        pc_info['timestamp'] = str(lidar['timestamp'])
        pc_info['frame_id'] = "{}.pcd".format(frame_idx)

        pc_src = os.path.join(frame_dir, lidar['path'])
        pc_dst = os.path.join(dst_path, "pc_frames", "{}.pcd".format(frame_idx))
        pcd = pcl.LoadXYZI(pc_src)

        bpearls = frame.get('bpearls', None)
        innos = frame.get('innos', None)
        if bpearl_lidars is not None and bpearls is not None and len(bpearls) > 0 and len(bpearl_lidars) > 0:
            pcd_lst = [pcd]
            for bpearl in bpearl_lidars:
                bpearl_frame = bpearls.get(bpearl, None)
                if bpearl_frame is None:
                    logger.warning(f"\t{pc_info['timestamp']}loss {bpearl} pcd")
                    continue
                bpearl_pcd_src = os.path.join(frame_dir, bpearl_frame['path'])
                if not os.path.exists(bpearl_pcd_src):
                    logger.warning(f"\t{pc_info['timestamp']}loss {bpearl} pcd")
                    continue
                bpearl_pcd = pcl.LoadXYZI(bpearl_pcd_src)
                itensity_mat = bpearl_pcd[:, 3:]
                bpearl_ext = bpearl_exts[bpearl]
                r = bpearl_ext[:3, :3]
                t = bpearl_ext[:3, 3:].reshape((1,3))
                bpearl_trans_pcd = np.matmul(bpearl_pcd[:, :3], r.T) + t
                bpearl_trans_pcd = np.concatenate([bpearl_trans_pcd, itensity_mat], axis=-1)
                pcd_lst.append(bpearl_trans_pcd)
            if len(pcd_lst) > 1:
                merge_pcd = np.concatenate(pcd_lst, axis=0)
                pcl.SavePcdZ(merge_pcd, pc_dst)
            else:
                pcl.SavePcdZ(pcd, pc_dst)
        elif inno_lidars is not None and innos is not None and len(innos) > 0 and len(inno_lidars) > 0:
            pcd_lst = [pcd]
            for inno in inno_lidars:
                inno_frame = innos.get(inno, None)
                if inno_frame is None:
                    logger.warning(f"\t{pc_info['timestamp']}loss {inno} pcd")
                    continue
                inno_pcd_src = os.path.join(frame_dir, inno_frame['path'])
                if not os.path.exists(inno_pcd_src):
                    logger.warning(f"\t{pc_info['timestamp']}loss {inno} pcd")
                    continue
                inno_pcd = pcl.LoadXYZI(inno_pcd_src)
                itensity_mat = inno_pcd[:, 3:]
                inno_ext = bpearl_exts[inno]
                r = inno_ext[:3, :3]
                t = inno_ext[:3, 3:].reshape((1,3))
                inno_trans_pcd = np.matmul(inno_pcd[:, :3], r.T) + t
                inno_trans_pcd = np.concatenate([inno_trans_pcd, itensity_mat], axis=-1)
                pcd_lst.append(inno_trans_pcd)
            if len(pcd_lst) > 1:
                merge_pcd = np.concatenate(pcd_lst, axis=0)
                pcl.SavePcdZ(merge_pcd, pc_dst)
            else:
                pcl.SavePcdZ(pcd, pc_dst)
        else:
            pcl.SavePcdZ(pcd, pc_dst)

        frame_info['pc_frame'] = pc_info
        return frame_info     
    
    meta_json = os.path.join(segment_path, "meta.json")
    meta_fp = open(meta_json, "r")
    meta = json.load(meta_fp)
    enable_cameras = meta['cameras']
    segid = meta['seg_uid']
    car_name = meta['car']
    calib = meta['calibration']
    calib_sensors = calib['sensors']
    cameras = [item for item in enable_cameras if item in calib_sensors ]
    max_lost_limit = len(cameras) * MAX_LOST_LIMIT

    first_lidar_pose = np.array(meta['frames'][0]['lidar']['pose']).astype(np.float32)
    dft_pose_matrix = np.array(DEFAULT_POSE_MATRIX).astype(np.float32)
    if (first_lidar_pose==dft_pose_matrix).all():
        logger.warning(f"{segid} not selected .")
        return

    gnss_json = os.path.join(segment_path, "gnss.json")
    gnss_fp = open(gnss_json, "r")
    gnss = json.load(gnss_fp)

    veh_json =  os.path.join(segment_path, "vehicle.json")
    veh_fp = open(veh_json, "r")
    vehicle = json.load(veh_fp)

    if meta['lost_image_num'] > (max_lost_limit) and car_name not in BYPASS_CARS:
        logger.warning(f"{segid} skip. Because lost too much frame. {meta['lost_image_num']} > {max_lost_limit}")
        return

    os.makedirs(os.path.join(dst_path, "image_frames"), mode=0o775, exist_ok=True)
    os.makedirs(os.path.join(dst_path, "pc_frames"), mode=0o775, exist_ok=True)
    for cam in cameras:
        os.makedirs(os.path.join(dst_path, "image_frames", cam), mode=0o775, exist_ok=True)

    info = {}
    info['calib_params'] = calib
    info['frames'] = []

    # HPP模式下，上平台的数据由预先计算的关键帧生成，尽可能提供到120帧
    if seg_mode == 'hpp' or seg_mode == 'hpp_luce':
        if 'key_frames' not in meta:
            logger.warning(f"{segid} skip. Because no [key_frame] field.")
            return
        sig_frames = meta['key_frames']
        if len(sig_frames) <= 10:
            logger.warning(f"{segid} skip. Because too few key frame.")
            return
        sig_frames_lost = meta.get('key_frames_lost', 0)
        if sig_frames_lost > MAX_LOST_LIMIT:
            logger.warning(f"{segid} skip. Because too many key frame lost. [{sig_frames_lost}]")
            return
        
        # 不超过120帧的段直接转为标注数据
        seg_interval = 1
        if len(sig_frames) > 180: # 超过120帧的段，抽帧后转为标注数据
            seg_interval = math.ceil(len(sig_frames) / 180)
        frames = meta['frames']
        for idx, f in enumerate(sig_frames):
            if idx % seg_interval == 0:
                frame_idx = f['frame_idx']
                frame = frames[frame_idx]
                frame_info = convert_frame(int(idx / seg_interval), frame, meta, dst_path, cameras)
                info['frames'].append(frame_info)
        with open(os.path.join(dst_path, "{}_infos.json".format(segid)), "w") as fp:
            ss = json.dumps(info)
            fp.write(ss)

        meta_fp.close()
        gnss_fp.close()
        return 

    # if meta['lost_image_num'] > (max_lost_limit) and car_name not in BYPASS_CARS:
    #     logger.warning(f"{segid} skip. Because lost too much frame. {meta['lost_image_num']} > {max_lost_limit}")
    #     return

    frames = meta['frames']
    frame_cnt = len(frames)
    # only use middle 400M for obstacle anno
    start_idx = int(frame_cnt * 0.25)
    end_idx = int(frame_cnt * 0.75)
    if seg_mode == 'luce' or seg_mode == 'test':
        start_idx = int(frame_cnt * 0.05)
        end_idx = int(frame_cnt * 0.95)
    
    if seg_mode == 'aeb':
        start_idx = 0
        end_idx = frame_cnt - 1

    if seg_mode == 'traffic_light':
        start_idx = 0
        end_idx = int(frame_cnt * 0.5)

    if boundary is not None and isinstance(boundary, list):
        start_idx = boundary[0] - 50 
        if start_idx < 0:
            start_idx = 0
        end_idx = boundary[1] - 50
        if end_idx > frame_cnt:
            end_idx = int(frame_cnt)

    cnt = 0
    pre_speed = []
    for f_idx in range(start_idx, end_idx):
        # use frame every 0.5s
        sample_interval = PICK_INTERVAL
        if seg_mode == "hpp":
            sample_interval *= 2
        if f_idx % PICK_INTERVAL != 0:
            continue
        # total frame not bigger than 80
        if cnt > MAX_FRAMES and seg_mode != "aeb":
            break
        frame = frames[f_idx]
        frame_gnss = gnss[str(frame['gnss'])]
        speed = float(frame_gnss['speed'])
        # skip static frame
        if speed == 0:
            frame_veh = vehicle[str(frame['vehicle'])]
            speed = float(frame_veh['vehicle_spd'])
        if speed < 0.01 and sum(pre_speed) < 0.1:
            continue

        if len(pre_speed) > 10:
            pre_speed.pop(0)
        pre_speed.append(speed)
        
        frame_info = convert_frame(cnt, frame, meta, dst_path, cameras)
        info['frames'].append(frame_info)
        cnt += 1
    
    with open(os.path.join(dst_path, "{}_infos.json".format(segid)), "w") as fp:
        ss = json.dumps(info)
        fp.write(ss)

    meta_fp.close()
    gnss_fp.close()
    return 

def prepare_check(dst_check, obs_dst_path, meta):
    camera_names = meta['cameras']
    segid = meta["seg_uid"]
    frame_path = meta["frames_path"]
    clip_time = os.path.basename(frame_path)

    cam_calibs = {}
    cameras = [cam for cam in camera_names if 'around' not in cam and "30_8M" not in cam]  
    for cam in cameras:
        cam_calib = load_calibration(meta, cam, (1920,1080))        
        cam_calibs[cam] = cam_calib

    dst_proj = dst_check
    os.makedirs(dst_proj, mode=0o775, exist_ok=True)
    obs_info_json = os.path.join(obs_dst_path, "{}_infos.json".format(segid))
    if not os.path.exists(obs_info_json):
        logger.warning(f"skip clip[{segid}] check as {obs_info_json} not exist")
        return
    info_fp = open(obs_info_json, "r")
    info = json.load(info_fp)

    frames = info['frames']
    for frame in frames:
        frame_id_str = frame['pc_frame']['frame_id']
        _frame_id, _ = os.path.splitext(frame_id_str)
        frame_id = int(_frame_id)
        if frame_id % 5 != 0:
            continue
        top_res = []
        bottom_res = []

        pcd_file = os.path.join(obs_dst_path, "pc_frames", "{}.pcd".format(frame_id))
        check_img = os.path.join(dst_proj, "{}_{}.jpg".format(segid, frame_id))
        imgs = frame['image_frames']  
        curr_cam_names = list(imgs.keys())      

        half_cam_idx = len(cameras) // 2
        for cam_idx, cam in enumerate(cameras):
            if cam not in curr_cam_names:
                img = np.zeros((1080, 1920, 3))
                if cam_idx < (half_cam_idx):
                    top_res.append(img)
                else:
                    bottom_res.append(img)
            else:                
                points = pcl.LoadXYZI(pcd_file)[:, :3]

                img_file = os.path.join(
                    obs_dst_path, "image_frames", cam, "{}.jpg".format(frame_id))
                pt_size = 1
                if cam == "surround_front_60_8M":
                    pt_size = 2
                image = cv2.imread(img_file)
                if image is None:
                    img = np.zeros((1080, 1920, 3))
                    if cam_idx < (half_cam_idx):
                        top_res.append(img)
                    else:
                        bottom_res.append(img)
                    continue
                img = cv2.resize(image, (1920,1080))
                ret_img_array = project_lidar2img(points, img, cam_calibs[cam], pt_size)
                if cam_idx < (half_cam_idx):
                    top_res.append(ret_img_array)
                else:
                    bottom_res.append(ret_img_array)
        if len(bottom_res) - len(top_res) == 1:
            top_res.append(np.zeros((1080, 1920, 3)))
        if len(top_res) - len(bottom_res) == 1:
            bottom_res.append(np.zeros((1080, 1920, 3)))
        top = np.concatenate(top_res, axis=1)
        bottom = np.concatenate(bottom_res, axis=1)
        res = np.concatenate([top, bottom], axis=0)
        cv2.imwrite(check_img, res)

def node_main(run_config, spec_xlsx=None):
    seg_config = run_config["preprocess"]
    tgt_seg_path = seg_config["segment_path"]
    car_name = seg_config['car']
    seg_mode = seg_config['seg_mode']
    rec_cfg = run_config["reconstruction"]
    skip_reconstruct = False
    if rec_cfg['enable'] != "True":
        skip_reconstruct = True
    pre_anno_cfg = run_config['annotation']
    clip_lane = pre_anno_cfg['clip_lane']
    clip_lane_check = clip_lane.replace("clip_lane", "clip_lane_check")
    clip_obstacle =  pre_anno_cfg['clip_obstacle']
    clip_obstacle_test =  pre_anno_cfg['clip_obstacle_test']
    clip_check =  pre_anno_cfg['clip_check']
    test_road_gnss_file = pre_anno_cfg['test_gnss_info']
    deploy_cfg = run_config["deploy"]
    # src_deploy_root = deploy_cfg["clip_submit_data"]
    tgt_deploy_root = deploy_cfg["tgt_rdg_path"]
    subfix = deploy_cfg['data_subfix']
    preprocess_spec_clips = seg_config.get("spec_clips", None)
    spec_clips = None
    if spec_xlsx is not None:
        spec_clips = parse_xlsx(spec_xlsx)

    specs = list()
    if seg_mode == "hpp" and os.path.exists(clip_lane_check):
        for rgbs in os.listdir(clip_lane_check):
            seg_id, _ = os.path.splitext(rgbs)
            specs.append(seg_id)
    
    odometry_mode = run_config["odometry"]["pattern"]
    pool = Pool(processes=8)
    seg_root_path = tgt_seg_path
    if not os.path.exists(seg_root_path):
        logger.warning(f"{seg_root_path} NOT Exist...")
        sys.exit(0)
    seg_names = os.listdir(seg_root_path)
    seg_names.sort()
    seg_anno_dst = {}
    logger.info(f"......\t{tgt_seg_path} prepare_obstatcle... {str(datetime.now())}")
    for i, _seg in enumerate(seg_names):
        if preprocess_spec_clips is not None:
            go_on = False
            for clip in preprocess_spec_clips:
                if clip in _seg:
                    go_on = True
                    break
            if not go_on:
                continue
        seg_dir = os.path.join(seg_root_path, _seg)
        if not os.path.exists(os.path.join(seg_dir, "meta.json")):
            logger.warning("\tskip seg {} for not seleted".format(_seg))
            #db_update_seg(seg_dir, "", "")
            continue
        seg_meta_json = open(os.path.join(seg_dir, "meta.json"))
        try:
            meta = json.load(seg_meta_json)
        except Exception as e:
            logger.error(f"{_seg}_meta.json load error.")
            # os.remove(os.path.join(seg_dir, "meta.json"))
            continue

        gnss_json = os.path.join(seg_dir, "gnss.json")
        # tt_mode, _, _ = get_road_name(meta, gnss_json, test_road_gnss_file, odometry_mode)
        tt_mode, _, _ = gen_datasets(meta, gnss_json, odometry_mode=odometry_mode)
        obs_dst_path = os.path.join(clip_obstacle, _seg)
        if seg_mode == 'luce' or seg_mode == "hpp_luce" or seg_mode == 'test' or tt_mode == 'test' or seg_mode == 'aeb':
            obs_dst_path = os.path.join(clip_obstacle_test, _seg)
            tt_mode = 'test'
        seg_anno_dst[_seg] = {
            "obs": obs_dst_path,
            "mode": tt_mode
        }

        seg_clip = _seg[12:29]
        if spec_clips is not None:
            if seg_clip in spec_clips:
                if not os.path.exists(os.path.join(obs_dst_path, "{}_info.json".format(_seg))):
                    logger.info("Prepare Obstacle Anno Data {} in {}".format(_seg, obs_dst_path))
                    spec_boundary = [spec_clips[seg_clip][0], spec_clips[seg_clip][1]]
                    prepare_obstacle(seg_dir, obs_dst_path, tt_mode, seg_mode, spec_boundary)
                    continue
        else:
            if len(specs) > 0 and _seg not in specs:
                continue
            if not os.path.exists(os.path.join(obs_dst_path, "{}_info.json".format(_seg))):            
                logger.info("Prepare Obstacle Anno Data {} in {}".format(_seg, obs_dst_path))
                # prepare_obstacle(seg_dir, obs_dst_path, tt_mode, seg_mode)
                pool.apply_async(prepare_obstacle, args=(seg_dir, obs_dst_path, tt_mode, seg_mode))
            else:
                logger.info("Prepare Obstacle {} Done".format(_seg))

    pool.close()
    pool.join()  
    logger.info(f"......\t{car_name}.{tgt_seg_path} prepare_obstatcle end... {str(datetime.now())}")  
    if seg_mode == 'luce' or seg_mode == 'hpp_luce' or seg_mode == 'aeb':
        logger.info(f"......\t{car_name}.{tgt_seg_path} skip prepare_check data... ")
        return
        
    for i, _seg in enumerate(seg_names):
        seg_clip = _seg[12:29]
        if spec_clips is not None:
            if seg_clip not in spec_clips:
                continue
        if len(specs) > 0 and _seg not in specs:
                continue
        if preprocess_spec_clips is not None:
            go_on = False
            for clip in preprocess_spec_clips:
                if clip in _seg:
                    go_on = True
                    break
            if not go_on:
                continue
        seg_dir = os.path.join(seg_root_path, _seg)
        if not os.path.exists(os.path.join(seg_dir, "meta.json")):
            continue

        seg_meta_json = open(os.path.join(seg_dir, "meta.json"))
        meta = json.load(seg_meta_json)
        # lane_dst_path = seg_anno_dst[_seg]['lane']
        obs_dst_path = seg_anno_dst[_seg]['obs']
        obs_info_json = os.path.join(obs_dst_path, "{}_infos.json".format(_seg))
        if not os.path.exists(os.path.join(obs_dst_path, "{}_infos.json".format(_seg))):
            logger.warning(f" {obs_dst_path}/{_seg}_infos.json not Exists.")
            # os.system("rm -rf {}".format(obs_dst_path))
            db_update_seg(seg_dir, "", "")
        else:
            with open(obs_info_json, "r") as info_fp:
                info_json_dict = json.load(info_fp)
                if len(info_json_dict['frames']) < 5:
                    logger.warning(f"REMOVE {_seg} for too little frames.")
                    # os.system("rm -rf {}".format(obs_dst_path))
                    continue
            db_update_seg(seg_dir, "", obs_dst_path)                    
            logger.info("Produce check images {}".format(_seg))
            prepare_check(clip_check, obs_dst_path, meta)  

    logger.info(f"......\t{tgt_seg_path} prepare_check end... {str(datetime.now())}")

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        logger.error(f"{config_file} Not Exists.")
        sys.exit(1)

    spec_xlsx = None
    if len(sys.argv) > 2:
        spec_xlsx = sys.argv[2]

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "prepare_obstacle.log"))

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    node_main(run_config, spec_xlsx)
