import os, sys
import numpy as np
from copy import deepcopy
import traceback as tb
import pandas as pd
import cv2
import json
import tqdm

def get_cam_pos(cam):
    if "front_left" in cam:
        return "FL"
    if "front_right" in cam:
        return "FR"
    if "rear_left" in cam:
        return "RL"
    if "rear_right" in cam:
        return "RR"
    if "front" in cam:
        return "F"
    if  "rear" in cam:
        return "R"
    return "F"

def draw_rect_img(image, points, color, linewidth):
    prev = points[-1]
    for corner in points:
        cv2.line(image, (int(prev[0]), int(prev[1])), (int(corner[0]), int(corner[1])), color, linewidth, cv2.LINE_AA)
        prev = corner 
    return image

def load_raw_calibs(calibration, cam_names):
    exts = calibration['extrinsics']
    ints = calibration['intrinsics']
    calibs = {}
    for cam_name in cam_names:
        calibs[cam_name] = {}

    for ext_ in exts:
        cam_name = ext_['target']
        if cam_name in cam_names:
            r, t = np.array(ext_["rvec"]), np.array(ext_["tvec"])
            r = cv2.Rodrigues(r)[0]
            r = np.reshape(r, [3, 3])
            t = np.reshape(t, [3, 1])
            extrinsic = np.concatenate([r, t], -1)
            calibs[cam_name]['extrinsics'] = extrinsic

    for int_ in ints:
        cam_name = int_['sensor_position']
        if cam_name in cam_names:
            intrinsic = np.array(int_["mtx"], dtype=np.float32).reshape([3, 3])
            distortion = np.array(int_["dist"], dtype=np.float32).reshape([-1])
            calibs[cam_name]['intrinsics'] = intrinsic
            calibs[cam_name]['distortion'] = distortion
            calibs[cam_name]['image_size'] = int_['image_size']
            calibs[cam_name]['cam_model'] = int_["cam_model"]

    return calibs

def load_calibration(calibs, cam_name, new_image_shape=None):    
    calib_raw_dict = calibs[cam_name]
    extrinsic = calib_raw_dict["extrinsics"]

    intrinsic = calib_raw_dict['intrinsics']
    distortion = calib_raw_dict['distortion']
    image_shape = calib_raw_dict["image_size"]
    undistort_mode = calib_raw_dict["cam_model"]
    
    calib_camera = {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "distortion": distortion,
        "image_shape": image_shape,
        "undistort_mode": undistort_mode,
    }

    if undistort_mode == "fisheye":
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            intrinsic, distortion, np.eye(3), intrinsic, image_shape, cv2.CV_16SC2
        )
        calib_camera['map1'] = map1
        calib_camera['map2'] = map2
        calib_camera['undist_intrinsic'] = intrinsic
    elif undistort_mode == "pinhole":
        undist_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
            intrinsic,
            distortion,
            image_shape,
            alpha=0.0,
            newImgSize=image_shape,
        )
        calib_camera['undist_intrinsic'] = undist_intrinsic
    else:
        raise TypeError("wrong mode: %s" % undistort_mode)

    if new_image_shape is not None:
        calib_camera['new_image_shape'] = new_image_shape
        calib_camera['new_undist_intrinsic'] = deepcopy(calib_camera['undist_intrinsic'])
        calib_camera['new_undist_intrinsic'][0] = new_image_shape[0]/image_shape[0]*calib_camera['new_undist_intrinsic'][0]
        calib_camera['new_undist_intrinsic'][1] = new_image_shape[1]/image_shape[1]*calib_camera['new_undist_intrinsic'][1]
        calib_camera['new_lidar_to_image'] = np.matmul(np.array(calib_camera['new_undist_intrinsic']), 
                                np.array(calib_camera["extrinsic"]))
    return calib_camera

def func_gen_seg_crop_imgs(segid, clip_anno, clip_obstacle, spec_obj_ids, spec_class):
    # clip_obstacle = os.path.join(obs_root, subfix, segid)
    clip_obs_info_json = os.path.join(clip_obstacle, f"{segid}_infos.json")
    anno_json = os.path.join(clip_anno, "annotation.json")
    # anno_json = os.path.join(anno_path, subfix, segid, "annotation.json")

    if not os.path.exists(clip_obs_info_json):
        print(f"{clip_obs_info_json} not exists")
        return None

    if not os.path.exists(anno_json):
        print(f"{anno_json} not exists")
        return None

    anno = json.load(open(anno_json))
    obs_anno = anno['obstacle']['annotations']
    # obs_anno = anno['obstacle_static']['annotations']
    if len(obs_anno) < 1:
        return None
    
    calib = anno['calib']
    sensors = calib['sensors']
    cameras = [item for item in sensors if item not in ["gnss", "vehicle", "lidar"] and "lidar" not in item and "around" not in item]
    calibs = load_raw_calibs(calib, cameras)
    cam_calibs = {}  
    for cam in cameras:
        cam_calib = load_calibration(calibs, cam, (1920,1080))        
        cam_calibs[cam] = cam_calib
    
    clip_obs_info = json.load(open(clip_obs_info_json))
    frames = clip_obs_info['frames']
    # print(len(frames))
    obj_id_crop_imgs = {}
    for i, f in enumerate(frames):
        ts = f['pc_frame']['timestamp']
        if ts not in obs_anno:
            print(f"{ts} not in obs_anno")
            continue

        # frame = f
        frame_annos = obs_anno[ts]
        for obj_ann in frame_annos:
            # if obj_ann['isolation'] == 'False':
            #     continue
            static = (obj_ann['static'] == "True")
            ann_class = obj_ann['class_name']
            if ann_class not in spec_class:
                continue
            
            points_lidar = np.array(obj_ann['points_3d']) 
            distance = np.min(points_lidar[:1])
            obj_height = abs(points_lidar[0,2] - points_lidar[2,2])
            if abs(distance) > 70:
                continue
            obj_id = obj_ann['track_id']
            if obj_id not in spec_obj_ids:
                continue
            if obj_id not in obj_id_crop_imgs:
                obj_id_crop_imgs[obj_id] = []

            for cam in cameras:
                if cam not in f['image_frames']:
                    continue
                cam_frame_jpg = os.path.join(clip_obstacle, "image_frames", cam, f"{f['image_frames'][cam]['frame_id']}.jpg")
                cam_frame_raw = cv2.imread(cam_frame_jpg)
                cam_frame = cv2.resize(cam_frame_raw, (1920,1080))
                cam_calib = cam_calibs[cam]

                pt_camera = np.matmul(cam_calib['extrinsic'][:3, :3], np.array(points_lidar).T).T + cam_calib['extrinsic'][:3,3]
                pt_image = np.matmul(cam_calib['new_undist_intrinsic'], pt_camera.T).T
                pt_image[:,:2] = pt_image[:,:2]/pt_image[:,[2]]
                if pt_image[:,2].mean()<0.5:
                    continue
                # After the marking result is projected into the image, it is cut off the image part. 
                # 200 here is an experience value.
                pt_image[:, :2] = np.where((pt_image[:, :2] > -200) & (pt_image[:, :2] < 0), 1, pt_image[:, :2])
                if (pt_image[:, :2]<0).any():
                    continue
                pt_image[:, 0] = np.where((pt_image[:, 0] > 1920) & (pt_image[:, 0] < 2120), 1918, pt_image[:, 0])
                pt_image[:, 1] = np.where((pt_image[:, 1] > 1080) & (pt_image[:, 1] < 1280), 1078, pt_image[:, 1])
                if (pt_image[:, 0] > 1920).any() or (pt_image[:, 1] > 1080).any():
                    continue

                image = cam_frame[np.min(pt_image[:,1]).astype(np.int32): np.max(pt_image[:,1]).astype(np.int32), 
                                np.min(pt_image[:,0]).astype(np.int32): np.max(pt_image[:,0]).astype(np.int32)]
                
                if image.shape[0] < 128 and image.shape[1] < 128:                    
                    center = [
                        np.min(pt_image[:, 0]).astype(np.int32) + image.shape[1] // 2,
                        np.min(pt_image[:, 1]).astype(np.int32) + image.shape[0] // 2,
                    ]
                    rect = np.array([[np.min(pt_image[:,0]), np.min(pt_image[:,1])],
                                    [np.max(pt_image[:,0]), np.min(pt_image[:,1])],
                                    [np.max(pt_image[:,0]), np.max(pt_image[:,1])],
                                    [np.min(pt_image[:,0]), np.max(pt_image[:,1])]]).astype(np.int32)
                    image =  draw_rect_img(cam_frame, rect.tolist(), (0,255,0), 2)
                    min_h = center[0] - 64 if (center[0] - 64) > 0 else 0
                    max_h = center[0] + 64 if (center[0] + 64) < 1920 else 1920
                    min_w = center[1] - 64 if (center[1] - 64) > 0 else 0
                    max_w = center[1] + 64 if (center[1] + 64) < 1080 else 1080
                    tmp_img = image[min_w:max_w, min_h:max_h]
                    if tmp_img.shape[0] != 128 or tmp_img.shape[1] != 128:
                        tmp_img = cv2.resize(tmp_img, (128,128))
                else:           
                    tmp_img = cv2.resize(image, (128,128))
                cv2.putText(tmp_img, get_cam_pos(cam), (20,20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                obj_id_crop_imgs[obj_id].append((cam, i, ann_class, tmp_img, float(obj_height), static))
                
    return obj_id_crop_imgs

def parse_annotation(segid, clip_anno, clip_obstacle, spec_class):
    clip_obs_info_json = os.path.join(clip_obstacle, f"{segid}_infos.json")
    anno_json = os.path.join(clip_anno, "annotation.json")

    if not os.path.exists(clip_obs_info_json):
        print(f"{clip_obs_info_json} not exists")
        return None

    if not os.path.exists(anno_json):
        print(f"{anno_json} not exists")
        return None
    
    anno = json.load(open(anno_json))
    if "obstacle" not in anno:
        return None
    if "annotations" not in anno['obstacle']:
        return None
    obs_anno = anno['obstacle']['annotations']
    # obs_anno = anno['obstacle_static']['annotations']
    if len(obs_anno) < 1:
        return None
    
    clip_obs_info = json.load(open(clip_obs_info_json))
    frames = clip_obs_info['frames']
    seg_count = {"obj_ids": [], "lost_frames": []}
    for f in frames:
        ts = f['pc_frame']['timestamp']
        if ts not in obs_anno:
            # print(f"{ts} not in obs_anno")
            seg_count['lost_frames'].append(ts)
            continue

        # frame = f
        frame_annos = obs_anno[ts]
        for obj_ann in frame_annos:
            static = (obj_ann['static'] == "True")
            ann_class = obj_ann['class_name']
            if ann_class not in spec_class:
                continue
            
            points_lidar = np.array(obj_ann['points_3d']) 
            distance = np.min(points_lidar[:1])
            obj_height = abs(points_lidar[0,2] - points_lidar[2,2])
            if distance > 70:
                continue
            obj_id = obj_ann['track_id']
            if obj_id not in seg_count:
                seg_count['obj_ids'].append(obj_id)
                seg_count[obj_id] = {
                    "class_name": ann_class,
                    'static': static,
                    'obj_height': obj_height
                }
            
            if 'static' in seg_count[obj_id]:
                if not seg_count[obj_id]['static']:
                    continue
                else:
                    # 如果某一帧的标注里标注为动态物体，那么一定有人，所以这里一定置为动态物体
                    seg_count[obj_id]['static'] = static
            
            #  这里的目标高度取最大高度，这里可能有人上车
            if 'obj_height' in  seg_count[obj_id]:
                if seg_count[obj_id]['obj_height'] < obj_height:
                    seg_count[obj_id]['obj_height'] = obj_height
            
    return seg_count

def parse_filter_xlss(filter_xls_files: list):
    segids = {}
    obj_cnt = 0
    for filter_xls_file in filter_xls_files:
        df = pd.read_excel(filter_xls_file, header=0)
        col_lists = [df[col].tolist() for col in df.columns]
        # print(df.columns[0]," : ",col_lists[0])
        result = dict(zip(df.columns, col_lists))

        for key in [
        "bicycle-withoutrider",
        "bicycle-withrider",
        "tricyclist-withoutrider",
        "tricyclist-withrider",
        ]:
            if key in result.keys():
                segs = [item for item in result[key] if isinstance(item, str)]
                # print(key, " : ", segs)
                for seg in segs:
                    seg, _ = os.path.splitext(seg)
                    segid, _, objid = seg.split("+")
                    obj_cnt += 1
                    if segid not in segids:
                        segids[segid] = {objid: key}
                    else:
                        segids[segid][objid] = key
    print(f"People filter {len(segids)} segs with {obj_cnt} objs")
    return segids

def main_xls():
    spec_class=['bicycle', 'tricyclist']
    xlsx_file = "/data_autodrive/users/brli/dev_raw_data/xingrenerlunche_segs.xlsx"
    # xlsx_file = "/data_autodrive/users/brli/dev_raw_data/test_segs.xlsx"
    # dst_path = "/data_autodrive/auto/subdivise/20240305"
    if len(sys.argv) > 1:
        xlsx_file = sys.argv[1]
    df = pd.read_excel(xlsx_file, header=0)
    # df = pd.read_excel(xlsx_file, header=0, sheet_name="segs_0")

    col_lists = [df[col].tolist() for col in df.columns]
    types = ['train', 'test']
    anno_type = [
        "annotation_train",
        "annotation_test"
    ]
    segids = {}
    seg_frame_cnt = {
        "seg_cnt":0,
        "find_seg":0,
        # "problem_segs":[],        
    }
    for idx, segid in enumerate(col_lists[2]):
        car = col_lists[0][idx]
        date = str( col_lists[1][idx])
        seg_path = col_lists[3][idx]    
        seg_frame_cnt["seg_cnt"] += 1
        if type(col_lists[4][idx]) is not str:
            # print(col_lists[4][idx])
            print(f"\t {segid} cannot find autopath in excel.")
            continue
        else:
            base_path = col_lists[4][idx]        
            pack = base_path.split("/")         
            if len(pack) == 6:
                car_name = pack[2]
                task = pack[3]
                base_auto_path= f"/data_autodrive/auto/{pack[1]}/{car_name}/{task}"
            elif len(pack) == 7:
                car_name = pack[2]
                task = pack[3]
                sub_task = pack[4]
                base_auto_path = f"/data_autodrive/auto/{pack[1]}/{car_name}/{task}/{sub_task}"
            elif len(pack) == 5:
                car_name = pack[2]
                base_auto_path = f"/data_autodrive/auto/{pack[1]}/{car_name}"
            else:
                # print(base_path)
                print(f"\t {segid}/{base_path} illegal.")
                continue

            obs_path = os.path.join(base_auto_path, "clip_obstacle", date, segid)
            anno_path = os.path.join(
                    base_auto_path, "clip_submit", "annotation_train", date, segid
                )                

            if not os.path.exists(obs_path):
                obs_path = os.path.join(base_auto_path, "clip_obstacle", date, segid)
                if not os.path.exists(obs_path):
                    obs_path = None
                    # seg_frame_cnt['problem_segs'].append(segid)
                    print(f"\t {segid} cannot find clip_obs or clip_anno")
                    continue
            if not os.path.exists(anno_path):                
                anno_path = os.path.join(
                        base_auto_path, "clip_submit", "annotation_test", date, segid
                    )                 
                if not os.path.exists(anno_path):
                    anno_path = None  
                    # seg_frame_cnt['problem_segs'].append(segid)
                    print(f"\t {segid} cannot find clip_obs or clip_anno")
                    continue

                # print(f"{idx}:{segid} <-> {anno_path}")
                # print(f"\t\t {obs_path}")
            segids[segid] = {
                "obs_path": obs_path, 
                "anno_path": anno_path,
                "subfix": date,
                "car": car_name
            }
            seg_frame_cnt['find_seg'] += 1

    print(">>>>>>>>>>")
    print(f"GET [{len(segids)}] seg to handle")
    print(f"{seg_frame_cnt}")
    print("<<<<<<<<<<")
    return segids

def filter_one_by_rules(segids, segid, spec_class, total_cnt):    
    clip_obstacle = segids[segid]["obs_path"]
    clip_anno = segids[segid]["anno_path"]
    subfix = segids[segid]["subfix"]
    car_name =  segids[segid]["car"]

    try:
        seg_count = parse_annotation(segid, clip_anno, clip_obstacle, spec_class)
    except Exception as e:
        print(f"Error occurred while processing segid: {segid}")
        tb.print_exc()
        return None
    if seg_count is None:
        return None

    if len(seg_count['obj_ids']) == 0:
        return None

    seg_info = deepcopy(seg_count)
    # dumpinfos[segid] = seg_info
    if total_cnt is not None:
        total_cnt['total'] += len(seg_count['obj_ids'])
    for ann_id in seg_count['obj_ids']:
        obj_info = seg_count[ann_id]
        class_name = obj_info['class_name']
        height = obj_info['obj_height']
        static = obj_info['static']
        if static:
            seg_info[ann_id]['static'] = 's'
            if class_name == 'bicycle':
                if total_cnt is not None:
                    total_cnt['bicycle']['total'] += 1
                    total_cnt['bicycle']['static']["total"] += 1
                if height > 1.55:
                    if total_cnt is not None:
                        total_cnt['bicycle']['static']['high'] += 1
                    seg_info[ann_id]['subclass'] = 'bicycle-withrider'
                elif height < 1.35:
                    seg_info[ann_id]['subclass'] = 'bicycle-withoutrider'
                    if total_cnt is not None:
                        total_cnt['bicycle']['static']['low'] += 1
                else:
                    if total_cnt is not None:
                        total_cnt['bicycle']['static']['middle'] += 1
            else:
                if total_cnt is not None:
                    total_cnt['tricyclist']['total'] += 1
                    total_cnt['tricyclist']['static']['total'] += 1
                if height > 1.55:
                    if total_cnt is not None:
                        total_cnt['tricyclist']['static']['high'] += 1
                elif height < 1.35:
                    seg_info[ann_id]['subclass'] =  'tricyclist-withoutrider'
                    if total_cnt is not None:
                        total_cnt['tricyclist']['static']['low'] += 1
                else:
                    if total_cnt is not None:
                        total_cnt['tricyclist']['static']['middle'] += 1

        else:
            seg_info[ann_id]['static'] = 'd'
            if class_name == 'bicycle':
                total_cnt['bicycle']['total'] += 1
                total_cnt['bicycle']['dynamic'] += 1
                seg_info[ann_id]['subclass'] = 'bicycle-withrider'
            else:
                total_cnt['tricyclist']['total'] += 1
                total_cnt['tricyclist']['dynamic']['total'] += 1
                seg_info[ann_id]['subclass'] = 'tricyclist-withrider'
    return seg_info

def filter_from_rules(segids, spec_class=['bicycle', 'tricyclist']):
    dumpinfos = {
        "total": len(segids),
        "use_seg": 0,
        "unuse_seg": 0
    }
    total_cnt = {
        "total": 0,
        "use_seg": 0,
        "unuse_seg": 0,
        "total_seg": len(segids),
        "bicycle": {
            "total": 0,
            "static": {"high": 0, "middle": 0, "low": 0, "total": 0},
            "dynamic": 0,
        },
        "tricyclist": {
            "total": 0,
            "static": {"high": 0, "middle": 0, "low": 0, "total": 0},
            "dynamic": {"high": 0, "middle": 0, "low": 0, "total": 0},
        },
    }
    for segid in tqdm.tqdm(list(segids.keys())):
        dumpinfos[segid] = {}
        seg_info = filter_one_by_rules(segids, segid, spec_class, total_cnt)           
        if seg_info is not None:         
            dumpinfos[segid] = seg_info

    total_cnt['use_seg'] = dumpinfos['use_seg']
    total_cnt['unuse_seg'] =  dumpinfos['unuse_seg']
    print(total_cnt)
    # with open("dump_infos.json", "w") as fp:
    #     ss = json.dumps(dumpinfos)
    #     fp.write(ss)
    
    obj_ids = {}
    obj_cnt = 0
    for seg_id, seg_info in dumpinfos.items():
        # seg_info = value
        if 'seg' not in seg_id:
            continue
        if seg_id == "use_seg" or seg_id == "unuse_seg":
            continue
        if len(seg_info) == 0:
            continue

        seg_obj_ids = seg_info['obj_ids']
        for obj_id in seg_obj_ids:
            obj_info = seg_info[obj_id]
            if 'subclass' in obj_info:
                obj_cnt += 1
                if seg_id not in obj_ids:
                        obj_ids[seg_id] = {obj_id: obj_info['subclass']}
                else:
                    obj_ids[seg_id][obj_id] = obj_info['subclass']
        
    print(f"Total filter {len(obj_ids)} segs with {obj_cnt} objs")
    return obj_ids

def func_combine_objids(filter_objids, xls_objids):
    ret_objids = deepcopy(filter_objids)
    for segid in xls_objids:
        seg_info = xls_objids[segid]
        if segid in ret_objids:
            b_seg_info = ret_objids[segid]
            for objid in seg_info:
                b_seg_info[objid] = seg_info[objid]
        else:
            ret_objids[segid] = seg_info
    return ret_objids

def check_anno(obs_anno: dict):
    if len(obs_anno) < 0:
        return []

    ret = []
    for ts, anns in obs_anno.items():
        for ann in anns:
            track_id = ann["track_id"]
            class_name = ann["class_name"]
            if "bicycle" in class_name or "tricyclist" in class_name:
                points_lidar = np.array(ann["points_3d"])
                distance = np.min(points_lidar[:1])
                if distance > 70:
                    continue
                if class_name not in [
                    "bicycle-withoutrider",
                    "bicycle-withrider",
                    "tricyclist-withoutrider",
                    "tricyclist-withrider",
                ] and 'crowd' not in class_name:
                    # print(f"{segid}/{ts} <-> {track_id} class error")
                    ret.append(track_id)
            else:
                continue

    return ret

def put_classname(img, text):
    height, width = img.shape[:2]
    font = cv2.FONT_HERSHEY_PLAIN
    color = (255, 0, 0)
    font_scale = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, 2)
    text_x = width - text_size[0] - 10 
    text_y = height - text_size[1] - 10  
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, 2, cv2.LINE_AA)

if __name__ ==  "__main__":
    segids = main_xls()
    spec_class=['bicycle', 'tricyclist']
    objids = filter_from_rules(segids)
    
    filters = [
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results/erlunche_result.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results/20230308-19cp2.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results/20230308-2940.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results/20230308-3xx23.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results/20230308-b8615.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results//20230308-0fx60.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results//20230308-1482.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results/20230308-2xx71.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results/20230308-32694.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results/20230308-53054.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results/20230308-8j998.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results/20230308-y7862.xlsx",
        "/data_autodrive/users/brli/dev_raw_data/erlunche_results/20240308-77052.xlsx",
    ]

    human_filter_objids = parse_filter_xlss(filters)
    total_objids = func_combine_objids(objids, human_filter_objids)

    output_root = "/data_autodrive/auto/subdivise/result_v5"
    os.makedirs(output_root, exist_ok=True)
    a_output_root = os.path.join(output_root, "subclass")
    b_output_root = os.path.join(output_root, "subclass_no")
    img_output_root = os.path.join(output_root, "subdivise")

    with open(os.path.join(output_root, "seg_ids.json"), "w") as wfp:
        ss = json.dumps(segids)
        wfp.write(ss)

    with open(os.path.join(output_root, "obj_ids.json"), "w") as wfp:
        ss = json.dumps(total_objids)
        wfp.write(ss)
    # with open(os.path.join(output_root, "obj_ids.json"), "r") as fp:
    #     total_objids = json.load(fp)

    seg_id_lst = list(segids.keys())
    _seg_id_cnt = len(seg_id_lst)
    for i,seg_id in enumerate(seg_id_lst):
        if i % 10 == 0:
            print(f"progress {i}/{_seg_id_cnt}")
        clip_obstacle = segids[seg_id]["obs_path"]
        clip_anno = segids[seg_id]["anno_path"]
        subfix = segids[seg_id]["subfix"]
        car_name =  segids[seg_id]["car"]

        try:
            seg_count = parse_annotation(seg_id, clip_anno, clip_obstacle, spec_class)
        except Exception as e:
            print(f"Error occurred while processing segid: {seg_id}")
            tb.print_exc()
            continue

        if seg_count is None:
            continue

        if len(seg_count['obj_ids']) == 0:
            continue

        spec_objids = []
        if seg_id not in total_objids:
            spec_objids.extend(seg_count['obj_ids'])            
        else:
            filter_objids = total_objids[seg_id]        
            for ann_id in seg_count['obj_ids']:
                if ann_id not in filter_objids:
                    if ann_id not in spec_objids:
                        spec_objids.append(ann_id)  

            if len(spec_objids) == 0:
                clip_obs_info_json = os.path.join(clip_obstacle, f"{seg_id}_infos.json")
                clip_obs_info = json.load(open(clip_obs_info_json))
                frames = clip_obs_info["frames"]

                anno_json = os.path.join(clip_anno, "annotation.json")
                prev_anno = json.load(open(anno_json))
                curr_anno = deepcopy(prev_anno)
                if "obstacle" not in prev_anno:
                    continue
                if "annotations" not in prev_anno['obstacle']:
                    continue
                curr_obs_anno = curr_anno["obstacle"]["annotations"]
                prev_obs_anno = prev_anno["obstacle"]["annotations"]
                if len(prev_obs_anno) < 1:
                    continue

                for i, f in enumerate(frames):
                    ts = f["pc_frame"]["timestamp"]
                    if ts not in prev_obs_anno:
                        continue

                    # frame = f
                    curr_frame_annos = curr_obs_anno[ts]
                    prev_frame_annos = prev_obs_anno[ts]
                    for i, obj_ann in enumerate(prev_frame_annos):
                        ann_class = obj_ann["class_name"]
                        if ann_class not in spec_class:
                            continue

                        points_lidar = np.array(obj_ann["points_3d"])
                        distance = np.min(points_lidar[:1])
                        if abs(distance) > 70:
                            continue
                        obj_id = obj_ann["track_id"]
                        if obj_id in filter_objids:
                            subclass = filter_objids[obj_id]
                            if ann_class == "tricyclist":
                                if subclass == "bicycle-withoutrider":
                                    subclass = "tricyclist-withoutrider"
                                if subclass == "bicycle-withrider":
                                    subclass = "tricylist-withrider"
                            curr_obj_ann = curr_frame_annos[i]
                            curr_obj_ann["class_name"] = subclass
                ret_ids = check_anno(curr_obs_anno)
                if len(ret_ids) == 0:
                    output = os.path.join(a_output_root, seg_id, "annotation.json")
                    os.makedirs(os.path.join(a_output_root, seg_id), exist_ok=True)
                    with open(output, "w") as wfp:
                        ss = json.dumps(curr_anno)
                        wfp.write(ss)
                else:
                    spec_objids.extend(ret_ids)
        
        unique_spec_objids = list(set(spec_objids))
        if len(unique_spec_objids) > 0:
            print(f"{seg_id} should gen crop images with objects[{unique_spec_objids}]")
            try:
                obj_id_crop_imgs = func_gen_seg_crop_imgs(seg_id, clip_anno, clip_obstacle, unique_spec_objids, spec_class)
            except Exception as e:
                print(f"Error occurred while processing segid: {seg_id}")
                tb.print_exc()
                continue 
            if obj_id_crop_imgs is None:
                continue 

            for k, v in obj_id_crop_imgs.items():
                if len(v) < 1:
                    continue
                obj_id = k
                obj_info = {
                    "obj_id": obj_id,
                    "frame_ids": []
                }

                imgs = v
                img_nums = len(imgs)
                col_num = 8
                img_list = []
                tmp_list = []
                for i, item in enumerate(imgs):        
                    _, frame_id, class_name, img, objh, static = item
                    objh_str = "{:.2f}".format(objh)
                    static_str = "s" if static else "d"
                    obj_info['class_name'] =  class_name
                    obj_info['static'] = static_str
                    obj_info['object_height'] = objh_str
                    if img is None:
                        continue
                    tmp_list.append(img)
                    if i != 0 and (i + 1) % col_num == 0:
                        _tmp = deepcopy(tmp_list)
                        img_list.append(_tmp)
                        del(tmp_list[:])     
                if len(tmp_list) > 0:     
                    _tmp = deepcopy(tmp_list)
                    img_list.append(_tmp)
                    del(tmp_list[:])   

                if len(img_list) == 0:
                    # this objid has no image to filter
                    continue
                res = []
                for lst in img_list:
                    while len(lst) < 8:
                        lst.append(np.zeros((128, 128, 3)))

                    lst_res = np.concatenate(lst, axis=1)                
                    res.append(lst_res)

                img_name = os.path.join(img_output_root, car_name, seg_id, f"{seg_id}+objid+{obj_id}.jpeg")     
                os.makedirs(os.path.join(img_output_root, car_name, seg_id), exist_ok=True)

                if len(res) == 1:
                    concat_img = res[0]
                    put_classname(concat_img, f"{obj_info['class_name']}:{objh_str}:{static_str}")
                    cv2.imwrite(img_name, concat_img)
                else:
                    concat_img = np.concatenate(res, axis=0)    
                    put_classname(concat_img,  f"{obj_info['class_name']}:{objh_str}:{static_str}")
                    cv2.imwrite(img_name, concat_img)

            
