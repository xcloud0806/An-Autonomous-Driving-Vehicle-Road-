import os
import numpy as np
import json
from .class_names import obstacle_classname_list, static_obstacle_classname_list, hpp_obstacle_classname_list

import PIL.Image as Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_velocity(ann_prev, ann_cur, ann_next):
    time_prev = (int(ann_cur["timestamp"]) - int(ann_prev["timestamp"])) / 1000
    time_next = (int(ann_next["timestamp"]) - int(ann_cur["timestamp"])) / 1000
    if time_prev > 0.8 or time_next > 0.8:
        return [None, None, None]
    vel_prev = (ann_cur["center"] - ann_prev["center"]) / time_prev
    vel_next = (ann_next["center"] - ann_cur["center"]) / time_next
    vel_cur = (vel_prev * time_next + vel_next * time_prev) / (time_next + time_prev)
    return vel_cur.tolist()

def gen_label_obstacle_scene(
        segid, obstacle_ann_path, obstacle_data_path, meta
):
    obstacle_ann_clip_path = os.path.join(obstacle_ann_path, "annotations/result.json")
    if not os.path.exists(obstacle_ann_clip_path):
        return dict()
    obstacle_clip_ann = json.load(open(obstacle_ann_clip_path))["researcherData"]
    obstacle_clip_path = obstacle_data_path
    # 兼容平台之前info格式
    if os.path.exists(os.path.join(obstacle_clip_path, "infos.json")):
        obstacle_infos = json.load(open(os.path.join(obstacle_clip_path, "infos.json")))
    else:
        obstacle_infos = json.load(
            open(os.path.join(obstacle_clip_path, "{}_infos.json".format(segid)))
        )

    ret = []
    for idx, obstacle_frame_ann in enumerate(obstacle_clip_ann):
        annos =  obstacle_frame_ann["clip_annotations"]
        for k, v in annos.items():
            v.replace(' ', ":")
            if v not in ret:
                ret.append(v)
    return ret            

def gen_label_obstacle(
    segid, obstacle_ann_path, obstacle_data_path, meta, classname_list=None
):
    obstacle_ann_clip_path = os.path.join(obstacle_ann_path, "annotations/result.json")
    if not os.path.exists(obstacle_ann_clip_path):
        return dict()
    obstacle_clip_ann = json.load(open(obstacle_ann_clip_path))["researcherData"]

    if classname_list is None:
        classname_list = obstacle_classname_list

    obstacle_clip_path = obstacle_data_path
    # 兼容平台之前info格式
    if os.path.exists(os.path.join(obstacle_clip_path, "infos.json")):
        obstacle_infos = json.load(open(os.path.join(obstacle_clip_path, "infos.json")))
    else:
        obstacle_infos = json.load(
            open(os.path.join(obstacle_clip_path, "{}_infos.json".format(segid)))
        )
    obstacle_clip_info = obstacle_infos["frames"]

    # 从clip info 中calib_dir中生成标定信息
    calib_dict = obstacle_infos["calib_params"]

    for idx, obstacle_frame_ann in enumerate(obstacle_clip_ann):
        obstacle_frame_ann_new = list()
        for object_ann in obstacle_frame_ann["frame_annotations"]:
            try:
                if object_ann["class-name"] not in classname_list:
                    continue
            except Exception as e:
                print("***************************{}*****************".format(segid))
                print(e)
                continue
            try:
                points_3d = [
                    [point["x"], point["y"], point["z"]]
                    for point in object_ann["points_3d"]
                ]
            except Exception as e:
                print("***************************{}*****************".format(segid))
                print(e)
                continue
            object_ann["points_3d"] = points_3d
            obstacle_frame_ann_new.append(object_ann)
        obstacle_clip_ann[idx] = obstacle_frame_ann_new

    # 按track_id索引整个序列, 生成id_ann_dict
    id_ann_dict = dict()
    for idx, (obstacle_frame_ann, obstacle_frame_info) in enumerate(
        zip(obstacle_clip_ann, obstacle_clip_info)
    ):
        for object_ann in obstacle_frame_ann:
            # object_ann.update(obstacle_frame_info)
            object_ann["timestamp"] = obstacle_frame_info["pc_frame"]["timestamp"]
            # object_ann['frame_idx'] = obstacle_frame_info['pc_frame']['clip_idx']
            object_ann["pose"] = obstacle_frame_info["pc_frame"]["pose"]
            object_ann["center"] = np.array(object_ann["points_3d"]).mean(0)
            track_id = object_ann["track_id"]
            if track_id in id_ann_dict:
                id_ann_dict[track_id].append(object_ann)
            else:
                id_ann_dict[track_id] = [object_ann]

    # 生成 velocity, 生成障碍物annotation
    obstacle_clip_label = dict()
    for track_id, ann_list in id_ann_dict.items():
        for idx, ann in enumerate(ann_list):
            if idx == 0 or idx == len(ann_list) - 1:
                velocity = [None, None, None]
            else:
                velocity = get_velocity(
                    ann_list[idx - 1], ann_list[idx], ann_list[idx + 1]
                )
            timestamp = ann["timestamp"]
            obstacle_object_label = {
                "points_3d": ann["points_3d"],
                "class_name": ann["class-name"],
                "track_id": ann["track_id"],
                "static": ann["static"],
                "isolation": ann["isolation"],
                "velocity": velocity,
                #'frame_idx': ann['frame_idx'],
            }
            ann_necessary_keys = [
                "timestamp",
                "class_name",
                "track_id",
                "static",
                "isolation",
                "velocity",
            ]
            for key in ann.keys():
                if key in ann_necessary_keys:
                    continue
                obstacle_object_label[key] = ann[key]
            # if "cipv" in ann:
            #     obstacle_object_label["cipv"] = ann["cipv"]

            # if "cipvfront" in ann:
            #     obstacle_object_label["cipvfront"] = ann["cipvfront"]

            # if 'target-obj' in ann:
            #     obstacle_object_label['target-obj'] = ann['target-obj']

            if timestamp in obstacle_clip_label:
                obstacle_clip_label[timestamp].append(obstacle_object_label)
            else:
                obstacle_clip_label[timestamp] = [obstacle_object_label]

    obstacle_clip_label = {
        "annotations": obstacle_clip_label,
        "classes": classname_list,
    }
    return obstacle_clip_label


def gen_label_obstacle_static(
    segid, obstacle_ann_path, obstacle_data_path, meta, no_static=False
):
    classname_list = static_obstacle_classname_list
    obstacle_static_clip_label = dict()
    if not no_static:
        obstacle_static_clip_label = gen_label_obstacle(
            segid, obstacle_ann_path, obstacle_data_path, meta, classname_list
        )
    return obstacle_static_clip_label


def gen_label_obstacle_hpp(
    segid, obstacle_ann_path, obstacle_data_path, meta, no_static=False
):
    classname_list = hpp_obstacle_classname_list
    obstacle_hpp_clip_label = dict()
    if not no_static:
        obstacle_hpp_clip_label = gen_label_obstacle(
            segid, obstacle_ann_path, obstacle_data_path, meta, classname_list
        )
    return obstacle_hpp_clip_label
