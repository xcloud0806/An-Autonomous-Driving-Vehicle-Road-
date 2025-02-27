import copy
import os
import shutil
import pandas as pd
import json
import numpy as np


def check_anno(obs_anno: dict, segid: str):
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


spec_class = ["bicycle", "tricyclist"]

def run_update(info_file, xlsx_file, filter_xls_files, output_root, imgs_root):
    # info_file = "/data_autodrive/users/brli/dev_raw_data/seg_objid_info.json"
    info_dict = json.load(open(info_file, "r"))
    a_output_root = os.path.join(output_root, "subclass")
    b_output_root = os.path.join(output_root, "subclass_no")
    image_output_root = os.path.join(output_root, "subdivise")

    # filter_xls_file = "/data_autodrive/users/brli/dev_raw_data/erlunche_result.xlsx"

    data_list = pd.read_excel(xlsx_file, header=0)
    data_col_lists = [data_list[col].tolist() for col in data_list.columns]

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

    for seg_id, seg_info in info_dict.items():
        # seg_info = value
        if 'seg' not in seg_id:
            continue
        if seg_id == "use_seg" or seg_id == "unuse_seg":
            continue
        if len(seg_info) == 0:
            continue

        # seg_obj_ids = seg_info['obj_ids']
        seg_obj_ids = seg_info
        for obj_id in seg_obj_ids:
            obj_info = seg_info[obj_id]
            if 'class_name' in obj_info and obj_info['class_name'] == 'bicycle':
                if obj_info['static'] == 'd':
                    obj_info['subclass'] = 'bicycle-withrider'
                    obj_cnt += 1
                    if seg_id not in segids:
                        segids[seg_id] = {obj_id: obj_info['subclass']}
                    else:
                        if obj_id in segids[seg_id]:
                            continue
                        segids[seg_id][obj_id] = obj_info['subclass']
                else:
                    if obj_info["obj_height"] >= 1.55:
                        obj_info['subclass'] =  'bicycle-withrider'
                        obj_cnt += 1
                        if seg_id not in segids:
                            segids[seg_id] = {obj_id: obj_info['subclass']}
                        else:
                            if obj_id in segids[seg_id]:
                                continue
                            segids[seg_id][obj_id] = obj_info['subclass']
                    elif  obj_info["obj_height"] < 1.35:
                        obj_info['subclass'] = 'bicycle-withoutrider'
                        obj_cnt += 1
                        if seg_id not in segids:
                            segids[seg_id] = {obj_id: obj_info['subclass']}
                        else:
                            if obj_id in segids[seg_id]:
                                continue
                            segids[seg_id][obj_id] = obj_info['subclass']
            elif 'class_name' in obj_info and obj_info['class_name'] == 'tricyclist':
                if obj_info['static'] == 'd':
                    obj_info['subclass'] = 'tricyclist-withrider'
                    obj_cnt += 1
                    if seg_id not in segids:
                        segids[seg_id] = {obj_id: obj_info['subclass']}
                    else:
                        segids[seg_id][obj_id] = obj_info['subclass']
                else:
                    if obj_info["obj_height"] < 1.35:
                        obj_info['subclass'] =  'tricyclist-withoutrider'
                        obj_cnt += 1
                        if seg_id not in segids:
                            segids[seg_id] = {obj_id: obj_info['subclass']}
                        else:
                            segids[seg_id][obj_id] = obj_info['subclass']
            else:
                obj_class = seg_info[obj_id]
                if obj_class in  [
                    "bicycle-withoutrider",
                    "bicycle-withrider",
                    "tricyclist-withoutrider",
                    "tricyclist-withrider",
                    ]:
                    if seg_id not in segids:
                        segids[seg_id] = {obj_id: obj_class}
                    else:
                        segids[seg_id][obj_id] = obj_class
                    obj_cnt += 1
                    
        
    print(f"Total filter {len(segids)} segs with {obj_cnt} objs")
        # for k, v in seg_info.items():
        #     _, _, objid = k.split("+")
        #     if "subclass" in v:
        #         if seg_id not in segids:
        #             segids[seg_id] = {objid: v["subclass"]}
        #         else:
        #             segids[seg_id][objid] = v["subclass"]

    with open(os.path.join(output_root, "updated_segids.json"), "w") as wfp:
        ss = json.dumps(segids)
        wfp.write(ss)
    types = ["train", "test"]
    seg_frame_cnt = {
        "seg_cnt":0,
        "use_seg":0,
        "problem_segs":[],
        "use_frame":0,
        "lost_frames":{
            "count":0,
        }
        
    }
    for idx, segid in enumerate(data_col_lists[2]):
        seg_frame_cnt["seg_cnt"] += 1
        car = data_col_lists[0][idx]
        date = str(data_col_lists[1][idx])
        seg_path = data_col_lists[3][idx]

        if type(data_col_lists[4][idx]) is not str:
            print(f"\t {segid} cannot find autopath in excel.")
            continue
        else:
            base_path = data_col_lists[4][idx]
            
            pack = base_path.split("/")
            if len(pack) == 6:
                car_name = pack[2]
                task = pack[3]
                base_auto_path = f"/data_autodrive/auto/{pack[1]}/{car_name}/{task}"
                imgs_seg_path = os.path.join(imgs_root, car_name, segid)
            elif len(pack) == 5:
                car_name = pack[2]
                base_auto_path = f"/data_autodrive/auto/{pack[1]}/{car_name}"
                imgs_seg_path = os.path.join(imgs_root, car_name, segid)
            else:
                print(f"\t {base_path} illegal.")
                continue

            obs_path = os.path.join(base_auto_path, "clip_obstacle", date, segid)
            anno_path = os.path.join(
                    base_auto_path, "clip_submit", "annotation_train", date, segid
                )                

            if not os.path.exists(obs_path):
                obs_path = os.path.join(base_auto_path, "clip_obstacle", date, segid)
                if not os.path.exists(obs_path):
                    obs_path = None
                    seg_frame_cnt['problem_segs'].append(segid)
                    print(f"\t {segid} cannot find clip_obs or clip_anno")
                    continue
            if not os.path.exists(anno_path):                
                anno_path = os.path.join(
                        base_auto_path, "clip_submit", "annotation_test", date, segid
                    )                 
                if not os.path.exists(anno_path):
                    anno_path = None  
                    seg_frame_cnt['problem_segs'].append(segid)
                    print(f"\t {segid} cannot find clip_obs or clip_anno")
                    continue

            seg_frame_cnt['use_seg'] += 1
            clip_obs_info_json = os.path.join(obs_path, f"{segid}_infos.json")
            clip_obs_info = json.load(open(clip_obs_info_json))
            frames = clip_obs_info["frames"]
            anno_json = os.path.join(anno_path, "annotation.json")
            if segid in segids:
                seg_info = segids[segid]

                prev_anno = json.load(open(anno_json))
                curr_anno = copy.deepcopy(prev_anno)
                if "obstacle" not in prev_anno:
                    seg_frame_cnt['use_seg'] -= 1
                    seg_frame_cnt['problem_segs'].append(segid)
                    continue
                if "annotations" not in prev_anno['obstacle']:
                    seg_frame_cnt['use_seg'] -= 1
                    seg_frame_cnt['problem_segs'].append(segid)
                    continue
                curr_obs_anno = curr_anno["obstacle"]["annotations"]
                prev_obs_anno = prev_anno["obstacle"]["annotations"]
                if len(prev_obs_anno) < 1:
                    continue

                for i, f in enumerate(frames):
                    ts = f["pc_frame"]["timestamp"]
                    if ts not in prev_obs_anno:
                        # print(f"{segid}/{ts} not in obs_anno")
                        seg_frame_cnt["lost_frames"]['count'] += 1
                        if segid not in seg_frame_cnt['lost_frames']:
                            seg_frame_cnt["lost_frames"][segid] = []
                        seg_frame_cnt["lost_frames"][segid].append(ts)
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
                        if distance > 70:
                            continue
                        obj_id = obj_ann["track_id"]
                        if obj_id in seg_info:
                            subclass = seg_info[obj_id]
                            if ann_class == "tricyclist":
                                if subclass == "bicycle-withoutrider":
                                    subclass = "tricyclist-withoutrider"
                                if subclass == "bicycle-withrider":
                                    subclass = "tricylist-withrider"
                            curr_obj_ann = curr_frame_annos[i]
                            curr_obj_ann["class_name"] = subclass
                ret_ids = check_anno(curr_obs_anno, segid)
                if len(ret_ids) == 0:
                    output = os.path.join(a_output_root, segid, "annotation.json")
                    os.makedirs(os.path.join(a_output_root, segid), exist_ok=True)
                    with open(output, "w") as wfp:
                        ss = json.dumps(curr_anno)
                        wfp.write(ss)
                else:
                    output = os.path.join(b_output_root, segid, "annotation.json")
                    os.makedirs(os.path.join(b_output_root, segid), exist_ok=True)
                    with open(output, "w") as wfp:
                        ss = json.dumps(curr_anno)
                        wfp.write(ss)

                    # for track_id in ret_ids:
                    #     img_name = f"{segid}+objid+{track_id}.jpeg"
                    #     img_obj_path = os.path.join(imgs_seg_path, img_name)
                    #     if not os.path.exists(img_obj_path):
                    #         continue
                    #     os.makedirs(os.path.join(image_output_root, car_name, segid), exist_ok=True)
                    #     shutil.copy(img_obj_path, os.path.join(image_output_root, car_name, segid, img_name))

    with open(os.path.join(output_root, "frame_count.json"), "w") as wfp:
        ss = json.dumps(seg_frame_cnt)
        wfp.write(ss)



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
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-0fx60.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-1482.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-19cp2.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-2940.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-2xx71.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-32694.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-3xx23.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-53054.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-77052.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-8j998.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-b8615.xlsx",
    "/data_autodrive/users/brli/dev_raw_data/erlunche_results/result-v5-y7862.xlsx"
]
run_update(
    "/data_autodrive/auto/subdivise/result_v5/obj_ids.json",
    "/data_autodrive/users/brli/dev_raw_data/xingrenerlunche_segs.xlsx",
    filters,
    "/data_autodrive/auto/subdivise/result_v6",
    "/data_autodrive/auto/subdivise/20240305"
)
