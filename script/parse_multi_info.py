import os
import json
from pandas import DataFrame
from tqdm import tqdm

paths = [
    # car, path
    ("sihao_7xx65", "/data_cold2/origin_data/sihao_7xx65/custom_coll/frwang_chadaohuichu/day"),
    ("sihao_7xx65", "/data_cold2/origin_data/sihao_7xx65/custom_coll/frwang_chengshilukou"),
    ("sihao_7xx65", "/data_cold2/origin_data/sihao_7xx65/custom_coll/T_lukou_multifo"),
    ("chery_13484", "/data_cold2/origin_data/chery_13484/custom_coll/frwang_chadaohuichu/day"),
    ("chery_13484", "/data_cold2/origin_data/chery_13484/custom_coll/frwang_chadaohuichu/night"),
    ("sihao_1482", "/data_cold2/origin_data/sihao_1482/custom_coll/frwang_chadaohuichu"),
    ("sihao_1482", "/data_cold2/origin_data/sihao_1482/custom_coll/frwang_chadaohuichu/day"),
    ("sihao_1482", "/data_cold2/origin_data/sihao_1482/custom_coll/frwang_chengshilukou"),
    ("sihao_19cp2", "/data_cold2/origin_data/sihao_19cp2/custom_coll/frwang_chadaohuichu"),
    ("sihao_19cp2", "/data_cold2/origin_data/sihao_19cp2/custom_coll/frwang_chengshilukou"),
    ("sihao_19cp2", "/data_cold2/origin_data/sihao_19cp2/custom_coll/frwang_hf_wh_zadaokou"),
    ("sihao_19cp2", "/data_cold2/origin_data/sihao_19cp2/custom_coll/T_lukou_multifo"),
    ("sihao_27en6", "/data_cold2/origin_data/sihao_27en6/custom_coll/frwang_chadaohuichu"),
    ("sihao_27en6", "/data_cold2/origin_data/sihao_27en6/custom_coll/frwang_chadaohuichu/day"),
    ("sihao_27en6", "/data_cold2/origin_data/sihao_27en6/custom_coll/frwang_chadaohuichu/night"),
    ("sihao_27en6", "/data_cold2/origin_data/sihao_27en6/custom_coll/frwang_chengshilukou"),
    ("sihao_27en6", "/data_cold2/origin_data/sihao_27en6/custom_coll/frwang_chadaohuiru/day"),
    ("sihao_27en6", "/data_cold2/origin_data/sihao_27en6/custom_coll/frwang_chadaohuiru/night"),
    ("sihao_27en6", "/data_cold2/origin_data/sihao_27en6/custom_coll/T_lukou_multifo"),
    ("sihao_27en6", "/data_cold2/origin_data/sihao_27en6/custom_coll/ztwen_yewandache"),
    ("sihao_2xx71", "/data_cold2/origin_data/sihao_2xx71/custom_coll/frwang_chengshilukou"),
    ("sihao_2xx71", "/data_cold2/origin_data/sihao_2xx71/custom_coll/T_lukou_multifo"),
    ("sihao_36gl1", "/data_cold2/origin_data/sihao_36gl1/custom_coll/frwang_chadaohuichu"),
    ("sihao_36gl1", "/data_cold2/origin_data/sihao_36gl1/custom_coll/frwang_chadaohuichu/day"),
    ("sihao_36gl1", "/data_cold2/origin_data/sihao_36gl1/custom_coll/frwang_chadaohuichu/night"),
    ("sihao_36gl1", "/data_cold2/origin_data/sihao_36gl1/custom_coll/frwang_feiduichenlukou"),
    ("sihao_47465", "/data_cold2/origin_data/sihao_47465/custom_coll/frwang_chadaohuichu/day"),
    ("sihao_47465", "/data_cold2/origin_data/sihao_47465/custom_coll/frwang_chadaohuiru/day"),
    ("sihao_47465", "/data_cold2/origin_data/sihao_47465/custom_coll/frwang_chadaohuiru/night"),
    ("sihao_47466", "/data_cold2/origin_data/sihao_47466/custom_coll/frwang_chadaohuichu/day"),
    ("sihao_47466", "/data_cold2/origin_data/sihao_47466/custom_coll/frwang_chadaohuiru/day"),
    ("sihao_47466", "/data_cold2/origin_data/sihao_47466/custom_coll/frwang_chadaohuiru/night"),
    ("sihao_y7862", "/data_cold2/origin_data/sihao_y7862/custom_coll/frwang_chadaohuichu"),
    ("sihao_y7862", "/data_cold2/origin_data/sihao_y7862/custom_coll/frwang_chadaohuichu/day"),
    ("sihao_y7862", "/data_cold2/origin_data/sihao_y7862/custom_coll/frwang_feiduichenlukou"),
]

def parse_coll(car, coll_path):
    ret = []
    if not os.path.exists(coll_path):
        return ret
        
    if not os.path.isdir(coll_path):
        return ret

    dates = os.listdir(coll_path)
    dates.sort()
    
    for date in dates:
        date_path = os.path.join(coll_path, date)
        if not os.path.isdir(date_path):
            continue
        
        clips = os.listdir(date_path)
        clips.sort()
        for clip in clips:
            clip_path = os.path.join(date_path, clip)
            if not os.path.isdir(clip_path):
                continue
            
            info_path = os.path.join(clip_path, "multi_info.json")
            if not os.path.exists(info_path):
                continue
            
            with open(info_path, "r") as f:
                info = json.load(f)
                main_segs = info[clip].get("main_clip_path", [])
                if isinstance(main_segs, str):
                    if len(main_segs) > 0:
                        main_segs = [main_segs]
                    else:
                        main_segs = []
                aid_segs = info[clip].get("clips_path", [])
                if isinstance(aid_segs, str):
                    if len(aid_segs) > 0:
                        aid_segs = [aid_segs]
                    else:
                        aid_segs = []
                clip_segs = main_segs + aid_segs
                for seg in clip_segs:
                    seg_id = os.path.basename(seg)
                    ret.append({
                        "car": car,
                        "coll": clip,
                        "seg": seg_id,
                        "subfix": date,
                        "segment_path": seg
                    })

    return ret

def main():
    colls = []
    for path in tqdm(paths):
        car, coll_path = path
        tqdm.write(coll_path)
        coll_info = parse_coll(car, coll_path)
        colls.extend(coll_info)
    
    df = DataFrame(colls, columns=["car", "coll", "seg", "subfix", "segment_path"])
    df.to_csv("./multi_coll_info.csv", index=False)

if __name__ == "__main__":
    main()