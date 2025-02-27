import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm
import json
sys.path.append("../utils")
from db_utils import query_seg
from loguru import logger
from multiprocessing.pool import Pool

xlsx_file = "/data_autodrive/users/brli/dev_raw_data/seleted_obs_segments.xlsx"
logger.add("search_segs.log")

DST_ROOT = "/data_autodrive/auto/label_4d/first"
DST_INFO_ROOT = "/data_autodrive/auto/label_4d/first/info"

def handle_seg(idx, total, segid, src, dst, dst_info):
    print(f">>> [{idx}/{total}] {segid} going...")
    # src_info =  os.path.join(src, f"{segid}_infos.json")
    # shutil.copy(src_info, dst_info)
    shutil.copytree(src, dst)

def multi_process_error_callback(error):
    # get the current process
    process = os.getpid()
    # report the details of the current process
    print(f"Callback Process: {process}, Exeption {error}", flush=True)

if __name__ == "__main__":
    df = pd.read_excel(xlsx_file)
    use_count = 0
    total_count = df.shape[0]
    pool = Pool(processes=16)
    for idx, row in df.iterrows():
        segid, objcnt, speed, daynight = row
        res = query_seg([segid])

        res_cnt = res[0]
        if res_cnt > 0:
            seg_content = res[1][0]
            seg_clip_obs_path = seg_content['pathMap']['obstacle3dAnnoDataPath']
            if not os.path.exists(seg_clip_obs_path):
                logger.error(f"{seg_clip_obs_path} not exists")
                continue

            subfix = seg_content['collectionDataDate']
            car_name = seg_content['calibrationCar']
            logger.info(f"{seg_clip_obs_path} going...")

            dst_path = os.path.join(DST_ROOT, car_name, subfix, segid)
            if os.path.exists(dst_path):
                # shutil.rmtree(dst_path)
                logger.warning(f"{dst_path} exists")
                continue
            dst_info_dir = os.path.join(DST_INFO_ROOT, car_name, subfix, segid)
            os.makedirs(dst_info_dir, exist_ok=True, mode=0o777)
            dst_info = os.path.join(DST_INFO_ROOT, car_name, subfix, segid, f"{segid}_infos.json")
            
            use_count += 1
            os.makedirs(os.path.join(DST_ROOT, car_name, subfix), exist_ok=True, mode=0o777)
            # print(f">>> {segid} going...")
            # src_info =  os.path.join(seg_clip_obs_path, f"{segid}_infos.json")
            # shutil.copy(src_info, dst_info)
            # shutil.copytree(seg_clip_obs_path, dst_path)
            pool.apply_async(handle_seg, (idx, total_count, segid, seg_clip_obs_path, dst_path, dst_info), error_callback=multi_process_error_callback)

    pool.close()
    pool.join()
    logger.info(f"{use_count}/{total_count}")
            