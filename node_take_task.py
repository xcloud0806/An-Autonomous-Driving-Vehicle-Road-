from utils import (
    get_redis_rcon,
    acquire_lock_with_timeout,
    release_lock,
    RECONSTRUCT_QUEUE, 
    RECONSTRUCT_PRIORITY_QUEUE,
    push_msg,
    read_msg,
    RECONSTRUCT_LOCK_KEY,
    # HPP_LOCK_KEY,
    # HPP_PRIORITY_QUEUE,
    # HPP_QUEUE,
)

import os, sys
import json
import numpy as np
from node_reconstruct_v1 import run_reconstruct_v2, run_reconstruct_parking
import traceback as tb
import time
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)

# 创建一个RotatingFileHandler，设置最大文件大小为20MB，最多保留5个文件
handler = RotatingFileHandler("reconstruct.log", maxBytes=20*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 将handler添加到日志记录器
logger.addHandler(handler)

MODE='common'

if __name__ == '__main__':
    GPUID = 0
    if len(sys.argv) > 1:
        GPUID = int(sys.argv[1])

    rcon = get_redis_rcon()
    task_queue = RECONSTRUCT_QUEUE
    task_prior_queue = RECONSTRUCT_PRIORITY_QUEUE
    task_lock_key = RECONSTRUCT_LOCK_KEY

    while True:
        if rcon.llen(task_queue) == 0 and rcon.llen(task_prior_queue) == 0:
            time.sleep(10)
            continue
        v = acquire_lock_with_timeout(rcon, task_lock_key)
        if not v or v is None:
            continue
        if rcon.llen(task_prior_queue) > 0:
            task = read_msg(rcon, task_prior_queue)
        else:
            task = read_msg(rcon, task_queue)
        release_lock(rcon, task_lock_key, v)
        
        config_file = task['config']
        work_temp_dir = os.path.dirname(config_file)
        spec_segs = task['specs']
        spec = ",".join(spec_segs)

        with open(config_file, "r") as fp:
            run_config = json.load(fp)

        seg_config = run_config["preprocess"]
        tgt_seg_path = seg_config["segment_path"]
        rec_cfg = run_config["reconstruction"]
        if rec_cfg['enable'] != "True":
            print(f"{tgt_seg_path} skip reconstruct.")
            break
        gpuid = rec_cfg["gpuid"]
        version = run_config['reconstruction']['version']
        rec_cfg['force'] = seg_config['force']

        if 'min_distance' not in rec_cfg.keys():
            rec_cfg['min_distance'] = 150

        try:
            # logger.info(f">>>> +{str(datetime.now())}+ Start reconstruct {work_temp_dir}.{spec_segs}")
            if ('parking' not in rec_cfg.keys()) or (rec_cfg['parking'].lower() == 'false'):
                if version == '2':
                    logger.info(f">>>> +{str(datetime.now())}+ Start reconstruct v2 {work_temp_dir}.{spec_segs}")
                    run_reconstruct_v2(rec_cfg, GPUID, tgt_seg_path, spec_segs)
                else:
                    raise ValueError(f'version should be 1 or 2')
            elif rec_cfg['parking'].lower() == 'true':
                logger.info(f">>>> +{str(datetime.now())}+ Start parking reconstruct {work_temp_dir}.{spec_segs}")
                run_reconstruct_parking(rec_cfg, GPUID, tgt_seg_path, spec_segs)
            # run_reconstruct_v2(rec_cfg, GPUID, tgt_seg_path, spec_segs)
        except:
            tb.print_exc()
            logger.error(f"!!!! +{str(datetime.now())}+ reconstruct {work_temp_dir}.{spec_segs} FATAL FAILED")

        logger.info(f"<<<< +{str(datetime.now())}+ Finish reconstruct {work_temp_dir}.{spec_segs}")
        os.makedirs(os.path.join(work_temp_dir, "reconstruct_status"), exist_ok=True)
        for seg_id in spec_segs:
            done_flag = os.path.join(work_temp_dir, "reconstruct_status", f"{seg_id}:DONE")
            with open(done_flag, "w") as wfp:
                wfp.write(f"{seg_id}:DONE")

