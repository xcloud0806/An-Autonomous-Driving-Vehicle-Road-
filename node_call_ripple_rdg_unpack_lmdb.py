import time
import requests
import json
from loguru import logger
import os, sys

commit_url = 'http://www.ripples.airdg/restApi/dragon/v1/bizFlow/exec'
query_url = 'http://www.ripples.airdg/restApi/dragon/v1/bizFlow/info'
headers = {
    'Authorization': 'Basic YXllcnM6UzNDYjlZa01jSDRydkw1WA=='
}

def node_main(config_file):
    if not os.path.exists(config_file):
        logger.error(f"{config_file} Not Exists.")
        sys.exit(1)

    run_config = json.load(open(config_file, 'r'))
    deploy_cfg = run_config["deploy"]
    subfix = deploy_cfg['data_subfix']
    src_deploy_root = deploy_cfg['tgt_rdg_path']
    seg_config = run_config["preprocess"]
    car_name = seg_config['car']
    task_name = f"{car_name}.{subfix}.unpack_lmdb"
    data = {
        "bizName": task_name,
        "bizType": "TASK",
        "categoryId": "665edbc25e14c637e9146efb",
        "templateId": "665edcc15e14c637e9146efe",
        "enableEmail": "true",
        "emailToUsers": [
            # "brli@iflytek.com",
            "xbchang2@iflytek.com",
            "xuanliu11@iflytek.com"
        ]
    }

    # files = {'file':("config_file", open(config_file, "rb"))}
    files = {'fileParams[config_file]': open(config_file, "rb")}

    response = requests.post(commit_url, headers=headers, data=data, files=files)
    ss = json.loads(response.text)
    ret_code = ss['code']
    if ret_code != 0:
        reason = ss['message']
        logger.error(f"{task_name} Failed for {reason}.")
        sys.exit(1)
    
    bizId = ss['data']['bizId']
    logger.info(f"{task_name} success submit to unpack, source: {src_deploy_root} | bizId: {bizId}")
    # while True:
    #     response = requests.post(query_url, headers=headers, data={"bizId": bizId, "bizType": "TASK"})
    #     ss = json.loads(response.text)
    #     status = ss['code']
    #     if status != 0:
    #         reason = ss['message']
    #         logger.error(f"{task_name} Failed for {reason}.")
    #         time.sleep(10)
    #         continue
        
    #     ret_code = ss['code']
    #     if ret_code == 0:
    #         status = ss['data']['status']
    #         if status == "SUCCESS":
    #             logger.info(f"{task_name} success.")
    #             return 
    #         elif status == "FAILED":
    #             logger.error(f"{task_name} Failed.")
    #             sys.exit(1)
    #         else:
    #             logger.info(f"{task_name} status: {status}.")
    #     time.sleep(10)
    #     continue
        

if  __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    work_temp_dir = os.path.dirname(config_file)
    logger.add(os.path.join(work_temp_dir, "ripple_unpack_lmdb.log"))

    node_main(config_file)
    sys.exit(0)  
