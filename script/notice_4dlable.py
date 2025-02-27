import os,sys
import json
curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(os.path.dirname(curr_path))
print(curr_dir)
sys.path.append(curr_dir)
from utils import mail_handle

def node_main(run_config):
    deploy_cfg = run_config["deploy"]
    subfix = deploy_cfg['data_subfix']
    src_deploy_root = deploy_cfg["clip_submit_data"]
    anno_path = os.path.join(src_deploy_root, subfix)
    seg_names = os.listdir(anno_path)
    seg_config = run_config["preprocess"]
    car_name = seg_config['car']
    lmdb_deploy_path = deploy_cfg['tgt_rdg_path']

    mail_handle.send(
        f"[DR_NOTICE]{car_name}.{subfix} total {len(seg_names)} transit to RDG DONE",
        f"path is :\n\t LMDB:{lmdb_deploy_path};",
        ["xuanliu11@iflytek.com", "xbchang2@iflytek.com", "liyang16@iflytek.com"]
    )

if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)
    node_main(run_config)