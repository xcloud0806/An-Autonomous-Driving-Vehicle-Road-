import os, sys
import json
import torch


def cut_segs(clip_id, car_name, tgt_seg_path, force):
    part_num = torch.cuda.device_count()
    segs = os.listdir(tgt_seg_path)
    segs.sort()
    seg_cnt = len(segs)
    # num_per_part = int(seg_cnt / part_num)

    seg_lists = []
    for idx in range(part_num):
        seg_list = []
        seg_lists.append(seg_list)
    
    use_seg_cnt = 0
    for idx, seg in enumerate(segs):
        seg_list_idx = idx % part_num
        rec_path =  os.path.join(tgt_seg_path, seg, "reconstruct")
        if os.path.exists(rec_path) and not force:
            continue
        
        use_seg_cnt += 1
        seg_lists[seg_list_idx].append(seg)
    
    for idx in range(part_num):
        seg_list = seg_lists[idx]
        list_file_name = f"./tmp/{car_name}_{clip_id}_part_{idx}.json"
        with open (list_file_name, "w") as fp:
            json.dump(seg_list, fp)
    print(f"Cut {seg_cnt}[use {use_seg_cnt}] to {part_num}")


if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    with open(config_file, "r") as fp:
        run_config = json.load(fp)

    seg_config = run_config["preprocess"]
    force = (seg_config['force'] == "True")
    tgt_seg_path = seg_config["segment_path"]
    rec_cfg = run_config["reconstruction"]
    if rec_cfg["enable"] != "True":
        print(f"{tgt_seg_path} skip reconstruct.")
        sys.exit(0)

    car = seg_config['car']
    data_subfix = os.path.basename(tgt_seg_path)
    os.makedirs("./tmp", exist_ok=True)

    cut_segs(data_subfix, car, tgt_seg_path, force)

