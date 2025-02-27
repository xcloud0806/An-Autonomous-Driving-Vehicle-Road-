from multiprocessing.pool import Pool
import os, sys
import json


def run_reconstruct_v1(rec_cfg, gpuid, tgt_seg_path, specs=[]):
    from python import func_reconstruct_v1

    pool = Pool(processes=4)
    segs = os.listdir(tgt_seg_path)
    for seg in sorted(segs):
        if len(specs) > 0 and seg not in specs:
            continue
        print(f">>> Reconstruct {seg}")
        meta_file = os.path.join(tgt_seg_path, seg, "meta.json")
        seg_path = os.path.join(tgt_seg_path, seg)
        recon_dir = os.path.join(seg_path, 'reconstruct')
        if os.path.exists(recon_dir):
            if (rec_cfg['force'].lower() == 'false'):
                if (len(os.listdir(recon_dir)) >= 4): # 4: images, rgb, height, transform_matrix
                    # skip this clip
                    print('\033[0;32;40mThis clip has been reconstructed, skip...\033[0m')
                    continue
                else:
                    # delete old reconstruct folder
                    print('\033[0;33;40mThis clip has not been reconstructed completely, delete and reconstruct...\033[0m')
                    os.system('rm -rf {}'.format(recon_dir))  
            else:
                # delete old reconstruct folder
                print('\033[0;33;40mThis clip has been reconstructed, delete and reconstruct...\033[0m')
                os.system('rm -rf {}'.format(recon_dir))
        # func_reconstruct_v1(seg_path, gpuid, meta_file, rec_cfg)
        pool.apply_async(func_reconstruct_v1, (seg_path, gpuid, meta_file, rec_cfg))
    pool.close()
    pool.join()


def run_reconstruct_v2(rec_cfg, gpuid, tgt_seg_path, specs=[]):
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpuid)
    from python import func_reconstruct_v2

    camera_names_list = [rec_cfg["map_camera_name"], ]
    if rec_cfg["map_camera_name_add"] != '':
        camera_names_list.append(rec_cfg["map_camera_name_add"])
    segs = os.listdir(tgt_seg_path)
    for seg in sorted(segs):
        if len(specs) > 0 and seg not in specs:
            continue
        print(f">>> Reconstruct V2 {seg}")
        meta_path = os.path.join(tgt_seg_path, seg, "meta.json")
        seg_path = os.path.join(tgt_seg_path, seg)
        recon_dir = os.path.join(seg_path, 'reconstruct')
        if os.path.exists(recon_dir):
            if (rec_cfg['force'].lower() == 'false'):
                if (len(os.listdir(recon_dir)) >= 5): # 5: images, rgb, height, semantic, transform_matrix
                    # skip this clip
                    print('\033[0;32;40mThis clip has been reconstructed, skip...\033[0m')
                    continue
                else:
                    # delete old reconstruct folder
                    print('\033[0;33;40mThis clip has not been reconstructed completely, delete and reconstruct...\033[0m')
                    os.system('rm -rf {}'.format(recon_dir))                    
            else:
                # delete old reconstruct folder
                print('\033[0;33;40mThis clip has been reconstructed, delete and reconstruct...\033[0m')
                os.system('rm -rf {}'.format(recon_dir))
        try:
            func_reconstruct_v2(meta_path, camera_names_list, output_dir=seg_path, min_distance=rec_cfg['min_distance'])
        except Exception as e:
            print('\033[0;31;40mRuntime Error when reconstruct {}, {}\033[0m'.format(seg, e))
 

def run_reconstruct_parking(rec_cfg, gpuid, tgt_seg_path, specs=[]):
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpuid)
    from python import func_reconstruct_parking

    camera_names_list = [] # 泊车场景固定使用周视6相机
    
    segs = os.listdir(tgt_seg_path)
    for seg in sorted(segs):
        if len(specs) > 0 and seg not in specs:
            continue
        print(f">>> Reconstruct parking {seg}")
        meta_path = os.path.join(tgt_seg_path, seg, "meta.json")
        seg_path = os.path.join(tgt_seg_path, seg)
        recon_dir = os.path.join(seg_path, 'reconstruct')
        if os.path.exists(recon_dir):
            if (rec_cfg['force'].lower() == 'false'):
                if (len(os.listdir(recon_dir)) >= 4): # 4: images, normal vis, height, transform_matrix
                    # skip this clip
                    print('\033[0;32;40mThis clip has been reconstructed, skip...\033[0m')
                    continue
                else:
                    # delete old reconstruct folder
                    print('\033[0;33;40mThis clip has not been reconstructed completely, delete and reconstruct...\033[0m')
                    os.system('rm -rf {}'.format(recon_dir))                    
            else:
                # delete old reconstruct folder
                print('\033[0;33;40mThis clip has been reconstructed, delete and reconstruct...\033[0m')
                os.system('rm -rf {}'.format(recon_dir))
        try:
            func_reconstruct_parking(meta_path, camera_names_list)
        except Exception as e:
            print('\033[0;31;40mRuntime Error when reconstruct {}, {}\033[0m'.format(seg, e))


if __name__ == "__main__":
    config_file = "./utils/sample_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"{config_file} Not Exists.")
        sys.exit(1)

    spec_segs = []
    if len(sys.argv) > 2:
        spec = sys.argv[2]

        if os.path.isfile(spec) and os.path.exists(spec):
            _spec_segs = json.load(open(spec, "r"))
            if type(_spec_segs) is list:
                spec_segs = _spec_segs
        else:
            if type(spec) is str:
                spec_segs = spec.split(",")

    with open(config_file, "r") as fp:
        run_config = json.load(fp)

    seg_config = run_config["preprocess"]
    tgt_seg_path = seg_config["segment_path"]
    rec_cfg = run_config["reconstruction"]
    if rec_cfg['enable'] != "True":
        print(f"{tgt_seg_path} skip reconstruct.")
        sys.exit(0)
    gpuid = rec_cfg["gpuid"]
    version = run_config['reconstruction']['version']
    rec_cfg['force'] = seg_config['force']

    if 'min_distance' not in rec_cfg.keys():
        rec_cfg['min_distance'] = 150

    if ('parking' not in rec_cfg.keys()) or (rec_cfg['parking'].lower() == 'false'):
        if version == '1':
            run_reconstruct_v1(rec_cfg, gpuid, tgt_seg_path)
        elif version == '2':
            run_reconstruct_v2(rec_cfg, gpuid, tgt_seg_path)
        else:
            raise ValueError(f'version should be 1 or 2')
    elif rec_cfg['parking'].lower() == 'true':
        run_reconstruct_parking(rec_cfg, gpuid, tgt_seg_path)
    else:
        raise ValueError(f'Config variable [reconstruction/parking] should be True or False')
