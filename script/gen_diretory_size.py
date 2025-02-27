import os
import shutil

def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if 'input' not in dirpath:
                continue
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

dirs = [
    # "/data_cold/origin_data/sihao_3xx23/common/",
    # "/data_cold/origin_data/sihao_3xx23/custom/",
    # "/data_cold/origin_data/sihao_3xx23/validate/",
    "/data_cold/origin_data/zeer_uf327/common",
    "/data_cold/origin_data/zeer_uf327/custom"
]

for dir in dirs:
    total_size = 0
    for item in os.listdir(dir):
        full_path = os.path.join(dir,item)
        dir_size = get_directory_size(full_path)
        dir_size_gb = dir_size/1024/1024/1024
        print("{}, {:.2f} GB".format(full_path, dir_size_gb))
        total_size += dir_size_gb
    print(f"{dir} <-> {total_size} GB")