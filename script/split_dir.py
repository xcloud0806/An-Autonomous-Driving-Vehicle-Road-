import os
import sys
import re
import shutil
from multiprocessing import Pool

# DIR_LIST = ['20241202', '20241203', '20241204', '20241205']
# DIR_LIST = ['20241203_n', '20241205_d', '20241205_n']
# DIR_LIST = ['20241202_n', '20241202_d', '20241203_d', '20241204_n', '20241204_d']
# DIR_LIST = ['20241203_d', '20241204_n', '20241206_d']
# DIR_LIST = ['20241206_d', '20241206_n', '20241208_n']
# DIR_LIST = ['20241205_n', '20241207_d', '20241207_n', '20241210_n']

#xuzhoumeixing
DIR_LIST = ['20241213_d', '20241214_n', '20241216_d', '20241216_n', '20241217_d']

def create_directory(directory):  
    """  
    创建一个目录，如果目录不存在。  

    :param directory: 要创建的目录路径  
    """  
    try:  
        if not os.path.exists(directory):  
            os.makedirs(directory, mode=0o775, exist_ok=True)
            # print(f"目录 '{directory}' 已创建。")  
    except OSError:  
        print(f"创建目录 '{directory}' 失败。")
        exit(1)
# 自定义排序函数  
def sort_key(s):  
    # 使用正则表达式提取日期时间和seg数字  
    match = re.search(r'_(\d{8})-(\d{2}-\d{2}-\d{2})_seg(\d+)', s)  
    if match:  
        date_str = match.group(1)  # 日期部分  
        time_str = match.group(2)  # 时间部分  
        seg_number = int(match.group(3))  # seg后面的数字  
        return (date_str, time_str, seg_number)  
    return (None, None, None)  # 如果没有匹配到，则返回默认值 

def copy_directory(src, dst):  
    """  
    拷贝目录及其子目录和文件到另一个目录。  

    :param src: 源目录路径  
    :param dst: 目标目录路径  
    """  
    try:
        if isinstance(src, list):
            for d in src:
                seg_name = d.split('/')[-1]
                new_dst = os.path.join(dst, seg_name)
                os.makedirs(new_dst, mode=0o775, exist_ok=True)
                copy_directory(d, new_dst)
            return
        
        if not os.path.exists(src):  
            print(f"源目录 '{src}' 不存在。")  
            return
        
        shutil.copytree(src, dst, dirs_exist_ok=True)  
        print(f"成功将 '{src}' 拷贝到 '{dst}'。")  
    except Exception as e:  
        print(f"拷贝过程中发生错误: {e}")

def copy_segment(args):  
    clip_obstacle_seg_dir_path, clip_lane_seg_dir_path, \
    clip_obstacle_annotation_seg_dir_path, \
    clip_lane_4d_result_seg_dir_path, \
    clip_change_lane_seg_dir_list, \
    clip_obstacle_output_path, clip_lane_output_path, \
    clip_obstacle_annotation_output_path, \
    clip_lane_4d_result_output_path, \
    clip_change_lane_output_path = args

    # Perform the directory copy
    copy_directory(clip_obstacle_seg_dir_path, clip_obstacle_output_path) 
    copy_directory(clip_lane_seg_dir_path, clip_lane_output_path)
    copy_directory(clip_obstacle_annotation_seg_dir_path, clip_obstacle_annotation_output_path) 
    copy_directory(clip_lane_4d_result_seg_dir_path, clip_lane_4d_result_output_path)
    # copy_directory(clip_change_lane_seg_dir_list, clip_change_lane_output_path)

def generate_tasks(clip_obstacle_input_dir, sub_dir, clip_obstacle_output,
                    clip_lane_output, clip_obstacle_annotation_4d_result_output,
                    clip_lane_4d_result_output, clip_change_lane_output):
    
    seg_dirs = os.listdir(os.path.join(clip_obstacle_input_dir, sub_dir))  
    sorted_seg_dirs = sorted(seg_dirs, key=sort_key)  

    for index, seg_dir in enumerate(sorted_seg_dirs):  
        clip_obstacle_seg_dir_path = os.path.join(clip_obstacle_input_dir, sub_dir, seg_dir)
        clip_lane_seg_dir_path = clip_obstacle_seg_dir_path.replace("clip_obstacle", "clip_lane")

        clip_obstacle_annotation_seg_dir_path = clip_obstacle_seg_dir_path.replace("clip_obstacle", "clip_obstacle_annotation_4d_result")
        clip_obstacle_annotation_seg_dir_path = clip_obstacle_annotation_seg_dir_path.replace(sub_dir, sub_dir+'_'+sub_dir)

        clip_lane_4d_result_seg_dir_path = clip_obstacle_seg_dir_path.replace("clip_obstacle", "clip_lane_4d_result")

        clip_change_lane_seg_dir_path = clip_obstacle_seg_dir_path.replace("clip_obstacle", "clip_change_lane")
        clip_change_lane_seg_dir_path = clip_change_lane_seg_dir_path.replace(sub_dir, sub_dir + '/meixing_shanghai_data_11v')
        clip_change_lane_seg_dir_path = clip_change_lane_seg_dir_path.replace(seg_dir, seg_dir + '-')
        split_index = index % 4

        clip_obstacle_output_path = os.path.join(clip_obstacle_output, sub_dir + "_" + str(split_index))
        clip_lane_output_path = os.path.join(clip_lane_output, sub_dir + "_" + str(split_index))
        
        clip_obstacle_output_path = os.path.join(clip_obstacle_output_path, seg_dir)
        clip_lane_output_path = os.path.join(clip_lane_output_path, seg_dir)

        clip_obstacle_annotation_output_path = os.path.join(clip_obstacle_annotation_4d_result_output, sub_dir + "_" + str(split_index))
        clip_obstacle_annotation_output_path = os.path.join(clip_obstacle_annotation_output_path, seg_dir)

        clip_lane_4d_result_output_path = os.path.join(clip_lane_4d_result_output, sub_dir + "_" + str(split_index))
        clip_lane_4d_result_output_path = os.path.join(clip_lane_4d_result_output_path, seg_dir)

        clip_change_lane_output_path = os.path.join(clip_change_lane_output, sub_dir + "_" + str(split_index))
        # clip_change_lane_output_path = os.path.join(clip_change_lane_output_path, seg_dir)

        if not os.path.exists(clip_obstacle_seg_dir_path):
            print(f"warning {clip_obstacle_seg_dir_path} Not Exists.")
            continue
        
        if not os.path.exists(clip_lane_seg_dir_path):
            print(f"warning {clip_lane_seg_dir_path} Not Exists.")
            continue

        if not os.path.exists(clip_obstacle_annotation_seg_dir_path):
            print(f"warning {clip_obstacle_annotation_seg_dir_path} Not Exists.")
            continue

        if not os.path.exists(clip_lane_4d_result_seg_dir_path):
            print(f"warning {clip_lane_4d_result_seg_dir_path} Not Exists.")
            continue
        
        last_slash_index = clip_change_lane_seg_dir_path.rfind('/')
        temp_clip_change_lane_seg_dir = clip_change_lane_seg_dir_path[:last_slash_index]
        prefix_str = seg_dir + '-'
        clip_change_lane_seg_dir_list= []
        # for d in os.listdir(temp_clip_change_lane_seg_dir):
        #     if d.startswith(prefix_str) and os.path.isdir(os.path.join(temp_clip_change_lane_seg_dir, d)):
        #         clip_change_lane_seg_dir_list.append(os.path.join(temp_clip_change_lane_seg_dir, d))
        
        # if len(clip_change_lane_seg_dir_list) == 0:
        #     print(f"warning {temp_clip_change_lane_seg_dir} / {prefix_str } Not Exists.")
        #     continue

        create_directory(clip_obstacle_output_path)
        create_directory(clip_lane_output_path)
        create_directory(clip_obstacle_annotation_output_path)
        create_directory(clip_lane_4d_result_output_path)
        # create_directory(clip_change_lane_output_path)

        yield (clip_obstacle_seg_dir_path, clip_lane_seg_dir_path, 
                clip_obstacle_annotation_seg_dir_path, 
                clip_lane_4d_result_seg_dir_path,
                clip_change_lane_seg_dir_list,
                clip_obstacle_output_path, clip_lane_output_path,
                clip_obstacle_annotation_output_path, clip_lane_4d_result_output_path,
                clip_change_lane_output_path)
def process_sub_dir(clip_obstacle_input_dir, sub_dir, clip_obstacle_output,
                    clip_lane_output, clip_obstacle_annotation_4d_result_output,
                    clip_lane_4d_result_output, clip_change_lane_output):  
    go_on = False
    for dir in DIR_LIST:
        if dir not in sub_dir:
            continue
        go_on = True

    if not go_on:
        return

    print(f"Processing sub_dir = {sub_dir}")

    # 使用生成器  
    tasks = generate_tasks(clip_obstacle_input_dir, sub_dir, clip_obstacle_output,
                    clip_lane_output, clip_obstacle_annotation_4d_result_output,
                    clip_lane_4d_result_output, clip_change_lane_output)  

    with Pool() as pool:  
        pool.map(copy_segment, tasks)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_dir.py /data_autodrive/auto/custom/chery_24029/meixing_shanghai_data_11v/clip_obstacle output_dir")
        sys.exit(1)
    
    clip_obstacle_input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(sys.argv[1]):
        print(f"{sys.argv[1]} Not Exists.")
        sys.exit(1)
    
    clip_obstacle_output = os.path.join(output_dir, 'clip_obstacle_split')
    clip_lane_output = os.path.join(output_dir, 'clip_lane_split')
    clip_obstacle_annotation_4d_result_output = os.path.join(output_dir, 'clip_obstacle_annotation_4d_result_split')
    clip_lane_4d_result_output = os.path.join(output_dir, 'clip_lane_4d_result_split')
    clip_change_lane_output = os.path.join(output_dir, 'clip_change_lane_split')

    print(f"clip_obstacle_output = {clip_obstacle_output} clip_lane_output = {clip_lane_output}")
    os.makedirs(clip_obstacle_output, mode=0o775, exist_ok=True)
    os.makedirs(clip_lane_output, mode=0o775, exist_ok=True)
    os.makedirs(clip_obstacle_annotation_4d_result_output, mode=0o775, exist_ok=True)
    os.makedirs(clip_lane_4d_result_output, mode=0o775, exist_ok=True)
    os.makedirs(clip_change_lane_output, mode=0o775, exist_ok=True)

    sub_dirs = os.listdir(clip_obstacle_input_dir)
    for sub_dir in sub_dirs:
        process_sub_dir(clip_obstacle_input_dir, sub_dir, clip_obstacle_output, 
                        clip_lane_output, clip_obstacle_annotation_4d_result_output,
                        clip_lane_4d_result_output, clip_change_lane_output)


