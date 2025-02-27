import json  
import sys
import os  

""" 
示例:
python script/export_bpearl_json.py CHERY_E0Y_18047_lidar_bpearl_lidar_to_main_lidar.json   /data/jfzhang24/devel/temp/
        参数1为bpearl的外参json文件
        参数2为输出目录
    该脚本会在输出目录下生成bpearls文件夹,文件夹下包含四个json文件
        lidar_to_front.json
        lidar_to_left.json
        lidar_to_rear.json
        lidar_to_right.json
"""

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("参数个数不正确,请输入两个参数:参数1为bpearl的外参json文件,参数2为输出目录")
        sys.exit(1)

    input_file = sys.argv[1]
    output_directory = sys.argv[2]

    save_dir = os.path.join(output_directory, "bpearls")
    os.makedirs(save_dir, exist_ok=True)

    with open(input_file, 'r') as f:  
        data = json.load(f) 

    keys_to_save = {
        'bpearl_lidar_front':'lidar_to_front',
        'bpearl_lidar_left':'lidar_to_left',
        'bpearl_lidar_rear':'lidar_to_rear',
        'bpearl_lidar_right':'lidar_to_right'
    }

    for key in keys_to_save:  
        if key in data:  
            output_file = os.path.join(save_dir, f'{keys_to_save[key]}.json')  
            with open(output_file, 'w') as f:  
                json.dump(data[key], f, indent=4)  # 保存为JSON文件，格式化输出  
            print(f'Saved {key} to {output_file}')  
        else:  
            print(f'Key {key} not found in the input data.')
            sys.exit(1)
 
