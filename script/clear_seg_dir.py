import os  
import sys  
import shutil  
import time  
from datetime import datetime, timedelta  

DAY_DIFF_THRESHOLD = 90  

def check_path_exists(path):  
    """检查文件或目录是否存在"""  
    if os.path.exists(path):  
        return True  
    else:  
        return False
def exceed_day_diff_threshold(directory):  
    try:  
        # 获取当前系统时间  
        current_time = datetime.now()  
        print(f"当前系统时间: {current_time}")  

        # 遍历指定目录  
        for root, dirs, files in os.walk(directory):  
            for file in files:  
                file_path = os.path.join(root, file)  
                # 获取文件的最后修改时间  
                mod_time = os.path.path.getmtime(file_path)  
                # 转换为可读的时间格式  
                file_mod_time = datetime.fromtimestamp(mod_time)  

                # 计算时间差  
                time_difference = current_time - file_mod_time  
                days_difference = time_difference.days  

                # 比较文件最后修改时间与当前时间  
                if days_difference > DAY_DIFF_THRESHOLD:  
                    print(f"文件: {file_path}, 最后修改时间: {file_mod_time}, 相差天数: {days_difference}")  
                    return True  

        return False  
    except Exception as e:  
        print(f"获取文件修改时间失败: {e}")  
        return False
    

if __name__ == "__main__":  
    if len(sys.argv) != 2:  
        print(" wrong number of parameters! ")  
        sys.exit(-1)  

    txt_file = sys.argv[1]  

    with open(txt_file, "r") as file:  
        for line in file:  
            data_path = line.strip().split('\t')[1]  
            abs_data_path = data_path  

            last_slash_index = data_path.rfind('/')  
            temp_data_path = data_path[:last_slash_index]  
            deploy_path = temp_data_path.replace("/data/", "/deploy/")  
            abs_deploy_path = deploy_path + "/lidar"  

            ret = exceed_day_diff_threshold(abs_data_path)  

            if not ret:  
                continue  

            try:  
                shutil.rmtree(abs_data_path)  
                print(f"删除目录 '{abs_data_path}' 成功")  
            except Exception as e:  
                print(f"删除目录 '{abs_data_path}' 失败，错误信息: {e}")