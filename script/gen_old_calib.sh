#!/bin/bash

# 检查参数个数  
if [ $# -ne 2 ]; then  
    echo "错误：需要提供两个参数。"  
    echo "用法：./gen_old_calib /xxx/xxx/calib_params.json /xxx/xxx/output_dir"
    exit 1  
fi  

source /home/brli/conda.sh pcl
python3 calib_file_conversion.py $1 $2