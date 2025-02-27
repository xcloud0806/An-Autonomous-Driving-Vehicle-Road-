import os
import sys
import json
import numpy as np
import pandas as pd

def main(excel_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, mode=0o775, exist_ok=True)

    # 读取所有Sheet的数据  
    all_sheets = pd.read_excel(excel_file, sheet_name=None)  # 返回一个字典，键为Sheet名，值为DataFrame  

    # 遍历所有Sheet  
    for sheet_name, data in all_sheets.items():  
        # print(f"Sheet: {sheet_name}")

        if '数据内网数据' in data.columns and '标注后数据地址' in data.columns:
            data_inner = data[['数据内网数据', '标注后数据地址']]

            # 将数据转换为字典列表  
            data_dict = data_inner.dropna().to_dict(orient='records')  

            # 保存为JSON文件  
            json_file_path = output_dir + '/' + sheet_name + '.json' 
            with open(json_file_path, 'w', encoding='utf-8') as json_file:  
                json.dump(data_dict, json_file, ensure_ascii=False, indent=4)  

            print(f"Sheet: {sheet_name} done")
        else:
            print(f"Sheet: {sheet_name} not have '数据内网数据' and '标注后数据地址'")


"""  
示例:
    python parse_excel.py xlsx_file output_dir
"""

if __name__ == "__main__":

    if len(sys.argv) !=3:
        print("Usage: python parse_excel.py.py xlsx_file output_dir")
        sys.exit(1)
    
    xlsx_file = sys.argv[1]
    if not os.path.exists(xlsx_file):
        print(f"{xlsx_file} Not Exists.")
        sys.exit(1)

    output_dir = sys.argv[2]
    main(xlsx_file, output_dir)
    sys.exit(0)