import os
import sys
import time
import subprocess
import threading
import socket
curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(os.path.dirname(curr_path))
sys.path.append(curr_dir)

from python import init_root_logger, update_logger_rotate_file_handler
from python import get_redis_rcon, read_msg, acquire_lock_with_timeout, release_lock
from python import DataPool, AUTO_RUN_LOCK, AUTO_RUN, MONGODB_LOCK
from python import sender

LOG_PATH="/data_autodrive/users/xuanliu7/work_tmp/auto_run_log"
LOG_PATH="/data_autodrive/users/xuanliu7/data_handle_auto_v2/cv2_zj_DataPlatform_Toolchain/auto_run_log"

def get_local_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def update_root_logger(logger, task):
    local_ip = get_local_ip()
    car_name = str(task["car_name"])
    date = str(task["date"])
    log_date_path= os.path.join(LOG_PATH, date)
    if not os.path.exists(log_date_path):
        os.makedirs(log_date_path)
    file_name = f"{local_ip}@{car_name}-{date}"
    log_file = os.path.join(log_date_path, file_name + ".log")
    while os.path.exists(log_file):
        time.sleep(1)
        log_file = os.path.join(log_date_path, file_name + "_" + str(int(time.time())) + ".log")
    update_logger_rotate_file_handler(logger, log_file)    

def init_task_status(task:dict):
    current_timestamp = time.time()  
    local_time = time.localtime(current_timestamp)  
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time) 
    task["start_time"] = formatted_time
    task["ip"] = get_local_ip()
    task["status"] = "processing"

def update_task_success_status(task:dict):
    current_timestamp = time.time()  
    local_time = time.localtime(current_timestamp)  
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time) 
    task["status"] = "success"
    task["end_time"] = formatted_time

def monitor_output(stream, label):
    """监控并打印子进程输出，特别留意'exception'和'error'关键词"""
    for line in iter(stream.readline, b''):
        line_str = line.strip()
        logger.info(f"{label}: {line_str}")
        if 'exception' in line_str.lower() or 'error' in line_str.lower():
            logger.info(f"检测到错误或异常: {line_str}")

def run(task, rcon, datapool):
    logger.info("----------------------------------------------------------")
    logger.info(f"Receive task: {task['task_id']}, Running ...")
    car_name = str(task["car_name"])
    date = str(task["date"])
    address = str(task["address"])
    calib_date = str(task["calib_date"])
    task_name = str(task["task_name"])
    args = [car_name, date, address, calib_date, task_name]
    try:
        shell_path = os.path.join(
            os.path.dirname((os.path.dirname(__file__))),
            "templates",
            "template_pipeline_task_auto_run.sh"
        )
        # if os.path.exists(shell_path):
            # print(f"Exist shell path: {shell_path}")
        # shell_path = "/data/xlzhai3/work/develop_lianyi/cv2_zj_DataPlatform_Toolchain/python/tool_auto_run_pipeline/test.sh"
        subprocess.run([shell_path] + args)
        # 使用Popen启动子进程，同时监控stdout和stderr
        # with subprocess.Popen([shell_path] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        #     # 分别用线程监控stdout和stderr
        #     stdout_thread = threading.Thread(target=monitor_output, args=(process.stdout, "STDOUT"))
        #     stderr_thread = threading.Thread(target=monitor_output, args=(process.stderr, "STDERR"))
            
        #     # 启动监控线程
        #     stdout_thread.start()
        #     stderr_thread.start()
            
        #     # 等待子进程结束
        #     process.wait()
            
        #     # 确保所有输出都已处理完毕
        #     stdout_thread.join()
        #     stderr_thread.join()
        # print("test send email")
        time.sleep(10)
        update_task_success_status(task)
        while True:
            lock_val_mongodb = acquire_lock_with_timeout(rcon, MONGODB_LOCK)
            if not lock_val_mongodb or lock_val_mongodb is None:
                time.sleep(1)
                continue
            datapool.mongodb.update(task, datapool.collection)
            release_lock(rcon, MONGODB_LOCK, lock_val_mongodb)
            break
        logger.info("Success, run task: {}".format(task["task_id"]))
        
    except Exception as e:
        logger.error(f"error, {e}")
        task["status"] = "run_error"
        
        while True:
            lock_val_mongodb = acquire_lock_with_timeout(rcon, MONGODB_LOCK)
            if not lock_val_mongodb or lock_val_mongodb is None:
                time.sleep(1)
                continue
            datapool.mongodb.update(task, datapool.collection)
            release_lock(rcon, MONGODB_LOCK, lock_val_mongodb)
            break
        logger.info("Failed, run task: {}".format(task["task_id"]))
        
    sender.connect()
    sender.send(
        f"[AUTO_RUN]{car_name}.{date} {task['status']}",
        f"path is :\n\t frame_path:{address};\n\t calib_date:{calib_date}"
    )
    sender.disconnect()
        # ["xuanliu11@iflytek.com", "xbchang2@iflytek.com", "liyang16@iflytek.com", "taoliu19@iflytek.com"]
    
    

def call_back(logger, rcon, datapool:DataPool):
    while True:
        if rcon.llen(AUTO_RUN) == 0:
            logger.info("no new messages")
            time.sleep(5)
            continue
        v = acquire_lock_with_timeout(rcon, AUTO_RUN_LOCK)
        if not v or v is None:
            continue
        task = read_msg(rcon, AUTO_RUN)
        update_root_logger(logger, task)
        init_task_status(task)

        while True:
            lock_val_mongodb = acquire_lock_with_timeout(rcon, MONGODB_LOCK)
            if not lock_val_mongodb or lock_val_mongodb is None:
                time.sleep(1)
                continue
            datapool.mongodb.update(task, datapool.collection)
            release_lock(rcon, MONGODB_LOCK, lock_val_mongodb)
            break
        # datapool.mongodb.update(task, datapool.collection)
        run(task, rcon, datapool)
        release_lock(rcon, AUTO_RUN_LOCK, v)

if __name__ == "__main__":
    
    rcon = get_redis_rcon()
    datapool = DataPool()
    logger = init_root_logger()
    while True:
        call_back(logger, rcon, datapool)

