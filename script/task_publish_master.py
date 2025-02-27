"""监控excel任务,再数据库中更新手动执行的任务,将需要自动执行的任务发送到redis 队列
"""
import os
import sys
import time
curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(os.path.dirname(curr_path))
sys.path.append(curr_dir)
from python import get_logger, init_root_logger, update_logger_rotate_file_handler
from python import get_redis_rcon, acquire_lock_with_timeout, release_lock, push_msg
from python import DataPool, AUTO_RUN, AUTO_RUN_LOCK, MONGODB_LOCK
logger = get_logger("tool_publisher")
LOG_PATH="/data_autodrive/users/xuanliu7/work_tmp/auto_run_log"
LOG_PATH="/data_autodrive/users/xuanliu7/data_handle_auto_v2/cv2_zj_DataPlatform_Toolchain/auto_run_log"

def call_back(rcon, datapool:DataPool, task_count):
    """回调函数,将手动执行过的任务的状态更新到数据库中，
    将需要自动执行的任务发送到redis队列
    """
    logger.info("Start, send task to redis")
    if datapool.task_queue:
        for task_id, task in datapool.task_queue.items():
            # 如果状态为handle_done，则插入mongodb，表示该批数据是手动处理的
            if int(task["manual_status"]) == 1:
                task["status"] = "handle_done"
                while True:
                    lock_val_mongodb = acquire_lock_with_timeout(rcon, MONGODB_LOCK)
                    if not lock_val_mongodb or lock_val_mongodb is None:
                        time.sleep(1)
                        continue
                    datapool.mongodb.insert(task, datapool.collection)
                    release_lock(rcon, MONGODB_LOCK, lock_val_mongodb)
                    break
                continue
            else:
                task["status"] = "waiting_to_process"
                # 如果状态不是handle_done，则插入redis队列，开始自动运行
                while True:
                    lock_val = acquire_lock_with_timeout(rcon, AUTO_RUN_LOCK)
                    if not lock_val or lock_val is None:
                        time.sleep(1)
                        continue
                    push_msg(rcon, AUTO_RUN, task)
                    # datapool.mongodb.insert(task, datapool.collection)
                    release_lock(rcon, AUTO_RUN_LOCK, lock_val)
                    task_count += 1
                    logger.info(f"Success, send new task, task_id: {task_id}, "
                                f"task_count: {task_count}")
                    break
                # 更新mongodb，更新当前的数据的执行状态
                while True:
                    lock_val_mongodb = acquire_lock_with_timeout(rcon, MONGODB_LOCK)
                    if not lock_val_mongodb or lock_val_mongodb is None:
                        time.sleep(1)
                        continue
                    datapool.mongodb.insert(task, datapool.collection)
                    release_lock(rcon, MONGODB_LOCK, lock_val_mongodb)
                    break
    else:
        logger.info("End, no new task_id in queue")
        return
    logger.info("End, finish send msg")

def update_root_logger(logger):
    """更新root logger的日志文件
    """
    log_file = os.path.join(LOG_PATH, "task_publish_master.log")
    update_logger_rotate_file_handler(logger, log_file)
    

if __name__ == "__main__":

    datapool = DataPool()
    rcon = get_redis_rcon()
    task_count = 0
    logger =  init_root_logger()
    update_root_logger(logger)
    
    while True:
        datapool.update_task_queue()
        call_back(rcon, datapool, task_count)
        # 休眠5分钟，继续获取任务
        logger.info("Sleep 5 minutes, refresh task queue again")
        logger.info("Waiting for new task...")
        logger.info("---------------------------------------------------------------") 
        time.sleep(60*10)  # 休眠10分钟
        


