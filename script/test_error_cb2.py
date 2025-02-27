from multiprocessing import Pool, Event  
import sys  
import cv2
import time
def get_meta(meta):  
    # if not meta:  # 假设 meta 为空时抛出异常  
    #     raise ValueError("Meta data is missing!")
    try:
        img = cv2.imread("/data/3.jpg")
        if img is None:
            raise ValueError("Image is None!")
    except Exception as e:
        # print(f"Processing {e}")
        raise ValueError(e)
    # print(f"Processing {meta}")
    time.sleep(30)

def gen_seg_lmdb(meta, enable_cams, enable_bpearls, submit_data_path):  
    # 示例函数，模拟可能抛出异常的情况  
    # if not meta:  # 假设 meta 为空时抛出异常  
    #     raise ValueError("Meta data is missing!")  
    # # 其他逻辑  
    # print(f"Processing {meta} with {enable_cams}, {enable_bpearls}, {submit_data_path}")  
    get_meta(meta)
    time.sleep(300)
def error_callback(e, error_event):  
    # 捕获子进程中的异常并设置事件  
    print(f"Error occurred: {e}", file=sys.stderr)  
    error_event.set()  # 通知主进程有异常发生
    if error_event.is_set():
        print(f"error_event.is_set() done")
    sys.exit(1)  # 退出主程序

if __name__ == "__main__":  
    seg_names = ["seg1", "seg2", "seg3"]  # 示例数据
    meta = None  # 示例参数，故意设置为 None 以触发异常
    enable_cams = True
    enable_bpearls = False
    submit_data_path = "/path/to/data"

    # 创建一个事件对象，用于通知主进程  
    error_event = Event()  

    pool = Pool(processes=4)  
    try:  
        for segid in seg_names:

            # if not enable_bpearls:  # 示例：如果 segid 为空，抛出异常  
            #     raise ValueError(f"Invalid segment ID: {segid}") 

            pool.apply_async(  
                gen_seg_lmdb,  
                args=(meta, enable_cams, enable_bpearls, submit_data_path),  
                error_callback=lambda e: error_callback(e, error_event)  # 捕获异常并设置事件  
            )  
            print(f"segid = {segid}")

        pool.close()  # 关闭进程池，防止提交新任务
        print(f"pool.close()")
        pool.join()  # 等待子进程终止  
        # 等待所有任务完成或检测到异常  
        # while True:  
        #     print(f"while loop")
        #     if error_event.is_set():  # 如果检测到异常  
        #         print("Terminating all processes due to an error.", file=sys.stderr)  
        #         pool.terminate()  # 终止所有子进程  
        #         # pool.join()  # 等待子进程终止  
        #         sys.exit(1)  # 退出主程序 
        #     else:
        #         print("error_event not set") 
        #     # pool.close()  # 关闭进程池，防止提交新任务  
        #     pool.join()  # 等待所有子进程完成  
        #     break  # 如果没有异常，跳出循环  
        
        print(f"============")

    except Exception as e:  
        # 捕获主进程中的异常  
        print(f"Error occurred in main process: {e}", file=sys.stderr)  
        pool.terminate()  # 终止所有子进程  
        pool.join()  # 等待子进程终止  
        sys.exit(1)  # 退出主程序  

    # except KeyboardInterrupt:  
    #     print("KeyboardInterrupt detected. Terminating pool.", file=sys.stderr)  
    #     pool.terminate()  
    #     pool.join()  
    #     sys.exit(1)