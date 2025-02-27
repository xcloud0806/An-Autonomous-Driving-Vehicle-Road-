from multiprocessing import Pool, get_logger
import sys
import logging

# 设置日志记录器
logger = get_logger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 全局错误标志
error_occurred = False

def error_callback(error):
    """Error callback function to handle exceptions in the worker process."""
    global error_occurred
    logger.error("子进程发生错误: %s", error)
    error_occurred = True  # 标记有错误发生

def gen_seg_lmdb(meta, segid):
    try:
        # 模拟可能发生异常的操作
        with open(f"path/to/segment/{segid}", 'r') as f:
            content = f.read()
        # 其他处理...
    except Exception as e:
        # 如果发生异常，我们将它抛出去，这样 error_callback 就会被调用
        raise RuntimeError(f"处理段 {segid} 时失败: {e}")

if __name__ == '__main__':
    meta = {}  # 示例数据
    seg_names = ['seg1', 'seg2', 'seg3']  # 示例分段名称列表

    pool = Pool(processes=4)
    results = []

    # 提交所有任务
    for segid in seg_names:
        result = pool.apply_async(
            gen_seg_lmdb,
            args=(meta, segid),
            error_callback=error_callback
        )
        results.append(result)

    pool.close()

    try:
        # 等待所有任务完成，并获取它们的结果
        for result in results:
            result.get()  # 这里会等待异步调用完成，并重新抛出子进程中的任何异常
    except Exception as e:
        print(f"主程序检测到子进程错误: {e}")
        error_occurred = True  # 确保错误标志被设置

    if error_occurred:
        pool.terminate()  # 强制终止池中的所有工作进程
        pool.join()
        sys.exit(1)  # 以非零状态码退出，表示程序执行失败

    pool.join()