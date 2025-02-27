import os  
import tarfile  
import lmdb
import hashlib  
import time
def calculate_sha256(file_path):  
    """计算文件的 SHA-256 哈希值"""  
    hash_sha256 = hashlib.sha256()  
    with open(file_path, "rb") as f:  
        for chunk in iter(lambda: f.read(4096), b""):  
            hash_sha256.update(chunk)  
    return hash_sha256.hexdigest()  

def compress_directory_with_gzip(source_dir, output_tar_gz):  
    """  
    使用 tar.gz 压缩指定目录，并生成校验文件。  
    :param source_dir: 要压缩的目录路径  
    :param output_tar_gz: 输出的 tar.gz 文件路径  
    """  
    if not os.path.exists(source_dir):  
        print(f"错误：目录 {source_dir} 不存在！")  
        return  

    try:  
        # 创建 tar.gz 文件
        with tarfile.open(output_tar_gz, "w") as tar:  
            # sha256_data = []  
            for root, _, files in os.walk(source_dir):  
                for file in files:  
                    file_path = os.path.join(root, file)  
                    arcname = os.path.relpath(file_path, source_dir)  # 相对路径  
                    tar.add(file_path, arcname)  
                    # 计算 SHA-256 并记录  
                    # file_sha256 = calculate_sha256(file_path)  
                    # sha256_data.append(f"{arcname} {file_sha256}")
            
            # 写入校验文件  
            # checksum_file = "checksum.sha256"  
            # with open(checksum_file, "w") as f:  
                # f.write("\n".join(sha256_data))  
            # tar.add(checksum_file, checksum_file)  # 将校验文件添加到 tar.gz 包中  
            # os.remove(checksum_file)  # 删除临时校验文件  

        print(f"压缩完成！输出文件：{output_tar_gz}")  
    except Exception as e:  
        print(f"压缩过程中出现错误：{e}")  

def compress_directory_with_lmdb(source_dir, lmdb_path): 
    if not os.path.exists(source_dir):  
        print(f"错误：目录 {source_dir} 不存在！")  
        return  

    try:  
        env = lmdb.open(lmdb_path, map_size=1e12) 
        with env.begin(write=True) as txn:  
            for root, _, files in os.walk(source_dir):  
                for file in files:  
                    file_path = os.path.join(root, file)  
                    with open(file_path, 'rb') as f:  
                        data = f.read()  
                        # 使用文件相对路径作为键  
                        key = os.path.relpath(file_path, source_dir).encode()  
                        txn.put(key, data) 

        print(f"压缩完成！输出文件：{lmdb_path}")  
    except Exception as e:  
        print(f"压缩过程中出现错误：{e}")  

def extract_and_verify_tar_gz(input_tar_gz, output_dir):  
    """  
    解压 tar.gz 文件并校验文件完整性。  
    :param input_tar_gz: 输入的 tar.gz 文件路径  
    :param output_dir: 解压到的目标目录  
    """  
    if not os.path.exists(input_tar_gz):  
        print(f"错误:tar.gz 文件 {input_tar_gz} 不存在！")  
        return  

    try:  
        # 解压 tar.gz 文件  
        with tarfile.open(input_tar_gz, "r:gz") as tar:  
            tar.extractall(output_dir)  

        # 校验文件完整性  
        checksum_file = os.path.join(output_dir, "checksum.sha256")  
        if not os.path.exists(checksum_file):  
            print("错误：校验文件 checksum.sha256 不存在，无法验证完整性！")  
            return  

        with open(checksum_file, "r") as f:  
            checksum_data = f.readlines()  

        errors = []  
        for line in checksum_data:  
            file_name, original_sha256 = line.strip().split()  
            file_path = os.path.join(output_dir, file_name)  
            if not os.path.exists(file_path):  
                errors.append(f"文件缺失：{file_name}")  
                continue  

            current_sha256 = calculate_sha256(file_path)  
            if current_sha256 != original_sha256:  
                errors.append(f"文件损坏：{file_name} (SHA-256 不匹配)")

        if errors:  
            print("校验失败！以下文件存在问题：")  
            for error in errors:  
                print(error)  
        else:  
            print("校验成功！所有文件完整无误。")  
    except Exception as e:  
        print(f"解压或校验过程中出现错误：{e}")  

# 示例调用  
# input_tar_gz_file = "./output.tar.gz"  # 替换为你的 tar.gz 文件路径  
# output_directory = "./output"  # 解压目标目录  
# extract_and_verify_tar_gz(input_tar_gz_file, output_directory)

# 示例调用  
source_directory = "/data_cold3/autoparse/chery_e0y_01829/tojinfei/20241224-08-26-37/ofilm_around_front_190_3M"  # 替换为你的目录路径  
output_tar_gz_file = "/data_cold3/autoparse/chery_e0y_01829/tojinfei/20241224-08-26-37/ofilm_around_front_190_3M.tar.gz"  # 输出 tar.gz 文件路径  
output_lmdb_file = "/data_cold3/autoparse/chery_e0y_01829/tojinfei/20241224-08-26-37/ofilm_around_front_190_3M.lmdb"
t1 = time.time()
compress_directory_with_gzip(source_directory, output_tar_gz_file)
t2 = time.time()
print(f"压缩完成，耗时：{t2 - t1}秒")

compress_directory_with_lmdb(source_directory, output_lmdb_file)
t3 = time.time()
print(f"压缩完成，耗时：{t3 - t2}秒")
