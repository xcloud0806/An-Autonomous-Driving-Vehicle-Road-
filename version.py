import os, shutil

MAJOR_VERSION=0
MINOR_VERSION=1
REVISION_NUMBER=3
VERSION_COMMIT=240401
VERSION = (MAJOR_VERSION, MINOR_VERSION, REVISION_NUMBER, VERSION_COMMIT)
__version__ = '.'.join(map(str, VERSION))

remote_version_file = "/data_autodrive/users/brli/gitee/cv2_zj_DataPlatform_Toolchain/VERSION"
remote_repo = "/data_autodrive/users/brli/gitee/cv2_zj_DataPlatform_Toolchain/"

def get_version():
    with open(remote_version_file, "w") as fp:
        fp.write(__version__)

def fetch_latest_tool():
    with open(remote_version_file, "r") as fp:
        new_version = fp.readline()
        major, minor, rev, commit = new_version.split(".")
        if int(commit) > VERSION_COMMIT:
            print(f"Current tool version is {__version__}, But latest version is {new_version}, \nFetch Latest Tool......")
            os.system("rm -rf ./node_* ./lidar_odometry.py ./utils/ ./script ./python version.py")

            os.system(f"cp -a {remote_repo}/node_* {remote_repo}/lidar_odometry.py {remote_repo}/utils/ {remote_repo}/script {remote_repo}/python version.py ./")            
            print("Fetch Latest Tool Success!")
        else:
            print(f"Current version {__version__} work.")

if __name__ == "__main__":
    get_version()
