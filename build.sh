#!/bin/bash
# build all source

# build pcd iter
proj_dir=$PWD
cd source/pcd_iter_wrap
mkdir build
cd build
cmake ..
make -j4
cd $proj_dir
cp source/pcd_iter_wrap/lib/* lib/python3.8/site_packages/

# build lidar odometry
cd source/lidar_odometry
mkdir build
cd build
cmake ..
make -j8
cd $proj_dir
cp source/lidar_odometry/lib/* lib/python3.8/site_packages/