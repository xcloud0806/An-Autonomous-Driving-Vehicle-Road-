source /home/brli/conda.sh pcl

CAR=sihao_1482
DATES=(
    20240417_n
)

for DATE in ${DATES[@]}
do
# printf ">>>> 1/6 Create Run Config\n"
CONFIG=/data_autodrive/users/brli/dev_raw_data/sihao_1482/work_tmp/run_${CAR}_${DATE}.json
echo $CONFIG
# python3 node_create_config_cli.py -i /data_cold2/ripples/sihao_1482/common_frame/$DATE \
#                                 -c $CAR \
#                                 -t ./templates/run_common_collect_config.json \
#                                 --calib_date 20240228  -o $CONFIG
        
# printf  ">>>> 2/6 Match Frame\n"
# python3 node_match_frame.py $CONFIG

printf  ">>>> 3/6 Generate Segments\n"
python3 node_gen_night_segments.py $CONFIG /data_cold2/ripples/sihao_1482/common_seg/20240417_d

# printf  ">>>> 4/6 Calc Odometry\n"
# python3 lidar_odometry.py $CONFIG

# printf   ">>>> 5/6 Reconstruct Point Cloud\n"
# python3 node_reconstruct_v1.py $CONFIG

# printf  ">>>> 6/6 Prepare Annotation Data\n"
# python3 node_prepare_anno_data.py $CONFIG

# printf "<<<< %s Processing Completed\n" $DATE

done