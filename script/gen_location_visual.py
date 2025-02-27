import os
import argparse
import csv
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
import folium
from utils.gnss_coord_op import wgs84_gcj02
from scipy.spatial.distance import pdist, squareform

colors = ['#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347','#DC143C', 
         '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', 
         '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347',
         '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347','#DC143C', 
         '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', 
         '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
         '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
         '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347','#000000'] # 121 num

def parse_args():
    parser = argparse.ArgumentParser(description="Init work node")
    # clip decoder params
    parser.add_argument('--input', '-i', type=str, default="/data_autodrive/自动驾驶/hangbai/Traffic_Light/annotations")
    parser.add_argument('--data', '-r', type=str, default="/data_autodrive/auto/common/sihao_A2XX71")
    args = parser.parse_args()
    return args

# draw wgs84 pts in map 
def visual(locs, output_file="total.html"): 
    # start in iflytek
    map_ = folium.Map(location=[31.82877116479812, 117.14691878548942], zoom_start=12,
                      tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                      attr='default')

    for i, loc in enumerate(locs):
        color = colors[i % len(colors)]
        p_wgs84_lon = loc[0]
        p_wgs84_lat = loc[1]
        p_gc102_lon, p_gc102_lat = wgs84_gcj02(p_wgs84_lon, p_wgs84_lat)
        folium.CircleMarker(location=[p_gc102_lat, p_gc102_lon],
                            radius=4, popup='popup', color=color, fill_color=color,
                            fill=True).add_to(map_)
    map_.save(output_file)

def decode_csv(csv_file):
    ret = {}
    with open(csv_file, "r", encoding='UTF-8-sig') as fp:
        lines = fp.readlines()

        for line in lines:
            line = line.strip('\n')
            _time, _bag, _index, _,  _utc = line.split(',')
            bag_index = int(_bag)
            if bag_index in ret:
                bag_ret = ret[bag_index]
            else:
                bag_ret = {}
                ret[bag_index] = bag_ret
            frame_index = int(_index)
            bag_ret[frame_index] = _time

    return ret

def decode_gnss(gnss_file):
    ret = {}
    with open(gnss_file) as fp:
        rd = csv.reader(fp)
        header = rd.__next__()
        time_idx = 3 if "Time" in header[3] else None
        longi_idx = 5 if "Longitude" in header[5] else None
        lati_idx = 6 if "Latitude" in header[6] else None

        for msg in rd:
            if msg[time_idx] == 'na' or msg[longi_idx] == 'na' or msg[lati_idx] == 'na':
                continue
            time_key = int(float(msg[time_idx])*1000)
            longi_val = float(msg[longi_idx])
            lati_val = float(msg[lati_idx])
            ret[time_key] = [longi_val, lati_val]
        
    return ret

def call(args):
    anno_root = args.input
    data_root = args.data
    date_list = os.listdir(anno_root)
    clips = {}
    gnsss = {}
    positions = []
    for date in date_list:
        anno_path = os.path.join(anno_root, date)
        data_path = os.path.join(data_root, date)

        for anno in os.listdir(anno_path):
            anno_file = os.path.join(anno_path, anno)
            # ['20220831-11-32-43', 'B-0001-raw', 'camB', '4400', '4220-4500.json']
            clip, video, _, index, _ = anno.split('_')
            data_clip = os.path.join(data_path, clip)
            cam, bagindex, _ = video.split('-')

            if cam == 'B':
                if clip not in gnsss:
                    gnss_csv = os.path.join(data_clip, "output", "online", "sample", "gnssimu-sample-v7@0.csv")
                    if not os.path.exists(gnss_csv):
                        print("Skip clip {} as gnss file loss.".format(clip))
                        continue
                    gnss_dict = decode_gnss(gnss_csv)
                    gnsss[clip] = gnss_dict
                gnss_dict = gnsss[clip]
                tss_array = np.array(list(gnss_dict.keys()))

                if clip not in clips:
                    csv_file = os.path.join(data_clip, "input", "video", "B.csv")
                    print("Parse CSV {}".format(clip))
                    ret = decode_csv(csv_file)
                    clips[clip] = ret
                ret = clips[clip]
                def find_min(arr, val):
                    min_arr = np.abs(val-arr)
                    min_val = np.min(min_arr)
                    idx = np.where(min_arr == min_val)[0]
                    return arr[idx]
                try:
                    bag = ret[int(bagindex)]
                    ts = bag[int(index) + 1]
                    ts_int = int(float(ts)*1000)    
                    ts_key_np = find_min(tss_array, ts_int)
                    ts_key = ts_key_np.tolist()[0]
                    pos = gnss_dict[ts_key]
                    positions.append(pos)
                except:
                    print("Anno {}".format(anno))
                    pass

    pos_arr = np.array(positions, dtype=np.float64)
    print(pos_arr.shape)
    np.save("positions.npy", pos_arr)

R=6371.004
min_dist = 200
# cal distance in navi coordinate 
def call_distance(PointA, PointB):
    # if PointA == PointB:
    #     return 0
    MLatA = np.deg2rad(PointA[1])
    MLonA = np.deg2rad(PointA[0])
    MLatB = np.deg2rad(PointB[1])
    MLonB = np.deg2rad(PointB[0])
    dlon = MLonB - MLonA
    dlat = MLatB - MLatA
    a = np.sin(dlat / 2) ** 2 + np.cos(MLatA) * np.cos(MLatB) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * R * 1000

def classify(npy_file = "positions.npy"):
    locs = np.load(npy_file)
    main_locs = []

    temp_locs = locs.tolist()
    temp_locs.pop(0)
    loc = locs[0]
    target = loc.tolist()
    main_locs.append(target)
    while True:
        next_target = None
        for tmp in temp_locs:            
            dist = call_distance(target, tmp)
            #print("PA {} <-> PB {} --- DIST {}".format(target, tmp, dist))
            if dist == 0:
                continue
            if dist > min_dist:
                #print("PA {} <-> PB {} --- DIST {}".format(target, tmp, dist))
                if next_target is None:
                    print("PA {} <-> PB {} --- DIST {}".format(target, tmp, dist))
                    next_target = tmp
                    temp_locs.remove(tmp)
            else:
                temp_locs.remove(tmp)
        if next_target is None : 
            break
        target = next_target
        main_locs.append(target)
    
    print(len(main_locs))

    # show in map
    visual(main_locs)

def classify_dbscan(npy_file = "positions.npy"):
    locs = np.load(npy_file)
    pts = locs.tolist()

    distance_matrix = squareform(pdist(pts, (lambda u, v: call_distance(u, v))))

    model = DBSCAN(eps=100, min_samples=2, metric="precomputed")
    labels = model.fit_predict(distance_matrix)
    raito = len(labels[labels[:] == -1]) / len(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)

    # res = []
    # # 迭代不同的eps值
    # for eps in [80, 100, 150, 200]:
    #     # 迭代不同的min_samples值
    #     for min_samples in range(2,11):
    #         model = DBSCAN(eps = eps, min_samples = min_samples, metric="precomputed")
    #         # 模型拟合
    #         model.fit_predict(distance_matrix)
    #         # 统计各参数组合下的聚类个数（-1表示异常点）
    #         n_clusters = len([i for i in set(model.labels_) if i != -1])
    #         # 异常点的个数
    #         outliners = np.sum(np.where(model.labels_ == -1, 1,0))
    #         # 统计每个簇的样本个数
    #         # stats = pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts()
    #         # 计算聚类得分
    #         try:
    #             score = metrics.silhouette_score(locs, model.labels_)
    #         except:
    #             score = -99
    #         res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'outliners':outliners, 'score':score})
    #         print(res)

    # show in map
    map_ = folium.Map(location=[31.82877116479812, 117.14691878548942], zoom_start=12,
                      tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                      attr='default')
    
    for i, loc in enumerate(pts):        
        label = labels[i]
        if label == -1:
            continue
        color = colors[label] if label < len(colors) else colors[label % len(colors)]
        p_wgs84_lon = loc[0]
        p_wgs84_lat = loc[1]
        p_gc102_lon, p_gc102_lat = wgs84_gcj02(p_wgs84_lon, p_wgs84_lat)
        folium.CircleMarker(location=[p_gc102_lat, p_gc102_lon],
                            radius=4, popup='popup', color=color, fill_color=color,
                            fill=True).add_to(map_)
    map_.save("dbscan_result.html")

def classify_kmeans(npy_file = "positions.npy"):
    locs = np.load(npy_file)
    pts = locs.tolist()

    distance_matrix = squareform(pdist(pts, (lambda u, v: call_distance(u, v))))

    # model = DBSCAN(eps=100, min_samples=5, metric="precomputed")
    # labels = model.fit_predict(distance_matrix)
    # raito = len(labels[labels[:] == -1]) / len(labels)
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    model = KMeans(n_clusters=1000)
    model.fit(locs)
    labels = model.labels_
    centers = model.cluster_centers_

    # show in map
    map_ = folium.Map(location=[31.82877116479812, 117.14691878548942], zoom_start=12,
                      tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                      attr='default')
    
    for i, loc in enumerate(pts):        
        label = labels[i]
        if label == -1:
            continue
        color = colors[label] if label < len(colors) else colors[label % len(colors)]
        p_wgs84_lon = loc[0]
        p_wgs84_lat = loc[1]
        p_gc102_lon, p_gc102_lat = wgs84_gcj02(p_wgs84_lon, p_wgs84_lat)
        folium.CircleMarker(location=[p_gc102_lat, p_gc102_lon],
                            radius=4, popup='popup', color=color, fill_color=color,
                            fill=True).add_to(map_)
    map_.save("kmeans_result.html")

# classify()
#classify_dbscan("gnss_parse_traficlight/positions.npy")
classify_dbscan()
#classify_kmeans()
# args = parse_args()
# call(args)






