import os

# root_path = "/data_cold2/origin_data"
root_path = "/data_cold2/ripples"

cars = [
    "chery_04228",
    "sihao_19cp2",
    "sihao_2xx71",
    "chery_10034",
    "chery_53054",
    "chery_b8615",
    "aion_d77052",
    "aion_d77360",
    "chery_13484",
    "chery_06826",
    "sihao_1482",
    "sihao_27en6",
    "sihao_0fx60",
    "sihao_8j998",
    "sihao_47465",
    "chery_32694",
    "sihao_36gl1",
    "sihao_21pt6",
    "sihao_7xx65",
    "sihao_y7862",
    "sihao_47466",
    "sihao_37xu2",
    "sihao_35kw2",
    "sihao_23gc9",
    "sihao_72kx6"
]

frame_keys = [
    "common_frame",
    # "custom_frame",
    # "luce_frame"
]
total_size = 0
def handle_date(date_path, car, key, item, total_size):
    # date_path = os.path.join(frame_path, item)
    clips = os.listdir(date_path)
    for clip in clips:
        if not clip.startswith("202"):
            continue
        clip_path = os.path.join(date_path, clip)
        if not os.path.isdir(clip_path):
            continue
        print(f"handle {car}.{key}.{item}.{clip}")
        ppp = os.listdir(clip_path)
        for p in ppp:
            if p.endswith(".record"):
                size = os.path.getsize(os.path.join(clip_path, p))
                print(f"...find {p} with size [{int(size/1024/1024)} MB]")
                total_size += size
                os.remove(os.path.join(clip_path, p))

def handle_dir(root_dir, car, key, total_size):
    items = os.listdir(root_dir)
    for item in items:
        if not os.path.isdir(os.path.join(root_dir, item)):
            continue

        if item.startswith("202") :
            date_path = os.path.join(root_dir, item)
            handle_date(date_path, car, key, item, total_size)        
        else:
            item_path = os.path.join(root_dir, item)
            if os.path.isfile(item_path):
                continue
            handle_dir(item_path, car, key, total_size)

for car in cars:
    car_path = os.path.join(root_path, car)
    for key in frame_keys:
        frame_path = os.path.join(car_path, key)
        if not os.path.exists(frame_path):
            continue
        handle_dir(frame_path, car, key, total_size)

print("total size:", int(total_size/1024/1024), "MB")
