import os
import sys
import glob
import wave
from numpy import  arange,short
import scipy.signal as signal
import imageio 
# require imageio imageio-ffmpeg
from multiprocessing import pool

def generate_wav(Time,save_path):
    frameRate = 16000
    time = Time  # unit = second
    volumn =  0 # 30000

    # 通过调用 scipy.signal库中的 chrip 函数，产生长度为10秒、取样频率为44.1kHz、100Hz到1kHz的频率扫描波
    t = arange(0, time, 1.0 / frameRate)
    wave_data = signal.chirp(t, 100, time, 100, method='linear') * volumn
    # 由于chrip函数返回的数组为float64型，需要调用数组的astype方法将其转换为short型。
    wave_data = wave_data.astype(short)

    maxVal = 0
    minVal = 32768

    print('[min,max] value=[' + str(minVal) + ',' + str(maxVal) + ']' )
    f = wave.open(save_path, "wb")
    # 配置声道数、量化位数和取样频率
    f.setnchannels(1)  # 声道数
    f.setsampwidth(2)  # 量化位数
    f.setframerate(frameRate)  # 采样频率
    f.writeframes(wave_data.tostring())  # 将wav_data转换为二进制数据写入文件
    f.close()

def handle_clip(idx, date, clip, clip_path, root_path):    
    if not os.path.exists(clip_path):
        return 
    # file_list = os.path.join(root_path, f"{clip}_img_list.txt")
    print(f">>>Hanlde {idx} <-> {clip}")
    video = os.path.join(root_path, "video_audio", date, f"{clip}.mp4")
    images = os.listdir(clip_path)
    images.sort()
    with imageio.get_writer(video, 'ffmpeg', ffmpeg_params=['-c:v', "h264_nvenc", "-vf", "scale=640:480"], fps=10) as v_writer:
        for i, img in enumerate(images):
            if i % 10 == 0:
                img_path = os.path.join(clip_path, img)
                jpg = imageio.imread(img_path)
                v_writer.append_data(jpg)
    seconds = (len(images) / 10/ 10) + 1
    audio = os.path.join(root_path, "video_audio", date, f"{clip}.wav")
    generate_wav(seconds, audio)    

if __name__ == '__main__':
    root_path = sys.argv[1]
    date = sys.argv[2]
    date_path = os.path.join(root_path, date)
    
    clips = os.listdir(date_path)
    print(f"Total {len(clips)} in {date_path}\n\n\n")
    os.makedirs(os.path.join(root_path, "video_audio", date), exist_ok=True)
    pool = pool.Pool(4)
    for i, clip in enumerate(clips):
        clip_path = os.path.join(date_path, clip, "ofilm_surround_front_120_8M")
        # handle_clip(i, date, clip, clip_path, root_path)
        pool.apply_async(handle_clip, args=(i, date, clip, clip_path, root_path))
    pool.close()
    pool.join()

    print(f"{root_path} Done.")
        

