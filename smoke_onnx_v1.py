import cv2
import time
import os
import numpy as np
import argparse
import onnxruntime as ort
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, cur_path)
from python.dd_utils import check_img_size, non_max_suppression, get_color, scale_coords
# from pycocotools.coco import COCO
import torch
import os
from tqdm import tqdm
import json
stride_max = 32

class params():
    def __init__(self,):
        self.onnx2d_path = 'train_normal_yolov5s_base_barrier_wc_0228_0602_filter10_960.onnx'
        self.onnx2d_path = os.path.join(cur_path, self.onnx2d_path)
        self.classfile = 'iflytek.names'
        self.classfile = os.path.join(cur_path, self.classfile)
        self.confThreshold = 0.5
        self.nmsThreshold = 0.45


class yolov5_iflytek():
    def __init__(self, confThreshold=None, providers=['CUDAExecutionProvider']):
        config = self.get_config()
        if confThreshold is not None:
            self.confThreshold = confThreshold
        model_pb_path = self.onnx2d_path
        label_path = self.classfile
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so, providers=providers)
        self.classes = list(map(lambda x: x.strip(), open(label_path, 'r').readlines()))
        self.num_classes = len(self.classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = self.num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.input_shape = (640, 640)


    

    def get_config(self,):
        self.onnx2d_path = '../models/best_dynamic_new.onnx'
        self.onnx2d_path = os.path.join(cur_path, self.onnx2d_path)
        self.classfile = '../models/iflytek.names'
        self.classfile = os.path.join(cur_path, self.classfile)
        self.confThreshold = 0.5
        self.nmsThreshold = 0.45



    def cal_pad(self,new_shape,shape):
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r 
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        return dw,dh,ratio,new_unpad
    


    def letterbox_new(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
       
        shape = img.shape[:2]  
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        img = cv2.copyMakeBorder(img, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def read_img_numpy(self, img_paths):
        images = []
        new_shape = self.input_shape[0]
        new_shape = (new_shape,new_shape)
        ori_shape = img_paths[0].shape[:2]
        dw,dh,ratio,new_unpad = self.cal_pad(new_shape,ori_shape)
        for i in range(len(img_paths)):
            img = img_paths[i]
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            img = cv2.copyMakeBorder(img, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=(114, 114, 114)) 
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = img.astype(np.float16)  
            img = img[np.newaxis,...]
            images.append(img)
        batch_image = np.concatenate(images, axis=0)
        batch_image = torch.tensor(batch_image, device="cuda:0")  
        batch_image = batch_image/255.0
        img = batch_image.cpu().numpy()
        return img, ratio


    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def detect(self, img_path):
        dets = []

        img_rsz, ratio = self.read_img_numpy(img_path) 
      
        # t1 = time.time()
        dets_list = []

        outs_all = self.net.run(None, {self.net.get_inputs()[0].name: img_rsz})[0]
        # t2 = time.time()
        # print("use_time:",t2-t1)
        t1 = time.time()
        for i in range(outs_all.shape[0]):
            outs = outs_all[i,:,:]
            row_ind = 0
            for i in range(self.nl):
                h, w = int(self.input_shape[0] / self.stride[i]), int(self.input_shape[1] / self.stride[i])
                length = int(self.na * h * w)
                if self.grid[i].shape[2:4] != (h, w):
                    self.grid[i] = self._make_grid(w, h)
                outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                    self.grid[i], (self.na, 1))) * int(self.stride[i])
                outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                    self.anchor_grid[i], h * w, axis=0)
                
                outs[row_ind:row_ind + length, 0] =  outs[row_ind:row_ind + length, 0] / ratio[0]
                outs[row_ind:row_ind + length, 2] =  outs[row_ind:row_ind + length, 2] / ratio[0]
                outs[row_ind:row_ind + length, 1] =  outs[row_ind:row_ind + length, 1] / ratio[1]
                outs[row_ind:row_ind + length, 3] =  outs[row_ind:row_ind + length, 3] / ratio[1]
                row_ind += length
            outs = np.expand_dims(outs, axis=0)
            outs = torch.tensor(outs)  ##### cuda
            outs = outs.to(torch.float32)
            outs = non_max_suppression(outs, self.confThreshold, self.nmsThreshold)
        

            pred_np = outs[0].numpy()
            for i, res in enumerate(pred_np):
                x1, x2 = pred_np[i][0], pred_np[i][2]
                y1, y2 = pred_np[i][1], pred_np[i][3]
                prob = pred_np[i][4]
                cls_id = pred_np[i][5]
                intbox = [x1, y1, x2, y2, prob, cls_id]
                intbox = [float(pos) for pos in intbox]
                dets.append(intbox)
            dets_list.append(dets)
        return dets_list


