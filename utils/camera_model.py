import cv2
import json
import copy
import numpy as np


class PinholeCamera:
    def __init__(self):
        self.img_size = None
        self.intrinsic = None
        self.ud_intrinsic = None
        self.dist = None

    def copy(self):
        new_camera = PinholeCamera()
        if self.img_size is not None:
            new_camera.img_size = self.img_size
        if self.intrinsic is not None:
            new_camera.intrinsic = self.intrinsic.copy()
        if self.ud_intrinsic is not None:
            new_camera.ud_intrinsic = self.ud_intrinsic.copy()
        if self.dist is not None:
            new_camera.dist = self.dist.copy()
        return new_camera

    def set_camera_params(self, resolution, intrinsic, ud_intrinsic, dist):
        assert len(resolution) == 2 and isinstance(resolution[0], int) and isinstance(resolution[1], int) and resolution[0] > 0 and resolution[1] > 0
        assert intrinsic.shape == (3, 3)
        assert ud_intrinsic.shape == (3, 3)
        assert dist.shape[0] == 1 and dist.shape[1] >= 4

        self.img_size = tuple(resolution)
        self.intrinsic = intrinsic.copy()
        self.ud_intrinsic = ud_intrinsic.copy()
        self.dist = dist.copy()

    def load_from_params(self, pinhole_params):
        resolution = tuple(pinhole_params["image_size"])
        intrinsic = np.array(pinhole_params["mtx"], dtype=np.float64)
        ud_intrinsic = np.array(pinhole_params["undistored_mtx"], dtype=np.float64)
        dist = np.array(pinhole_params["dist"], dtype=np.float64)
        self.set_camera_params(resolution, intrinsic, ud_intrinsic, dist)
        
    def set_image_size(self, img_size):
        w_scale = img_size[0] / self.img_size[0]
        h_scale = img_size[1] / self.img_size[1]
        self.img_size = img_size
        self.intrinsic[0] = self.intrinsic[0] * w_scale
        self.intrinsic[1] = self.intrinsic[1] * h_scale
        self.ud_intrinsic[0] = self.ud_intrinsic[0] * w_scale
        self.ud_intrinsic[1] = self.ud_intrinsic[1] * h_scale

    def project_points(self, camera_points):
        fx = self.intrinsic[0, 0]
        fy = self.intrinsic[1, 1]
        cx = self.intrinsic[0, 2]
        cy = self.intrinsic[1, 2]

        k1 = self.dist[0, 0]
        k2 = self.dist[0, 1]
        p1 = self.dist[0, 2]
        p2 = self.dist[0, 3]
        k3 = self.dist[0, 4] if self.dist.shape[1] > 4 else 0
        k4 = self.dist[0, 5] if self.dist.shape[1] > 5 else 0
        k5 = self.dist[0, 6] if self.dist.shape[1] > 6 else 0
        k6 = self.dist[0, 7] if self.dist.shape[1] > 7 else 0
        s1 = self.dist[0, 8] if self.dist.shape[1] > 8 else 0
        s2 = self.dist[0, 9] if self.dist.shape[1] > 9 else 0
        s3 = self.dist[0, 10] if self.dist.shape[1] > 10 else 0
        s4 = self.dist[0, 11] if self.dist.shape[1] > 11 else 0

        x_h = camera_points[:, 0] / np.clip(camera_points[:, 2], 1e-6, np.inf)
        y_h = camera_points[:, 1] / np.clip(camera_points[:, 2], 1e-6, np.inf)

        x_2 = np.power(x_h, 2)
        y_2 = np.power(y_h, 2)

        r_2 = x_2 + y_2
        r_4 = np.power(r_2, 2)
        r_6 = np.power(r_2, 3)

        _2xy = 2 * x_h * y_h

        kr = (1 + k1 * r_2 + k2 * r_4 + k3 * r_6) / (1 + k4 * r_2 + k5 * r_4 + k6 * r_6)
        u = fx * (x_h * kr + _2xy * p1 + p2 * (r_2 + 2*x_2) + s1 * r_2 + s2 * r_4) + cx
        v = fy * (y_h * kr + p1 * (r_2 + 2 * y_2) + _2xy * p2 + s3 * r_2 + s4 * r_4) + cy

        image_points = np.stack([u, v], axis=1)

        valid_mask = np.all(np.stack([
            camera_points[:, 2] >= 1e-6,
            u >= 0,
            v >= 0,
            u <= self.img_size[0]-1,
            v <= self.img_size[1]-1
        ], axis=0), axis=0)
        image_points[np.logical_not(valid_mask)] = -100
        return image_points, valid_mask


class FisheyeCamera:
    def __init__(self):
        self.img_size = None
        self.intrinsic = None
        self.ud_intrinsic = None
        self.dist = None

    def copy(self):
        new_camera = FisheyeCamera()
        if self.img_size is not None:
            new_camera.img_size = self.img_size
        if self.intrinsic is not None:
            new_camera.intrinsic = self.intrinsic.copy()
        if self.ud_intrinsic is not None:
            new_camera.ud_intrinsic = self.ud_intrinsic.copy()
        if self.dist is not None:
            new_camera.dist = self.dist.copy()
        return new_camera
    
    def set_camera_params(self, resolution, intrinsic, ud_intrinsic, dist):
        assert len(resolution) == 2 and isinstance(resolution[0], int) and isinstance(resolution[1], int) and resolution[0] > 0 and resolution[1] > 0
        assert intrinsic.shape == (3, 3)
        assert ud_intrinsic.shape == (3, 3)
        assert dist.shape[0] == 4 and dist.shape[1] == 1

        self.img_size = tuple(resolution)
        self.intrinsic = intrinsic.copy()
        self.ud_intrinsic = ud_intrinsic.copy()
        self.dist = dist.copy()
        
    def load_from_params(self, pinhole_params):
        resolution = tuple(pinhole_params["image_size"])
        intrinsic = np.array(pinhole_params["mtx"], dtype=np.float64)
        ud_intrinsic = np.array(pinhole_params["undistored_mtx"], dtype=np.float64)
        dist = np.array(pinhole_params["dist"], dtype=np.float64)
        self.set_camera_params(resolution, intrinsic, ud_intrinsic, dist)
        
    def set_image_size(self, img_size):
        w_scale = img_size[0] / self.img_size[0]
        h_scale = img_size[1] / self.img_size[1]
        self.img_size = img_size
        self.intrinsic[0] = self.intrinsic[0] * w_scale
        self.intrinsic[1] = self.intrinsic[1] * h_scale
        self.ud_intrinsic[0] = self.ud_intrinsic[0] * w_scale
        self.ud_intrinsic[1] = self.ud_intrinsic[1] * h_scale

    def project_points(self, camera_points):
        fx = self.intrinsic[0, 0]
        fy = self.intrinsic[1, 1]
        cx = self.intrinsic[0, 2]
        cy = self.intrinsic[1, 2]

        k1 = self.dist[0, 0]
        k2 = self.dist[1, 0]
        k3 = self.dist[2, 0]
        k4 = self.dist[3, 0]

        x = camera_points[:, 0]
        y = camera_points[:, 1]
        z = camera_points[:, 2]

        r = np.sqrt(np.power(x, 2) + np.power(y, 2))
        theta = np.arctan2(r, z)
        theta_d = theta * (
            1 + k1 * np.power(theta, 2) + \
                k2 * np.power(theta, 4) + \
                    k3 * np.power(theta, 6) + \
                        k4 * np.power(theta, 8)
        )
        kr = theta_d / np.clip(r, 1e-6, np.inf)
        u = fx * (kr * x) + cx
        v = fy * (kr * y) + cy

        image_points = np.stack([u, v], axis=1)
        valid_mask = np.all(np.stack([
            u >= 0,
            v >= 0,
            u <= self.img_size[0]-1,
            v <= self.img_size[1]-1
        ], axis=0), axis=0)
        image_points[np.logical_not(valid_mask)] = -100
        return image_points, valid_mask
