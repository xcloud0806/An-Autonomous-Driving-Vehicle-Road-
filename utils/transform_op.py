import math
import numpy as np
from pyquaternion import Quaternion


def compute_norm(vec):
    norm = math.sqrt(np.sum(np.power(vec, 2)))
    return norm


def normalize_vector(vec):
    norm = compute_norm(vec)
    vec = vec/norm
    return vec


def vec_to_anti_symmetric_matrix(vec):
    x, y, z = vec.reshape(-1)
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ], dtype=np.float64)


def rmat_to_rquant(rmat):
    q = Quaternion(matrix=rmat)
    q_vec = np.array([q.x, q.y, q.z, q.w], dtype=np.float64).reshape(4, 1)
    return q_vec


def rquant_to_rmat(rquant):
    x, y, z, w = rquant.reshape(-1)
    q = Quaternion(x=x, y=y, z=z, w=w)
    return q.rotation_matrix.astype(np.float64)


def rvec_to_rmat(rvec):
    theta = compute_norm(rvec)
    if theta < 0.000001:
        return np.identity(3, dtype=np.float64)
    n = rvec/theta

    rmat = math.cos(theta) * np.identity(3, dtype=np.float64) + (1 - math.cos(theta)) * np.matmul(n, n.transpose()) + math.sin(theta) * vec_to_anti_symmetric_matrix(n)
    rmat[0, :] = normalize_vector(rmat[0, :])
    rmat[1, :] = normalize_vector(rmat[1, :])
    rmat[2, :] = normalize_vector(rmat[2, :])
    return rmat


def rmat_to_rvec(rmat):
    theta = math.acos(max(min(1, (np.trace(rmat)-1)/2), -1))
    if math.sin(theta) < 1e-6:
        return np.zeros((3, 1), dtype=np.float64)
    asm_n = (rmat - rmat.transpose()) / (2 * math.sin(theta))
    x = (asm_n[2, 1] - asm_n[1, 2])/2
    y = (asm_n[0, 2] - asm_n[2, 0])/2
    z = (asm_n[1, 0] - asm_n[0, 1])/2
    n = np.array([[x], [y], [z]], dtype=np.float64)
    n = normalize_vector(n)
    rvec = theta * n
    return rvec


def euler_to_rmat(euler, method='xyz'):
    assert method in ['xyz', "xzy", 'yxz', 'yzx', 'zxy', 'zyx']
    rmat = np.identity(3, dtype=np.float64)

    for char in method[::-1]:
        if char == "x":
            cur_rmat = rvec_to_rmat(euler[0, 0] * np.array([[1], [0], [0]]))
        elif char == "y":
            cur_rmat = rvec_to_rmat(euler[1, 0] * np.array([[0], [1], [0]]))
        elif char == "z":
            cur_rmat = rvec_to_rmat(euler[2, 0] * np.array([[0], [0], [1]]))
        else:
            raise NotImplementedError("unknown rotation type")
        rmat = np.matmul(rmat, cur_rmat)
    return rmat


def rmat_to_euler(rmat, method="xyz"):
    if method == "xyz":
        sy = (math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0]) + \
            math.sqrt(rmat[2,1] * rmat[2,1] +  rmat[2,2] * rmat[2,2]))/2
        singular = sy < 1e-6
        if not singular :
            x = math.atan2(rmat[2,1] , rmat[2,2])
            y = math.atan2(-rmat[2,0], sy)
            z = math.atan2(rmat[1,0], rmat[0,0])
        else:
            x = math.atan2(-rmat[1,2], rmat[1,1])
            y = math.atan2(-rmat[2,0], sy)
            z = 0
    elif method == "xzy":
        sz = (math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[2,0] * rmat[2,0]) + \
            math.sqrt(rmat[1,1] * rmat[1,1] +  rmat[1,2] * rmat[1,2]))/2
        singular = sz < 1e-6
        if not singular :
            x = math.atan2(-rmat[1,2], rmat[1,1])
            y = math.atan2(-rmat[2,0], rmat[0,0])
            z = math.atan2(rmat[1,0], sz)
        else:
            x = math.atan2(rmat[2,1], rmat[2, 2])
            y = 0
            z = math.atan2(rmat[1,0], sz)
    elif method == "yxz":
        sx = (math.sqrt(rmat[0,1] * rmat[0,1] +  rmat[1,1] * rmat[1,1]) + \
            math.sqrt(rmat[2,0] * rmat[2,0] +  rmat[2,2] * rmat[2,2]))/2
        singular = sx < 1e-6
        if not singular :
            x = math.atan2(rmat[2,1], sx)
            y = math.atan2(-rmat[2,0], rmat[2,2])
            z = math.atan2(-rmat[0,1], rmat[1,1])
        else:
            x = math.atan2(rmat[2,1], sx)
            y = math.atan2(rmat[0,2], rmat[0,0])
            z = 0
    elif method == "yzx":
        sz = (math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[0,2] * rmat[0,2]) + \
            math.sqrt(rmat[1,1] * rmat[1,1] +  rmat[2,1] * rmat[2,1]))/2
        singular = sz < 1e-6
        if not singular :
            x = math.atan2(rmat[2,1], rmat[1,1])
            y = math.atan2(rmat[0,2], rmat[0,0])
            z = math.atan2(-rmat[0,1], sz)
        else:
            x = 0
            y = math.atan2(-rmat[2, 0], rmat[2, 2])
            z = math.atan2(-rmat[0,1], sz)
    elif method == "zxy":
        sx = (math.sqrt(rmat[0,2] * rmat[0,2] +  rmat[2,2] * rmat[2,2]) + \
            math.sqrt(rmat[1,0] * rmat[1,0] +  rmat[1,1] * rmat[1,1]))/2
        singular = sx < 1e-6
        if not singular :
            x = math.atan2(-rmat[1,2], sx)
            y = math.atan2(rmat[0,2], rmat[2,2])
            z = math.atan2(rmat[1,0], rmat[1,1])
        else:
            x = math.atan2(-rmat[1,2], sx)
            y = 0
            z = math.atan2(-rmat[0,1], rmat[0,0])
    elif method == "zyx":
        sy = (math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[0,1] * rmat[0,1]) + \
            math.sqrt(rmat[1,2] * rmat[1,2] +  rmat[2,2] * rmat[2,2]))/2
        singular = sy < 1e-6
        if not singular :
            x = math.atan2(-rmat[1,2], rmat[2,2])
            y = math.atan2(rmat[0,2], sy)
            z = math.atan2(-rmat[0,1], rmat[0,0])
        else:
            x = 0
            y = math.atan2(rmat[0,2], sy)
            z = math.atan2(rmat[1,0], rmat[1,1])
    return np.array([[x], [y], [z]], dtype=np.float64)


def inter_pose(pose1, pose2, k):
    q1 = Quaternion(matrix=pose1[:3, :3], rtol=1e-5, atol=1e-5)
    q2 = Quaternion(matrix=pose2[:3, :3], rtol=1e-5, atol=1e-5)
    
    intered_rmat = Quaternion.slerp(q1, q2, k).rotation_matrix
    intered_tvec = (pose2[:3, 3:]-pose1[:3, 3:]) * k + pose1[:3, 3:]

    intered_pose = np.eye(4, dtype=np.float64)
    intered_pose[:3, :3] = intered_rmat
    intered_pose[:3, 3:] = intered_tvec
    return intered_pose


def cal_velocity(rela_pose, time_span):
    k = 1.0/time_span
    unit_pose = inter_pose(np.eye(4, dtype=np.float64), rela_pose, k)
    linear_velocity = unit_pose[:3, 3:].reshape(-1)
    anguler_velocity = rmat_to_rvec(unit_pose[:3, :3]).reshape(-1)
    return linear_velocity, anguler_velocity


def compute_sensor_to_proj_ground(sensor_to_ground, sensor_to_ego):
    ego_to_ground = np.matmul(sensor_to_ground, np.linalg.inv(sensor_to_ego))

    ego_origin_in_ego = np.array([0, 0, 0], dtype=np.float32).reshape(-1, 1)
    ego_origin_in_ground = np.matmul(ego_to_ground[:3, :3], ego_origin_in_ego) + ego_to_ground[:3, 3:]

    proj_ground_origin_in_ground = ego_origin_in_ground.copy()
    proj_ground_origin_in_ground[2, 0] = 0

    ego_x_axis_in_ego = np.array([1, 0, 0], dtype=np.float32).reshape(-1, 1)
    ego_x_axis_in_ground = np.matmul(ego_to_ground[:3, :3], ego_x_axis_in_ego)

    proj_ground_x_axis_in_ground = ego_x_axis_in_ground.copy().reshape(-1)
    proj_ground_x_axis_in_ground[2] = 0
    proj_ground_x_axis_in_ground = normalize_vector(proj_ground_x_axis_in_ground)

    proj_ground_z_axis_in_ground = np.array([0, 0, 1], dtype=np.float32)

    proj_ground_y_axis_in_ground = np.cross(proj_ground_z_axis_in_ground, proj_ground_x_axis_in_ground)
    proj_ground_y_axis_in_ground = normalize_vector(proj_ground_y_axis_in_ground)

    ground_to_proj_ground_rmat = np.stack([
        proj_ground_x_axis_in_ground, proj_ground_y_axis_in_ground, proj_ground_z_axis_in_ground
    ], axis=0)

    ground_to_proj_ground_tvec = np.matmul(ground_to_proj_ground_rmat, -proj_ground_origin_in_ground)

    ground_to_proj_ground = np.eye(4, dtype=np.float32)
    ground_to_proj_ground[:3, :3] = ground_to_proj_ground_rmat
    ground_to_proj_ground[:3, 3:] = ground_to_proj_ground_tvec

    sensor_to_proj_ground = np.matmul(ground_to_proj_ground, sensor_to_ground)
    return sensor_to_proj_ground
