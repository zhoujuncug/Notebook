#######################
# draw 3d pose using open3d
#######################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
import time
import open3d as o3d
####################
# load pose data and in_ex matrix
####################
pose_IR1 = np.load('lab/left_pose_data.npy', allow_pickle=True)[0]['keypoints']
pose_IR2 = np.load('lab/right_pose_data.npy', allow_pickle=True)[0]['keypoints']
pose_3d = np.load('lab/pose_3d.npy', allow_pickle=True)
pose_IR1[:, 2] = 1
pose_IR2[:, 2] = 1

IR1_matrix1 = np.identity(3)
IR1_matrix1[0,2] = 640.058
IR1_matrix1[1,2] = 401.652
IR1_matrix2 = np.array([
    [635.453, 0, 0],
    [0, 635.453, 0],
    [0, 0, 1]
]) * (1./635.453)

IR2_matrix1 = np.identity(3)
IR2_matrix1[0,2] = 640.058
IR2_matrix1[1,2] = 401.652
IR2_matrix2 = np.array([
    [635.453, 0, 0],
    [0, 635.453, 0],
    [0, 0, 1]
]) * (1./635.453)

IR2_extrin_matrix_2_IR1 = np.array([
    [1, 0, 0, 50.0627],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

####################
# transform to corresponding film coordinate
####################
IR1_film = np.dot(np.linalg.inv(IR1_matrix1), pose_IR1.T)
IR1_film = IR1_film.T
IR1_film[:, 2] = 635.453

IR2_film = np.dot(np.linalg.inv(IR2_matrix1), pose_IR2.T)
IR2_film = IR2_film.T
IR2_film[:, 2] = 635.453
IR2_film = np.hstack([IR2_film, np.ones([17, 1])])
IR2_film = np.dot(IR2_extrin_matrix_2_IR1, IR2_film.T).T[:, :3]
#####################
# show using open3d
#####################

# o
x = np.zeros([250000, 3])
y = np.zeros([250000, 3])
z = np.zeros([250000, 3])
_ = np.linspace(0, 2500, 250000)
x[:, 0] = _
y[:, 1] = _
z[:, 2] = _
xyz = np.vstack([x, y, z])
pcdxyz = o3d.geometry.PointCloud()
pcdxyz.points = o3d.utility.Vector3dVector(xyz)

xyz_2 = np.hstack([xyz, np.ones([750000, 1])])
xyz_2 = np.dot(IR2_extrin_matrix_2_IR1, xyz_2.T).T[:, :3]
pcdxyz_2 = o3d.geometry.PointCloud()
pcdxyz_2.points = o3d.utility.Vector3dVector(xyz_2[:-50000, :])



# IR1 film
pcd_IR1 = o3d.geometry.PointCloud()
pcd_IR1.points = o3d.utility.Vector3dVector(IR1_film)
colors = np.ones_like(IR1_film)
colors[:, ] = [1, 0, 0]
pcd_IR1.colors = o3d.utility.Vector3dVector(colors)

# IR2 film
pcd_IR2 = o3d.geometry.PointCloud()
pcd_IR2.points = o3d.utility.Vector3dVector(IR2_film)
colors = np.ones_like(IR2_film)
colors[:, ] = [0, 1, 0]
pcd_IR2.colors = o3d.utility.Vector3dVector(colors)

# pose 3d
pcd_pose_3d = o3d.geometry.PointCloud()
pcd_pose_3d.points = o3d.utility.Vector3dVector(pose_3d)
colors = np.ones_like(pose_3d)

colors[:, ] = [0, 0, 1]
pcd_pose_3d.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcdxyz, pcd_IR1, pcd_IR2, pcd_pose_3d, pcdxyz_2], mesh_show_wireframe=True, mesh_show_back_face=True)
