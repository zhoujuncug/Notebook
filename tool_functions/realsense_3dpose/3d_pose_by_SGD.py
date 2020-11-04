#####################
# In this file, the 3D pose coordinate is calculated from IR1 and IR2 image
# by SGD optimizer.
#####################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
import time
####################
# load pose data and in_ex matrix
####################
pose_IR1 = np.load('lab/left_pose_data.npy', allow_pickle=True)[0]['keypoints']
pose_IR2 = np.load('lab/right_pose_data.npy', allow_pickle=True)[0]['keypoints']
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
####################
# optimization using pytorch
####################
device = 'cpu'
IR1_film = torch.from_numpy(IR1_film.T).to(torch.float).to(device)
IR2_film = torch.from_numpy(IR2_film.T).to(torch.float).to(device)
IR2_extrin_matrix_2_IR1 = torch.from_numpy(IR2_extrin_matrix_2_IR1).to(torch.float).to(device)

t1 = nn.Parameter(3.75*torch.ones([17, 2], dtype=torch.float, requires_grad=True).to(device))
# optimizer = torch.optim.SGD([t1], lr=0.0001, momentum=0.97)
optimizer = torch.optim.Adam([t1], lr=0.01, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000], gamma=0.1)
t_start = time.time()
for i in tqdm.tqdm(range(50000)):
    vector_IR1 = IR1_film * t1[:, 0]
    vector_IR2 = IR2_film * t1[:, 1]
    vector_IR2 = torch.cat((vector_IR2, torch.ones(1, 17).to(device)), 0)
    vector_IR2 = torch.mm(IR2_extrin_matrix_2_IR1, vector_IR2)
    vector_IR2 = vector_IR2[:3, :]

    loss = F.mse_loss(vector_IR1, vector_IR2[:3, :])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # lr_scheduler.step()
t_end = time.time()
t_duration = t_end - t_start
pose_3D = (vector_IR1.T + vector_IR2.T)/2
pose_3D = pose_3D.detach().to('cpu').numpy()
print(t1)
print(pose_3D)
print(t_duration)
# np.save('lab/pose_3d.npy', pose_3D)


