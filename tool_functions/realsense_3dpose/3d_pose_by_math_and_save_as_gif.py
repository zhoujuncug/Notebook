##############################
# 1. 2 camera calibration
# 2. 3d pose from two 2d pose
# 3. draw dynamic figure and save as *.gif by using matplotlib
##############################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.linalg import solve

file_name = 'poses_whole_body.npy'
poses = np.load('lab/'+file_name, allow_pickle=True)
left_poses = poses[0]
right_poses = poses[1]

kps_num = left_poses[0][0]['keypoints'].shape[0]
if kps_num == 133:
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [16, 18],
                [16, 19], [16, 20], [17, 21], [17, 22], [17, 23], [92, 93],
                [93, 94], [94, 95], [95, 96], [92, 97], [97, 98], [98, 99],
                [99, 100], [92, 101], [101, 102], [102, 103], [103, 104],
                [92, 105], [105, 106], [106, 107], [107, 108], [92, 109],
                [109, 110], [110, 111], [111, 112], [113, 114], [114, 115],
                [115, 116], [116, 117], [113, 118], [118, 119], [119, 120],
                [120, 121], [113, 122], [122, 123], [123, 124], [124, 125],
                [113, 126], [126, 127], [127, 128], [128, 129], [113, 130],
                [130, 131], [131, 132], [132, 133]]
else:
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
skeleton = np.array(skeleton) - 1

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

pose_3d_all = []
for left_pose, right_pose in zip(left_poses, right_poses):
    ######################
    # map left and right pose to camera coordinate
    ######################
    left_pose = left_pose[0]['keypoints']
    left_pose_val = np.hstack([left_pose[:, :2], np.ones([kps_num, 1])])
    IR1_film = np.dot(np.linalg.inv(IR1_matrix1), left_pose_val.T)
    IR1_film = IR1_film.T
    IR1_film[:, 2] = 635.453

    right_pose = right_pose[0]['keypoints']
    right_pose_val = np.hstack([right_pose[:, :2], np.ones([kps_num, 1])])
    IR2_film = np.dot(np.linalg.inv(IR2_matrix1), right_pose_val.T)
    IR2_film = IR2_film.T
    IR2_film[:, 2] = 635.453
    IR2_IR1_film = np.hstack([IR2_film, np.ones([kps_num, 1])])
    IR2_IR1_film = np.dot(IR2_extrin_matrix_2_IR1, IR2_IR1_film.T).T[:, :3]

    ######################
    #  calculate 3d pose
    ######################
    valid_joint_thresh = 0.3
    valid_mask = (left_pose[:, 2] > valid_joint_thresh) * (right_pose[:, 2] > valid_joint_thresh)

    pose_3d = np.zeros((kps_num, 3))
    for i, _ in enumerate(valid_mask):
        if not _ :
            continue

        # solving function AX = B
        F1, F2 = IR1_film[i], IR2_IR1_film[i]
        A = np.mat([
            [1,             0,     0,     0,              0,      0,      -F1[0], 0               ],
            [0,             1,     0,     0,              0,      0,      -F1[1], 0               ],
            [0,             0,     1,     0,              0,      0,      -F1[2], 0               ],
            [0,             0,     0,     1,              0,      0,      0,      -F2[0] + 50.0627],
            [0,             0,     0,     0,              1,      0,      0,      -F2[1]          ],
            [0,             0,     0,     0,              0,      1,      0,      -F2[2]          ],
            [F1[0],         F1[1], F1[2], -F1[0],         -F1[1], -F1[2], 0,      0               ],
            [F2[0]-50.0627, F2[1], F2[2], -F2[0]+50.0627, -F2[1], -F2[2], 0,      0               ],
        ])
        B = np.mat(
            [0, 0, 0, 50.0627, 0, 0, 0, 0]
        ).T


        x = solve(A, B)

        pose_3d[i, 0] = (x[0] + x[3]) / 2
        pose_3d[i, 1] = (x[1] + x[4]) / 2
        pose_3d[i, 2] = (x[2] + x[5]) / 2

    pose_3d_all.append(pose_3d)

#########################################
# draw dynamic 3d pose using matplotlib and save as *.gif
#########################################
def add_line(pose_3d, ax, skeleton):
    skeleton = np.array(skeleton) - 1
    for i in skeleton:
        if pose_3d[i[0]][2] != 0 and pose_3d[i[1]][2]:
            x = np.array([pose_3d[i[0]][0], pose_3d[i[1]][0]])
            y = np.array([pose_3d[i[0]][1], pose_3d[i[1]][1]])
            z = np.array([pose_3d[i[0]][2], pose_3d[i[1]][2]])
            ax.plot3D(x, y, z, 'red')


def add_point(pose_3d, ax):
    x, y, z = pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2]
    ax.scatter3D(x, y, z, cmap='Greens')



fig = plt.figure()
ax1 = Axes3D(fig)


x_track = np.zeros((1, 3))
x_track_s = np.array([.0,.0,.0])
theta = 0
def gen_skeleton():
    global x_track_s,x_track,theta
    theta += 10*np.pi/180
    x = 6*np.sin(theta)
    y = 6*np.cos(theta)
    x_track_s +=[x,y,0.1]
    x_track = np.append(x_track, [x_track_s],axis=0)
    return x_track

def update(i):
    plt.cla()
    square_x = np.array([-1700, 700])
    square_y = np.array([-700, 1700])
    square_z = np.array([-200, 2200])
    ax1.scatter3D(square_x, square_y, square_z, color='whitesmoke')

    label = 'timestep {0}'.format(i)
    print(label)
    pose_3d = pose_3d_all[i]
    for i in skeleton:
        if pose_3d[i[0]][2] != 0 and pose_3d[i[1]][2]:
            x = np.array([pose_3d[i[0]][0], pose_3d[i[1]][0]])
            y = np.array([pose_3d[i[0]][1], pose_3d[i[1]][1]])
            z = np.array([pose_3d[i[0]][2], pose_3d[i[1]][2]])
            ax1.plot3D(x, y, z, 'red')

    if kps_num != 133:
        x, y, z = pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2]
        ax1.scatter3D(x, y, z, cmap='Greens')

    ax1.view_init(elev=-100, azim=-70)

    return ax1

if __name__ == '__main__':
    # FuncAnimation 会在每一帧都调用“update” 函数。
    # 在这里设置一个10帧的动画，每帧之间间隔200毫秒
    anim = FuncAnimation(fig, update, frames=len(pose_3d_all), interval=200)
    anim.save('line1.gif', dpi=80, writer='imagemagick')
