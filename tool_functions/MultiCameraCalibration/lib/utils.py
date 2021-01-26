import numpy as np
import cv2 as cv
import torch
import os
from tqdm import tqdm


def get_ex_matrix(in_matrix, world_point, camera_point):
    dist_coeffs = np.zeros([5])  # assume camera distortion is negligible
    ret, rvec, tvec = cv.solvePnP(world_point, camera_point, in_matrix, dist_coeffs)

    rotation, _ = cv.Rodrigues(rvec)
    rotation_t = np.hstack([rotation, tvec])
    ex_marix = np.vstack([rotation_t, np.array([[0, 0, 0, 1]])])

    return ex_marix


def draw_pts(image, pts):
    if pts is not None:
        for pt in pts:
            cv.circle(image, tuple(pt.astype(np.int)), 4, (0, 0, 255), -1)

    return image


def check_ex_matrix_guess(kinect_frame, realsense1_frame, realsense2_frame,
                          realsense3_frame, kinect2chess_ex_matrix,
                          rs12chess_ex_matrix, rs22chess_ex_matrix,
                          rs32chess_ex_matrix, world_point):
    kinect_in_matrix = np.zeros((3, 4))
    rs1_in_matrix = np.zeros((3, 4))
    rs2_in_matrix = np.zeros((3, 4))
    rs3_in_matrix = np.zeros((3, 4))

    kinect_in_matrix[:, :3] = kinect_frame.in_matrix[0]
    rs1_in_matrix[:, :3] = realsense1_frame.in_matrix[0]
    rs2_in_matrix[:, :3] = realsense2_frame.in_matrix[0]
    rs3_in_matrix[:, :3] = realsense3_frame.in_matrix[0]

    kinect_chess2pix_matrix = kinect_in_matrix.dot(kinect2chess_ex_matrix)
    rs1_chess2pix_matrix = rs1_in_matrix.dot(rs12chess_ex_matrix)
    rs2_chess2pix_matrix = rs2_in_matrix.dot(rs22chess_ex_matrix)
    rs3_chess2pix_matrix = rs3_in_matrix.dot(rs32chess_ex_matrix)

    Pw = np.hstack([world_point, np.ones([35, 1])])

    kinect_Pc = np.dot(kinect_chess2pix_matrix, Pw.T)
    rs1_Pc = np.dot(rs1_chess2pix_matrix, Pw.T)
    rs2_Pc = np.dot(rs2_chess2pix_matrix, Pw.T)
    rs3_Pc = np.dot(rs3_chess2pix_matrix, Pw.T)

    kinect_Pc = np.divide(kinect_Pc, kinect_Pc[-1, :]).T[:, :2]
    rs1_Pc = np.divide(rs1_Pc, rs1_Pc[-1, :]).T[:, :2]
    rs2_Pc = np.divide(rs2_Pc, rs2_Pc[-1, :]).T[:, :2]
    rs3_Pc = np.divide(rs3_Pc, rs3_Pc[-1, :]).T[:, :2]

    kinect_frame_copy = kinect_frame.color_frame.copy()
    rs1_frame_copy = realsense1_frame.color_frame.copy()
    rs2_frame_copy = realsense2_frame.color_frame.copy()
    rs3_frame_copy = realsense3_frame.color_frame.copy()

    kinect_frame_copy = draw_pts(kinect_frame_copy, kinect_Pc)
    rs1_frame_copy = draw_pts(rs1_frame_copy, rs1_Pc)
    rs2_frame_copy = draw_pts(rs2_frame_copy, rs2_Pc)
    rs3_frame_copy = draw_pts(rs3_frame_copy, rs3_Pc)

    kinect_frame_chess = kinect_frame.show_chessboard()
    rs1_frame_chess = realsense1_frame.show_chessboard()
    rs2_frame_chess = realsense2_frame.show_chessboard()
    rs3_frame_chess = realsense3_frame.show_chessboard()

    kinect_corners_project = np.hstack([kinect_frame_chess, kinect_frame_copy])
    rs1_corners_project = np.hstack([rs1_frame_chess, rs1_frame_copy])
    rs2_corners_project = np.hstack([rs2_frame_chess, rs2_frame_copy])
    rs3_corners_project = np.hstack([rs3_frame_chess, rs3_frame_copy])

    cv.imshow('kinect_corners_project', kinect_corners_project)
    cv.imshow('rs1_corners_project', rs1_corners_project)
    cv.imshow('rs2_corners_project', rs2_corners_project)
    cv.imshow('rs3_corners_project', rs3_corners_project)

    cv.waitKey(0)


def check_cam2cam_matrix(cam2, cam12chess_ex_matrix,
                         cam12cam2_ex_matrix, cam2_in_matrix_):
    world_point = np.zeros((7 * 5, 4), np.float32)
    world_point[:, :2] = 30 * np.mgrid[:7, :5].T.reshape(-1, 2)
    world_point[:, 3] = np.ones((35))

    cam2_in_matrix = np.zeros((3, 4))
    cam2_in_matrix[:, :3] = cam2_in_matrix_

    cam1_pts = cam12chess_ex_matrix.dot(world_point.T)
    cam12cam2_pts = cam12cam2_ex_matrix.dot(cam1_pts)
    cam2_pts = cam2_in_matrix.dot(cam12cam2_pts)

    cam2_pts = np.divide(cam2_pts, cam2_pts[-1, :])
    cam2_pts = cam2_pts.T[:, :2]

    org_cam2_pts = cam2.corners[:, 0, :]

    cam12cam2_image = cam2.color_frame.copy()
    org_image = cam2.color_frame.copy()
    cam12cam2_image = draw_pts(cam12cam2_image, cam2_pts)
    org_image = draw_pts(org_image, org_cam2_pts)
    corners = cam2.corners
    corners_image = cam2.color_frame.copy()
    corners_image = draw_pts(corners_image, corners[:, 0, :])

    cam2cam_org_image = np.hstack([org_image, corners_image, cam12cam2_image])
    cv.imshow('org and cam2cam corners', cam2cam_org_image)
    cv.waitKey(0)


def get_init_guess(calib_dataset, check=True):
    base_camera = 'realsense3'

    session = calib_dataset.sessions[0]
    kinect_frame = session.kinect[0]
    realsense1_frame = session.realsense1[0]
    realsense2_frame = session.realsense2[0]
    realsense3_frame = session.realsense3[0]

    world_point = np.zeros((7 * 5, 3), np.float32)
    world_point[:, :2] = 30 * np.mgrid[:7, :5].T.reshape(-1, 2)

    kinect2chess_ex_matrix = kinect_frame.get_frame_ex_matrix()
    rs12chess_ex_matrix = realsense1_frame.get_frame_ex_matrix()
    rs22chess_ex_matrix = realsense2_frame.get_frame_ex_matrix()
    rs32chess_ex_matrix = realsense3_frame.get_frame_ex_matrix()

    if check:
        viz_3d_space(np.linalg.inv(kinect2chess_ex_matrix), np.linalg.inv(rs12chess_ex_matrix),
                     np.linalg.inv(rs22chess_ex_matrix), np.linalg.inv(rs32chess_ex_matrix))

        check_ex_matrix_guess(kinect_frame, realsense1_frame, realsense2_frame,
                              realsense3_frame, kinect2chess_ex_matrix,
                              rs12chess_ex_matrix, rs22chess_ex_matrix,
                              rs32chess_ex_matrix, world_point)

    kinect2rs3_ex_matrix = rs32chess_ex_matrix.dot(np.linalg.inv(kinect2chess_ex_matrix))
    rs12rs3_ex_matrix = rs32chess_ex_matrix.dot(np.linalg.inv(rs12chess_ex_matrix))
    rs22rs3_ex_matrix = rs32chess_ex_matrix.dot(np.linalg.inv(rs22chess_ex_matrix))

    if check:
        check_cam2cam_matrix(realsense3_frame, kinect2chess_ex_matrix,
                             kinect2rs3_ex_matrix, calib_dataset.realsense3_in_matrix[0])
        check_cam2cam_matrix(realsense3_frame, rs12chess_ex_matrix,
                             rs12rs3_ex_matrix, calib_dataset.realsense3_in_matrix[0])
        check_cam2cam_matrix(realsense3_frame, rs22chess_ex_matrix,
                             rs22rs3_ex_matrix, calib_dataset.realsense3_in_matrix[0])

    return kinect2rs3_ex_matrix, rs12rs3_ex_matrix, rs22rs3_ex_matrix


def train(model, get_loss, train_dataset, optimizer, epoch):
    model.train()
    get_loss.train()
    loss_all = 0
    for batch_i, (kinect_cam_pts, rs1_cam_pts, rs2_cam_pts, rs3_cam_pts,
                  rs3_pix_proj_pts, rs3_pix_pts) in enumerate(train_dataset):

        kinect2rs3_cam_pts, rs12rs3_cam_pts, rs22rs3_cam_pts = \
            model(kinect_cam_pts, rs1_cam_pts, rs2_cam_pts)

        loss = get_loss(kinect2rs3_cam_pts, rs12rs3_cam_pts, rs22rs3_cam_pts,
                        rs3_cam_pts, rs3_pix_pts, rs3_pix_proj_pts)

        loss_all += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f'Train Epoch: {epoch:<3,d}',
        #       f'Train Batch: {batch_i:<5,d}',
        #       f'Loss: {loss.item():6.2f}')

    # print('kinect2rs3_ex_matrix\n', model.kinect2rs3_ex_matrix)
    # print('rs1rs3_ex_matrix\n', model.rs1rs3_ex_matrix)
    # print('rs2rs3_ex_matrix\n', model.rs2rs3_ex_matrix)
    print(f'Train Epoch: {epoch:<3,d}', 'all loss', loss_all)


def test(model, get_loss, train_dataset, optimizer, epoch):
    model.eval()
    get_loss.eval()
    loss_all = 0
    with torch.no_grad():
        for batch_i, (kinect_cam_pts, rs1_cam_pts, rs2_cam_pts, rs3_cam_pts,
                      rs3_pix_proj_pts, rs3_pix_pts) in enumerate(train_dataset):
            kinect2rs3_cam_pts, rs12rs3_cam_pts, rs22rs3_cam_pts = \
                model(kinect_cam_pts, rs1_cam_pts, rs2_cam_pts)

            loss = get_loss(kinect2rs3_cam_pts, rs12rs3_cam_pts, rs22rs3_cam_pts,
                            rs3_cam_pts, rs3_pix_pts, rs3_pix_proj_pts)

            loss_all += loss

            # print(f'Test Epoch: {epoch:<3,d}',
            #       f'Test Batch: {batch_i:<5,d}',
            #       f'Loss: {loss.item():6.2f}')

    print(f'Test Epoch: {epoch:<3,d}', 'all loss', loss_all)


def add_dim_T(nparray):
    if nparray is None:
        return None

    return np.hstack([nparray, np.ones([nparray.shape[0], 1])]).T


def pix_proj(cam2_pts):
    if cam2_pts is None:
        return None

    cam2_pts = np.divide(cam2_pts, cam2_pts[-1, :])
    cam2_pts = cam2_pts.T[:, :2]

    return cam2_pts


def np_dot(m1, m2):
    if m1 is None or m2 is None:
        return None

    return np.dot(m1, m2)


def viz(model, calib_dataset, epoch):
    kinect2rs3_ex_matrix = model.kinect2rs3_ex_matrix.clone().detach().numpy()
    rs1rs3_ex_matrix = model.rs1rs3_ex_matrix.clone().detach().numpy()
    rs2rs3_ex_matrix = model.rs2rs3_ex_matrix.clone().detach().numpy()

    # viz_3d_space(kinect2rs3_ex_matrix, rs1rs3_ex_matrix, rs2rs3_ex_matrix)
    # return 0

    rs3_in_matrix_ = calib_dataset.realsense3_in_matrix

    matched_frames = calib_dataset.matched_frames
    for i, frames in tqdm(enumerate(matched_frames)):
        kinect_frame = frames['kinect_frame']
        rs1_frame = frames['rs1_frame']
        rs2_frame = frames['rs2_frame']
        rs3_frame = frames['rs3_frame']

        kinect_frame.get_cam_pixproject_pts()
        rs1_frame.get_cam_pixproject_pts()
        rs2_frame.get_cam_pixproject_pts()
        rs3_frame.get_cam_pixproject_pts()

        kinect_cam_pts = add_dim_T(kinect_frame.cam_pts)
        rs1_cam_pts = add_dim_T(rs1_frame.cam_pts)
        rs2_cam_pts = add_dim_T(rs2_frame.cam_pts)
        rs3_cam_pts = add_dim_T(rs3_frame.cam_pts)

        kinect2rs3_cam_pts = np_dot(kinect2rs3_ex_matrix, kinect_cam_pts)
        rs12rs3_cam_pts = np_dot(rs1rs3_ex_matrix, rs1_cam_pts)
        rs22rs3_cam_pts = np_dot(rs2rs3_ex_matrix, rs2_cam_pts)

        rs3_in_matrix = np.zeros((3, 4))
        rs3_in_matrix[:, :3] = rs3_in_matrix_[0]

        kinect2rs3_pix_pts = np_dot(rs3_in_matrix, kinect2rs3_cam_pts)
        rs12rs3_pix_pts = np_dot(rs3_in_matrix, rs12rs3_cam_pts)
        rs22rs3_pix_pts = np_dot(rs3_in_matrix, rs22rs3_cam_pts)
        rs3_cam2pix_pts = np_dot(rs3_in_matrix, rs3_cam_pts)

        kinect2rs3_pix_pts = pix_proj(kinect2rs3_pix_pts)
        rs12rs3_pix_pts = pix_proj(rs12rs3_pix_pts)
        rs22rs3_pix_pts = pix_proj(rs22rs3_pix_pts)
        rs3_cam2pix_pts = pix_proj(rs3_cam2pix_pts)

        image1 = rs3_frame.color_frame.copy()
        image2 = rs3_frame.color_frame.copy()
        image3 = rs3_frame.color_frame.copy()
        image4 = rs3_frame.color_frame.copy()

        kinect2rs3_proj_image = draw_pts(image1, kinect2rs3_pix_pts)
        rs12rs3_proj_image = draw_pts(image2, rs12rs3_pix_pts)
        rs22rs3_proj_image = draw_pts(image3, rs22rs3_pix_pts)
        rs3proj_image = draw_pts(image4, rs3_cam2pix_pts)

        find_corner_image = rs3_frame.color_frame_draw_chessboard

        proj_image = np.vstack([
            np.hstack([find_corner_image, rs3proj_image,
                       rs3proj_image]),
            np.hstack([kinect2rs3_proj_image, rs12rs3_proj_image,
                       rs22rs3_proj_image])])

        scale = 1.5
        proj_image = cv.resize(proj_image, (int(proj_image.shape[1]/scale),
                                            int(proj_image.shape[0]/scale)))

        session_id = frames['session_id']
        if not os.path.exists(f'debug_img/1exam_cam2cam_ex_matrix/epoch{epoch}'):
            os.makedirs(f'debug_img/1exam_cam2cam_ex_matrix/epoch{epoch}/')

        cv.imwrite(f'debug_img/1exam_cam2cam_ex_matrix/epoch{epoch}/'
                   f'{i}_s{session_id}.jpg', proj_image)
    np.save(f'debug_img/1exam_cam2cam_ex_matrix/epoch{epoch}.npy',
            [kinect2rs3_ex_matrix, rs1rs3_ex_matrix, rs2rs3_ex_matrix])


def viz_3d_space(cam12camb_ex_matrix, cam22camb_ex_matrix,
                 cam32camb_ex_matrix, cam42camb_ex_matrix=None):
    import open3d as o3d
    x = np.linspace(0, 100, 10100)
    camb_axis = np.zeros((17*np.size(x), 3))
    camb_axis[:np.size(x), 0] = x
    camb_axis[np.size(x):4*np.size(x), 1] = np.linspace(0, 300, 30300)
    camb_axis[4*np.size(x):14*np.size(x), 2] = np.linspace(0, 1000, 101000)

    point_pcd_b = o3d.geometry.PointCloud()
    point_pcd_b.points = o3d.utility.Vector3dVector(camb_axis)
    point_pcd_b.paint_uniform_color([1, 0.706, 0])

    cam1_axis = camb_axis.copy()
    cam12camb_axis = np.hstack([cam1_axis, np.ones([cam1_axis.shape[0], 1])])
    cam12camb_axis = np.dot(cam12camb_ex_matrix, cam12camb_axis.T).T
    cam12camb_axis = cam12camb_axis[:, :3]

    point_pcd_12b =o3d.geometry.PointCloud()
    point_pcd_12b.points = o3d.utility.Vector3dVector(cam12camb_axis)
    point_pcd_12b.paint_uniform_color([1, 0, 0])

    cam2_axis = camb_axis.copy()
    cam22camb_axis = np.hstack([cam2_axis, np.ones([cam2_axis.shape[0], 1])])
    cam22camb_axis = np.dot(cam22camb_ex_matrix, cam22camb_axis.T).T
    cam22camb_axis = cam22camb_axis[:, :3]

    point_pcd_22b = o3d.geometry.PointCloud()
    point_pcd_22b.points = o3d.utility.Vector3dVector(cam22camb_axis)
    point_pcd_22b.paint_uniform_color([0, 1, 0])

    cam3_axis = camb_axis.copy()
    cam32camb_axis = np.hstack([cam3_axis, np.ones([cam3_axis.shape[0], 1])])
    cam32camb_axis = np.dot(cam32camb_ex_matrix, cam32camb_axis.T).T
    cam32camb_axis = cam32camb_axis[:, :3]

    point_pcd_32b = o3d.geometry.PointCloud()
    point_pcd_32b.points = o3d.utility.Vector3dVector(cam32camb_axis)
    point_pcd_32b.paint_uniform_color([0, 0, 1])

    if cam42camb_ex_matrix is not None:
        cam4_axis = camb_axis.copy()
        cam42camb_axis = np.hstack([cam4_axis, np.ones([cam4_axis.shape[0], 1])])
        cam42camb_axis = np.dot(cam42camb_ex_matrix, cam42camb_axis.T).T
        cam42camb_axis = cam42camb_axis[:, :3]

        point_pcd_42b = o3d.geometry.PointCloud()
        point_pcd_42b.points = o3d.utility.Vector3dVector(cam42camb_axis)
        point_pcd_42b.paint_uniform_color([0, 0.706, 1])

        o3d.visualization.draw_geometries([point_pcd_b, point_pcd_12b,
                                           point_pcd_22b, point_pcd_32b, point_pcd_42b])
    else:
        o3d.visualization.draw_geometries([point_pcd_b, point_pcd_12b,
                                           point_pcd_22b, point_pcd_32b])















