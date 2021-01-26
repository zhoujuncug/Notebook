import numpy as np
import torch
from pprint import pprint
from torch.utils.data import Dataset, DataLoader

class PtDataset(Dataset):
    def __init__(self, calib_dataset, is_train=True,):
        super(PtDataset, self).__init__()
        matched_frames = calib_dataset.match_frames()
        train_frames = [f for f in matched_frames
                        if f['session_id'] != 'a']
        test_frames = [f for f in matched_frames
                       if f['session_id'] == 'a']

        self.frames = train_frames if is_train else test_frames

    def __len__(self):
        return len(self.frames)

    def get_all(self, frame):
        frame.get_cam_pixproject_pts()
        in_matrix = frame.in_matrix
        ex_matrix = frame.in_matrix
        pix_pts = frame.corners
        cam_pts = frame.cam_pts
        pix_proj_pts = frame.pix_project_pts
        image = frame.color_frame

        return in_matrix, ex_matrix, pix_pts, cam_pts, pix_proj_pts, image

    def np2tensor(self, nparray):
        if nparray is None:
            return torch.zeros([4, 35]).float()

        return torch.from_numpy(nparray).float()

    def add_dim_T(self, nparray):
        if nparray is None:
            return None

        return np.hstack([nparray, np.ones([nparray.shape[0], 1])]).T

    def __getitem__(self, idx):
        kinect_frame = self.frames[idx]['kinect_frame']
        rs1_frame = self.frames[idx]['rs1_frame']
        rs2_frame = self.frames[idx]['rs2_frame']
        rs3_frame = self.frames[idx]['rs3_frame']

        kinect_in_matrix, kinect_ex_matrix, kinect_pix_pts, \
        kinect_cam_pts, kinect_pix_proj_pts, kinect_image = \
            self.get_all(kinect_frame)

        rs1_in_matrix, rs1_ex_matrix, rs1_pix_pts, \
        rs1_cam_pts, rs1_pix_proj_pts, rs1_image\
            = self.get_all(rs1_frame)

        rs2_in_matrix, rs2_ex_matrix, rs2_pix_pts, \
        rs2_cam_pts, rs2_pix_proj_pts, rs2_image\
            = self.get_all(rs2_frame)

        rs3_in_matrix, rs3_ex_matrix, rs3_pix_pts, \
        rs3_cam_pts, rs3_pix_proj_pts, rs3_image\
            = self.get_all(rs3_frame)

        kinect_cam_pts = self.np2tensor(self.add_dim_T(kinect_cam_pts))
        rs1_cam_pts = self.np2tensor(self.add_dim_T(rs1_cam_pts))
        rs2_cam_pts = self.np2tensor(self.add_dim_T(rs2_cam_pts))
        rs3_cam_pts = self.np2tensor(self.add_dim_T(rs3_cam_pts))


        rs3_pix_proj_pts = torch.from_numpy(rs3_pix_proj_pts).float() \
            if rs3_pix_proj_pts is not None else torch.zeros([35, 2]).float()
        rs3_pix_pts = torch.from_numpy(rs3_pix_pts).float() \
            if rs3_pix_pts is not None else torch.zeros([35, 1, 2]).float()

        return kinect_cam_pts, rs1_cam_pts, rs2_cam_pts, \
               rs3_cam_pts, rs3_pix_proj_pts, rs3_pix_pts

