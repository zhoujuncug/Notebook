import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, rs3_in_matrix_):
        super(Loss, self).__init__()
        self.rs3_in_matrix = torch.from_numpy(rs3_in_matrix_[0])
        self.loss_fn = torch.nn.functional.l1_loss

    def l2_loss(self, m1, m2):
        if m1 is None or m2.max() == 0:
            a = torch.tensor(0).float()
            a.requires_grad = True
            return a

        return self.loss_fn(m1, m2)

    def matmul(self, m1, m2):
        if m2 is None:
            return None

        return torch.matmul(m1, m2)

    def forward(self, kinect2rs3_cam_pts, rs12rs3_cam_pts, rs22rs3_cam_pts,
                rs3_cam_pts, rs3_pix_pts, rs3_pix_proj_pts):
        kinect2rs3_loss = self.l2_loss(kinect2rs3_cam_pts, rs3_cam_pts)
        rs12rs3_loss = self.l2_loss(rs12rs3_cam_pts, rs3_cam_pts)
        rs22rs3_loss = self.l2_loss(rs22rs3_cam_pts, rs3_cam_pts)

        loss = kinect2rs3_loss + rs12rs3_loss + rs22rs3_loss

        return loss


class Net(nn.Module):
    def __init__(self, kinect2rs3_ex_matrix_, rs12rs3_ex_matrix_,
                 rs22rs3_ex_matrix_):
        super(Net, self).__init__()

        self.kinect2rs3_ex_matrix = nn.Parameter(torch.from_numpy(kinect2rs3_ex_matrix_).float())
        self.rs1rs3_ex_matrix = nn.Parameter(torch.from_numpy(rs12rs3_ex_matrix_).float())
        self.rs2rs3_ex_matrix = nn.Parameter(torch.from_numpy(rs22rs3_ex_matrix_).float())

        self.kinect2rs3_ex_matrix.requires_grad = True
        self.rs1rs3_ex_matrix.requires_grad = True
        self.rs2rs3_ex_matrix.requires_grad = True

    def matmul(self, m1, m2):
        if m2.max() == 0:
            return None

        return torch.matmul(m1, m2)

    def forward(self, kinect_cam_pts, rs1_cam_pts, rs2_cam_pts):
        kinect2rs3_cam_pts = self.matmul(self.kinect2rs3_ex_matrix, kinect_cam_pts)
        rs12rs3_cam_pts = self.matmul(self.rs1rs3_ex_matrix, rs1_cam_pts)
        rs22rs3_cam_pts = self.matmul(self.rs2rs3_ex_matrix, rs2_cam_pts)



        return kinect2rs3_cam_pts, rs12rs3_cam_pts, rs22rs3_cam_pts






