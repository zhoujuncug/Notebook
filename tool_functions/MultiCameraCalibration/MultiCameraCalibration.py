import os
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lib.CalibDataset import CalibDataset
from lib.PtDataset import PtDataset
from lib.utils import get_init_guess, train, test, viz, viz_3d_space
from lib.model import Net, Loss


def deal_sub(calib_path, kinect_in_matrix, realsense1_in_matrix,
             realsense2_in_matrix, realsense3_in_matrix):

    cd = CalibDataset(calib_path, kinect_in_matrix, realsense1_in_matrix,
                      realsense2_in_matrix, realsense3_in_matrix)
    cd.load_images()

    train_dataset = DataLoader(PtDataset(cd, is_train=True),
                               batch_size=1,
                               num_workers=1,
                               pin_memory=True,
                               shuffle=True)

    test_dataset = DataLoader(PtDataset(cd, is_train=False),
                              batch_size=1,
                              num_workers=1,
                              pin_memory=True,
                              shuffle=False)

    kinect2rs3_ex_matrix, rs12rs3_ex_matrix, rs22rs3_ex_matrix = \
        get_init_guess(cd, check=False)


    viz_3d_space(kinect2rs3_ex_matrix, rs12rs3_ex_matrix, rs22rs3_ex_matrix)

    model = Net(kinect2rs3_ex_matrix, rs12rs3_ex_matrix, rs22rs3_ex_matrix)
    get_loss = Loss(cd.realsense3_in_matrix)

    optimizer = optim.Adadelta(model.parameters(), lr=1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    epochs = 30
    for epoch in range(0, epochs+1):
        if epoch == 0:
            # test(model, get_loss, test_dataset, optimizer, epoch)
            viz(model, cd, epoch)
            # continue

        train(model, get_loss, train_dataset, optimizer, epoch)

        # test(model, get_loss, test_dataset, optimizer, epoch)

        viz(model, cd, epoch)



        scheduler.step()


if __name__ == '__main__':
    calib_path = '/home/j/PycharmProjects/NewDataset/calibration1'
    kinect_color_intrin = np.array([[1102.454, 0,         960 - (938.518 - 480)],
                                   [0,          1099.676, 584.424 - 270],
                                   [0,          0,        1]])
    rs1_color_intrin = np.array([[612.924, 0,       326.315],
                                 [0,       612.382, 236.626],
                                 [0,       0,       1]])
    rs2_color_intrin = np.array([[615.713, 0,       319.684],
                                 [0,       615.783, 242.403],
                                 [0,       0,       1]])
    rs3_color_intrin = np.array([[615.993, 0,       326.349],
                                 [0,       616.134, 235.464],
                                 [0,       0,       1]])

    deal_sub(calib_path, [kinect_color_intrin], [rs1_color_intrin],
             [rs2_color_intrin], [rs3_color_intrin])
