import numpy as np
import os
import cv2
from natsort import natsorted


class FrameOneCam:
    def __init__(self, frame_path, color_only, subpix, in_matrix):
        self.frame_path = frame_path
        self.frame_time = int(os.path.split(frame_path)[1].split('.')[0])
        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        self.color_only = color_only
        self.color_frame = None
        self.color_frame_draw_chessboard = None
        self.gray = None
        self.found = None
        self.corners = None
        self.sub_corners = None
        self.subpix = subpix
        self.chessboard_pattern_size = (7, 5)
        self.in_matrix = in_matrix
        self.ex_matrix = None
        self.cam_pts = None
        self.pix_project_pts = None

    def load(self):
        if self.color_only:
            self.color_frame = cv2.imread(self.frame_path)
            if 'Kinect' in self.frame_path:
                self.color_frame = self.color_frame[:, ::-1, :]
            if self.color_frame is None:
                raise ValueError('Image path error.')

            return self.color_frame

    def find_chessboard(self, save_debug_frame=False):
        self.load()

        if self.color_only:
            self.color_frame_draw_chessboard = self.color_frame.copy()
            self.gray = cv2.cvtColor(self.color_frame, cv2.COLOR_BGR2GRAY)

            self.found, self.corners = cv2.findChessboardCorners(self.gray, (7, 5), None)

            if self.found:
                npy_path = self.frame_path.replace('jpg', 'npy')
                npy = np.load(npy_path, allow_pickle=True)
                self.corners = npy
                if 'Kinect' in self.frame_path:
                    self.corners[:, :, 0] = 960 - self.corners[:, :, 0]

                return self.corners

            # if save_debug_frame and not self.found:
            #     cv2.imshow('not found corners', self.color_frame_draw_chessboard)
            #     cv2.waitKey(1)
            #     frame_id = self.frame_path.replace('/', '_')
            #     cv2.imwrite('./debug_img/find_corners/'+frame_id, self.color_frame_draw_chessboard)
            #
            # if self.subpix:
            #     self.corners = cv2.cornerSubPix(self.gray, self.corners, (5, 5),
            #                                     (-1, -1),  self.criteria)

            return self.corners



        else:
            raise ValueError('Only color frame is allowed.')

    def show_chessboard(self):
        self.find_chessboard()
        cv2.drawChessboardCorners(self.color_frame_draw_chessboard,
                                  self.chessboard_pattern_size,
                                  self.corners, self.found)
        frame_id = self.frame_path.replace('/', '_')
        cv2.imwrite('./debug_img/find_corners/' + frame_id,
                    self.color_frame_draw_chessboard)

        return self.color_frame_draw_chessboard

    def get_frame_ex_matrix(self):
        self.show_chessboard()

        if self.corners is None:
            self.ex_matrix = None

            return None

        world_point = np.zeros((7 * 5, 3), np.float32)
        world_point[:, :2] = 30 * np.mgrid[:7, :5].T.reshape(-1, 2)

        dist_coeffs = np.array([0., 0., 0., 0., 0.])

        ret, rvec, tvec = cv2.solvePnP(world_point, self.corners,
                                       self.in_matrix[0], dist_coeffs)
        rotation, _ = cv2.Rodrigues(rvec)
        rotation_t = np.hstack([rotation, tvec])
        ex_matrix = np.vstack([rotation_t, np.array([[0, 0, 0, 1]])])
        self.ex_matrix = ex_matrix

        return self.ex_matrix

    def get_cam_pixproject_pts(self):
        self.get_frame_ex_matrix()

        if self.ex_matrix is None:
            self.cam_pts = None
            self.pix_project_pts = None

            return None, None

        world_point = np.zeros((7 * 5, 3), np.float32)
        world_point[:, :2] = 30 * np.mgrid[:7, :5].T.reshape(-1, 2)
        world_point = np.hstack([world_point, np.ones([35, 1])])

        cam_pts = self.ex_matrix.dot(world_point.T)
        self.cam_pts = cam_pts[:3, :].T.copy()

        in_matrix = np.zeros((3, 4))
        in_matrix[:, :3] = self.in_matrix[0]

        pix_proj_pts = in_matrix.dot(cam_pts)
        pix_proj_pts = np.divide(pix_proj_pts, pix_proj_pts[-1, :])

        self.pix_project_pts = pix_proj_pts.T[:, :2]

        return self.pix_project_pts


class Session:
    def __init__(self, session_path, color_only, subpix,
                 kinect_in_matrix, realsense1_in_matrix,
                 realsense2_in_matrix, realsense3_in_matrix):
        self.kinect = []
        self.realsense1 = []
        self.realsense2 = []
        self.realsense3 = []
        self.subpix = subpix
        self.session_path = session_path
        self.color_only = color_only
        self.kinect_in_matrix = kinect_in_matrix
        self.realsense1_in_matrix = realsense1_in_matrix
        self.realsense2_in_matrix = realsense2_in_matrix
        self.realsense3_in_matrix = realsense3_in_matrix

    def load(self):
        if self.color_only:
            def load_frame(path, in_matrix):
                frames = []
                frames_ids = natsorted([i for i in os.listdir(path) if 'jpg' in i])
                frames_paths = [path + '/' + frame_id for frame_id
                                in frames_ids]
                for frame_path in frames_paths:
                    frame = FrameOneCam(frame_path, self.color_only,
                                        self.subpix, in_matrix)
                    frames.append(frame)

                return frames

            kinect_path = self.session_path + '/right/Kinect/color'
            realsense1_path = self.session_path + '/right/Realsense/camera0/color_frame'
            realsense2_path = self.session_path + '/right/Realsense/camera1/color_frame'
            realsense3_path = self.session_path + '/right/Realsense/camera2/color_frame'

            self.kinect = load_frame(kinect_path,
                                     self.kinect_in_matrix)
            self.realsense1 = load_frame(realsense1_path,
                                         self.realsense1_in_matrix)
            self.realsense2 = load_frame(realsense2_path,
                                         self.realsense2_in_matrix)
            self.realsense3 = load_frame(realsense3_path,
                                         self.realsense3_in_matrix)

        else:
            raise ValueError('Only color frame is allowed.')


class CalibDataset:
    def __init__(self, path, kinect_in_matrix=None, realsense1_in_matrix=None,
                 realsense2_in_matrix=None, realsense3_in_matrix=None,
                 subpix=False, color_only=True):
        self.sessions = []  # type: list[Session]
        self.subpix = subpix  # type: bool
        self.path = path  # type: str
        self.color_only = color_only  # type: bool
        self.kinect_in_matrix = kinect_in_matrix
        self.realsense1_in_matrix = realsense1_in_matrix
        self.realsense2_in_matrix = realsense2_in_matrix
        self.realsense3_in_matrix = realsense3_in_matrix
        self.matched_sessions = []
        self.matched_frames = []

    def load_images(self):
        sessions_dirs = natsorted(os.listdir(self.path))
        sessions_dirs_path = [self.path + '/' + sessions_dir
                              for sessions_dir in sessions_dirs]

        for session_path in sessions_dirs_path:
            session = Session(session_path, self.color_only,
                              self.subpix, self.kinect_in_matrix,
                              self.realsense1_in_matrix,
                              self.realsense2_in_matrix,
                              self.realsense3_in_matrix)
            session.load()
            session.subpix = self.subpix
            self.sessions.append(session)

    def match_sessions(self):
        def match_session(session, sess_id):
            kinect_frames = session.kinect
            rs1_frames = session.realsense1
            rs2_frames = session.realsense2
            rs3_frames = session.realsense3

            kinect_frames_times = np.array([frame.frame_time for frame in kinect_frames])
            rs1_frames_times = np.array([frame.frame_time for frame in rs1_frames])
            rs2_frames_times = np.array([frame.frame_time for frame in rs2_frames])
            rs3_frames_times = np.array([frame.frame_time for frame in rs3_frames])

            rs3_kinect_match_matrix = rs3_frames_times - kinect_frames_times[:, None]
            rs3_rs1_match_matrix = rs3_frames_times - rs1_frames_times[:, None]
            rs3_rs2_match_matrix = rs3_frames_times - rs2_frames_times[:, None]

            rs3_kinect_match_id = np.argmin(np.abs(rs3_kinect_match_matrix), axis=0)
            rs3_rs1_match_id = np.argmin(np.abs(rs3_rs1_match_matrix), axis=0)
            rs3_rs2_match_id = np.argmin(np.abs(rs3_rs2_match_matrix), axis=0)

            matched_session = []
            for id, rs3_frame in enumerate(rs3_frames):
                matched_session.append({
                    'session_id': sess_id,
                    'kinect_frame': kinect_frames[rs3_kinect_match_id[id]],
                    'rs1_frame': rs1_frames[rs3_rs1_match_id[id]],
                    'rs2_frame': rs2_frames[rs3_rs2_match_id[id]],
                    'rs3_frame': rs3_frames[id],
                })

            return matched_session

        for sess_id, session in enumerate(self.sessions):
            self.matched_sessions.append(match_session(session, sess_id))

    def match_frames(self):
        self.match_sessions()
        for matched_session in self.matched_sessions:
            for matched_frame in matched_session:
                self.matched_frames.append(matched_frame)

        return self.matched_frames


# test
if __name__ == '__main__':
    kinect_color_intrin = np.array([[1102.454, 0,         938.518 - 480],
                                   [0,          1099.676, 584.424 - 270],
                                   [0,          0,        1]])
    rs1_color_intrin = np.array([[919.385, 0,       649.473],
                                 [0,       918.573, 354.939],
                                 [0,       0,       1]])
    rs2_color_intrin = np.array([[923.57, 0,        639.526],
                                 [0,       923.674, 363.604],
                                 [0,       0,       1]])
    rs3_color_intrin = np.array([[923.989, 0,       649.523],
                                 [0,       924.201, 353.196],
                                 [0,       0,       1]])
    cd = CalibDataset('/home/j/PycharmProjects/NewDataset/calibration1',
                      [kinect_color_intrin], [rs1_color_intrin],
                      [rs2_color_intrin], [rs3_color_intrin])

    cd.load_images()
    cd.match_frames()
    match_frames = cd.matched_frames
    # print(len(match_frames))
    for d in match_frames:
        print(d['kinect_frame'].get_cam_pixproject_pts())