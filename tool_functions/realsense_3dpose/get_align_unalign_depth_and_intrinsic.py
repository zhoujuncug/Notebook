########################
# get the intrinsic matrix of aligned depth image and unaligned depth image
########################
import pyrealsense2 as rs
import numpy as np
import cv2


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)  #10、15或者30可选,20或者25会报错，其他帧率未尝试
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 15)
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 15)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

i = 0
try:
    while True:
        frames = pipeline.wait_for_frames()

        i += 1
        if i < 30:
            continue

        # w/o alignment
        color_frame = frames.get_color_frame()
        color_profile = color_frame.get_profile()
        color_frame = np.asanyarray(color_frame.get_data())
        cv2.imshow('1 color_frame', color_frame)
        cv2.waitKey(0)

        cv2.imwrite('rgb.png', color_frame)
        depth_frame_ = frames.get_depth_frame()
        depth_frame = np.asanyarray(depth_frame_.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.1), cv2.COLORMAP_JET)
        cv2.imshow('2 depth_wo_alignment', depth_colormap)
        cv2.waitKey(0)

        depth_profile = depth_frame_.get_profile()
        depthprofile = rs.video_stream_profile(depth_profile)
        depth_intrin = depthprofile.get_intrinsics()
        print('depth_intrin', depth_intrin)
        print('-'*50)

        # w alignment
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_frame = np.asanyarray(aligned_depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.1), cv2.COLORMAP_JET)
        cv2.imshow('3 depth_w_alignment', depth_colormap)
        cv2.waitKey(0)

        aligned_depth_profile = aligned_depth_frame.get_profile()
        aligned_depthprofile = rs.video_stream_profile(aligned_depth_profile)
        aligned_depth_intrin = aligned_depthprofile.get_intrinsics()
        print('aligned_depth_intrin', aligned_depth_intrin)
        print('-'*50)

        exit()

finally:
    # Stop streaming
    pipeline.stop()
