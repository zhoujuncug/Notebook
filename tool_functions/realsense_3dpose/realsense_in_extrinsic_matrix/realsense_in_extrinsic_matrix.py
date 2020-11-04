########################
# get realsense's intrinsics and extrinsics matrix using pyrealsense2
########################
import pyrealsense2 as rs
import numpy as np
import cv2

# present rgb image, inferred_1 image, inferred2 image and depth image
# https://www.cnblogs.com/ljxislearning/p/13050821.html


pipeline = rs.pipeline()
config = rs.config()



config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)  #10、15或者30可选,20或者25会报错，其他帧率未尝试
config.enable_stream(rs.stream.infrared, 1, 1280, 800, rs.format.y8, 15)
config.enable_stream(rs.stream.infrared, 2, 1280, 800, rs.format.y8, 15)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

profile = pipeline.start(config)

# turn off IR projector(disable emitter)
# https://github.com/IntelRealSense/librealsense/issues/1258?language=en_US
device = profile.get_device()
depth_sensor = device.query_sensors()[0]
laser_pwr = depth_sensor.get_option(rs.option.laser_power)
laser_range = depth_sensor.get_option_range(rs.option.laser_power)
set_laser = 0
depth_sensor.set_option(rs.option.laser_power, set_laser)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
# align_to = rs.stream.color
# align = rs.align(align_to)


try:
    while True:
        frames = pipeline.wait_for_frames()

    # # Align the depth frame to color frame
    #     aligned_frames = align.process(frames)
    #     # Get aligned frames
    #     aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    #     if not aligned_depth_frame:
    #         continue
    #     depth_frame = np.asanyarray(aligned_depth_frame.get_data())
    # # 将深度图转化为伪彩色图方便观看
    #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
    #     cv2.imshow('1 depth', depth_colormap)
    #
    # # color frames
    #     color_frame = aligned_frames.get_color_frame()
    #     if not color_frame:
    #         continue
    #     color_frame = np.asanyarray(color_frame.get_data())
    #     cv2.imshow('2 color', color_frame)

        color_frame = frames.get_color_frame()
        color_profile = color_frame.get_profile()





    # left　frames
        left_frame = frames.get_infrared_frame(1)
        IR1_profile = left_frame.get_profile()
        IR1profile = rs.video_stream_profile(IR1_profile)
        IR1_intrin = IR1profile.get_intrinsics()
        print('IR1_intrin', IR1_intrin)
        print('-'*50)
        left_frame = np.asanyarray(left_frame.get_data())
        cv2.imshow('3 left_frame', left_frame)

    # right frames
        right_frame = frames.get_infrared_frame(2)
        # left　frames
        IR2_profile = right_frame.get_profile()
        IR2profile = rs.video_stream_profile(IR2_profile)
        IR2_intrin = IR2profile.get_intrinsics()
        IR2_extrin = IR2profile.get_extrinsics_to(IR1profile)
        print('IR2_intrin', IR2_intrin)
        print('IR2_extrin', IR2_extrin)
        print('-' * 50)

        cvsprofile = rs.video_stream_profile(color_profile)
        color_intrin = cvsprofile.get_intrinsics()
        color_extrin = cvsprofile.get_extrinsics_to(IR1profile)
        print('color_intrin', color_intrin)
        print('color_extrin', color_extrin)
        print('-' * 50)

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('rgb_frame', color_image)

        exit()

        right_frame = np.asanyarray(right_frame.get_data())
        cv2.imshow('4 right_frame', right_frame)

        c = cv2.waitKey(1)



        # 如果按下ESC则关闭窗口（ESC的ascii码为27），同时跳出循环
        if c == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()
