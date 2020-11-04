###################
# show IR1, IR2 and RGB image of realsense camera.
# and estimate human pose
# save them as video.mp4
###################
import os
import numpy as np
import cv2
import pyrealsense2 as rs
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

def stack_to_acceleration(pipeline, num_stack):
    rgb_imgs = []
    for stacks_for_batchsize_i in range(num_stack):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        rgb_imgs.append(color_image)

    if len(rgb_imgs) == 1:
        return color_image
    return rgb_imgs


def setup_realsense():
    # present rgb image, inferred_1 image, inferred2 image and depth image
    # https://www.cnblogs.com/ljxislearning/p/13050821.html
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)  # 10、15或者30可选,20或者25会报错，其他帧率未尝试
    config.enable_stream(rs.stream.infrared, 1, 1280, 800, rs.format.y8, 15)
    config.enable_stream(rs.stream.infrared, 2, 1280, 800, rs.format.y8, 15)
    # config.enable_stream(rs.stream.depth, 1280, 800, rs.format.z16, 15)

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
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align

def get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    # depth_frame = np.asanyarray(aligned_depth_frame.get_data())
    # 将深度图转化为伪彩色图方便观看
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)

    # color frames
    color_frame = aligned_frames.get_color_frame()
    color_frame = np.asanyarray(color_frame.get_data())

    # left　frames
    left_frame = frames.get_infrared_frame(1)
    left_frame = np.asanyarray(left_frame.get_data())

    # right frames
    right_frame = frames.get_infrared_frame(2)
    right_frame = np.asanyarray(right_frame.get_data())

    return color_frame, left_frame, right_frame


def det_posestim(det_model, img, pose_model, args, dataset):
    det_results = inference_detector(det_model, img)

    person_bboxes = det_results[0].copy()

    pose_results = inference_top_down_pose_model(
        pose_model,
        img,
        person_bboxes,
        bbox_thr=args.bbox_thr,
        format='xyxy',
        dataset=dataset)

    vis_img = vis_pose_result(
        pose_model,
        img,
        pose_results,
        dataset=dataset,
        kpt_score_thr=args.kpt_thr,
        show=False)

    return vis_img, pose_results


def main():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--stacks', type=int, default=1, help='Stacks for batchsize to accelerate inference time')

    args = parser.parse_args()

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)

    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device)

    dataset = pose_model.cfg.data['test']['type']

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True


    fps = 5
    size = (1280,
            800)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = 'demo/poses_whole_body.mp4'
    videoWriter = cv2.VideoWriter(
        os.path.join(args.out_video_root,
                     f'vis_{os.path.basename(video_path)}'), fourcc,
        fps, size)


    pipeline, align = setup_realsense()

    i = 0
    left_poses = []
    right_poses = []
    while True:

        rgb_img, left_infferred_img, right_inferred_img = get_frames(pipeline, align)
        left_infferred_img = np.stack([left_infferred_img, left_infferred_img, left_infferred_img], 2)
        right_inferred_img = np.stack([right_inferred_img, right_inferred_img, right_inferred_img], 2)

        # rgb_img, rgb_pose_result = det_posestim(det_model, rgb_img, pose_model, args, dataset)
        left_infferred_img, left_pose_result = det_posestim(det_model, left_infferred_img, pose_model, args, dataset)
        right_inferred_img, right_pose_result = det_posestim(det_model, right_inferred_img, pose_model, args, dataset)

        rgb_inferred2 = np.hstack([left_infferred_img, right_inferred_img])

        cv2.imshow('rgb_inferred2', rgb_inferred2)

        # cv2.imshow('rgb_img', rgb_img)
        # cv2.imshow('left_infferred_img', left_infferred_img)
        # cv2.imshow('right_inferred_img', right_inferred_img)
        # cv2.waitKey(1)

        i += 1
        if i <= 10:
            continue
        # key = cv2.waitKey(1)
        # cv2.imwrite('lab/rgb.png', rgb_img)
        # cv2.imwrite('lab/left.png', left_infferred_img)
        # cv2.imwrite('lab/right.png', right_inferred_img)
        # np.save('lab/rgb_pose_data', rgb_pose_result)
        # np.save('lab/left_pose_data', left_pose_result)
        # np.save('lab/right_pose_data', right_pose_result)
        left_poses.append(left_pose_result)
        right_poses.append(right_pose_result)
        # exit()

        key = cv2.waitKey(1)

        videoWriter.write(left_infferred_img)


        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()
    poses = [left_poses, right_poses]
    np.save('lab/poses_whole_body.npy', poses)

main()