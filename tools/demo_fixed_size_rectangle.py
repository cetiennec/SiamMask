# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# Modified by [Your Name]
# --------------------------------------------------------
import glob
import argparse
import cv2
import numpy as np
import torch
from os.path import isfile, join
from test import *
from custom import Custom

# Global variables for mouse callback
clicked = False
click_x, click_y = 0, 0

def click_event(event, x, y, flags, param):
    """
    Mouse callback function to capture the click coordinates.
    """
    global clicked, click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y
        clicked = True

def main():
    parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

    parser.add_argument('--resume', default='', type=str, required=True,
                        metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--config', dest='config', default='config_davis.json',
                        help='hyper-parameter of SiamMask in json format')
    parser.add_argument('--base_path', default='/Users/Etienne/PycharmProjects/drone_traj_mavlink/simu/drone_sqTeOgLN.mp4', help='datasets')
    parser.add_argument('--cpu', action='store_true', help='cpu mode')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Open the video file
    video_path = args.base_path  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        exit()

    # Define fixed rectangle size
    FIXED_WIDTH, FIXED_HEIGHT = 200, 200  # Set your desired fixed size here

    # Set up the window and mouse callback
    cv2.namedWindow("SiamMask")
    cv2.setMouseCallback("SiamMask", click_event)

    print("Please click on the desired location in the window to initialize tracking.")

    # Display the first frame and wait for the user to click
    while True:
        display_frame = first_frame.copy()
        if clicked:
            # Draw the fixed-size rectangle at the clicked position for visualization
            top_left_x = int(click_x - FIXED_WIDTH / 2)
            top_left_y = int(click_y - FIXED_HEIGHT / 2)

            # Ensure the rectangle stays within the frame boundaries
            top_left_x = max(0, min(top_left_x, first_frame.shape[1] - FIXED_WIDTH))
            top_left_y = max(0, min(top_left_y, first_frame.shape[0] - FIXED_HEIGHT))

            bottom_right_x = top_left_x + FIXED_WIDTH
            bottom_right_y = top_left_y + FIXED_HEIGHT

            cv2.rectangle(display_frame, (top_left_x, top_left_y),
                          (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
            cv2.imshow("SiamMask", display_frame)
            cv2.waitKey(500)  # Display the rectangle for half a second
            break

        cv2.imshow("SiamMask", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            print("Exiting without selecting ROI.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.destroyWindow("SiamMask")

    # Initialize the tracking rectangle based on the click
    init_rect = (top_left_x, top_left_y, FIXED_WIDTH, FIXED_HEIGHT)
    x, y, w, h = init_rect
    print(f"Initial rectangle: x={x}, y={y}, w={w}, h={h}")

    # Read the rest of the frames
    ims = [first_frame]
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available
        ims.append(frame)

    cap.release()  # Release the video capture object

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int32(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visualization!)'.format(toc, fps))
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
