import sys
sys.path.append('core')
from PIL import Image
import os
import argparse

import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from random import sample
from matplotlib import pyplot as plt
import pandas as pd
import time

DEVICE='cpu'

def load_image(imfile):
    im = Image.open(imfile)
    img = np.array(im).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def process_image(img):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, img2, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    plt.figure(figsize=(14,7))
    plt.imshow(np.concatenate([img.astype(np.uint8),img2.astype(np.uint8)],axis=0),)
    plt.figure()
    plt.imshow(np.abs(img-img2))
    # map flow to rgb image
    flo_viz = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img.astype(np.uint8), flo_viz], axis=0)
    plt.figure()
    plt.imshow(flo_viz)
    plt.figure()
    plt.hist(flo.flatten(), bins=np.arange(-25,25))
    plt.show()
    
def process_flow(flow,max_flow=20):
    flow = flow[0].permute(1,2,0).cpu().numpy()
    flow[flow>max_flow] = max_flow
    flow[flow<(-1*max_flow)] = -1 * max_flow
    
    flow_image = 255 * (flow + max_flow) / (2*max_flow)
    flow_image = flow_image.astype(np.uint8)
    flow_image = np.concatenate([flow_image, np.zeros((flow.shape[0],flow.shape[1],1),dtype=np.uint8)], axis=2)
    return flow_image

def load_video(video_fname):
    frames = []
    cap = cv2.VideoCapture(video_fname)
    ret = True
    fps = None
    fourcc = None
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    return frames, fps, fourcc

def write_frame_list_to_video(frames_list, fps, video_fname):
    h, w, _ = frames_list[0].shape
    num_frames = len(frames_list)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(video_fname, fourcc, fps, (w, h))
    for frame in frames_list:
        frame_fixed = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_fixed)
    writer.release()


def video_flow(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()
    input_dir = args.input_path
    all_videos = sorted(glob.glob(input_dir +'/**/**/*.mp4'))
    all_videos = sorted(sample(all_videos,3))
    output_dir = args.output_path
    with torch.no_grad():
        df = pd.DataFrame(([True] * len(all_videos)), index =all_videos,
                                                  columns =['has_flow'])
        for video_fname in all_videos:
            #print(video_fname)
            rgb_frames, fps, fourcc = load_video(video_fname)
            if len(rgb_frames) < 2:
                df.loc[video_fname,'has_flow']=False
            flow_video = []
            for image1, image2 in zip(rgb_frames[:-1], rgb_frames[1:]):
                image1_proc = process_image(image1)
                image2_proc = process_image(image2)
                padder = InputPadder(image1_proc.shape)
                image1_pad, image2_pad = padder.pad(image1_proc,image2_proc)
                start=time.time()
                flow_low_pad, flow_up_pad = model(image1_pad, image2_pad, iters=10, test_mode=True)
                end = time.time()
                flow_up = padder.unpad(flow_up_pad)
                #viz(image1, image2, flow_up)
                flow_image = process_flow(flow_up)
                flow_video.append(flow_image)
            save_filename = video_fname.replace(input_dir, output_dir)
            save_dir = os.path.dirname(save_filename)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            write_frame_list_to_video(flow_video, fps, save_filename)
        df.to_csv('video_has_flow.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--input_path', help="dataset for evaluation")
    parser.add_argument('--output_path', help="output path for optical flow videos")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    video_flow(args)
