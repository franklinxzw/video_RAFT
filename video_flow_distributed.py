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
    
def process_flow(flow,max_flow=20):
    "Function for converting flow to saveable images or videos, truncates all values of flow beyond +/- max_flow and stores flow as 3D array of uint8"
    flow = flow[0].permute(1,2,0).cpu().numpy()
    flow[flow > max_flow] = max_flow
    flow[flow < (-1*max_flow)] = -1 * max_flow
    flow_image = 255 * (flow + max_flow) / (2*max_flow)
    flow_image = flow_image.astype(np.uint8)
    flow_image = np.concatenate([flow_image, np.zeros((flow.shape[0],flow.shape[1],1),dtype=np.uint8)], axis=2)
    return flow_image

def load_video(video_fname):
    "Loads video frames with associated metadata"
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
    "Function for writing out frames to videos"
    h, w, _ = frames_list[0].shape
    num_frames = len(frames_list)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(video_fname, fourcc, fps, (w, h))
    for frame in frames_list:
        frame_fixed = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_fixed)
    writer.release()
    
    
def check_if_video_has_been_processed(video):
    video_results = azure_filepaths_for_video_output(video)
    return azcopy_check_if_file_exists(video_results)

def run(node_id):
    video_list = get_video_list(node_id)
    for video in video_list:
        if not check_if_video_has_been_processed(video):
            local_video_path = azcopy_download(video)
            out = compute_optical_flow(local_video_path)
            local_results = save_frames_and_video(out)
            azcopy_upload(local_results)
            remove(local_video_path)
            remove(local_results)
            
def get_video_list_partition(video_list, node_id, total_nodes):
    video_list = sorted(video_list)
    splits = np.array_split(np.array(video_list, dtype=object), total_nodes)
    video_subset_list = list(splits[node_id])
    return video_subset_list

def check_if_video_has_been_processed(video_name, azure_directory):
    #### TODO for Yale ####
    
    
    #######################
    
    
def compute_optical_flow(model, video):
    with torch.no_grad()
        flow_video = []
        # Compute flow one frame at a time instead of batches, or else quality degrades
        # Todo: figure out why quality degrades when using batches
        for image1, image2 in zip(rgb_frames[:-1], rgb_frames[1:]):
            image1_proc = process_image(image1)
            image2_proc = process_image(image2)
            padder = InputPadder(image1_proc.shape)
            image1_pad, image2_pad = padder.pad(image1_proc,image2_proc)
            flow_low_pad, flow_up_pad = model(image1_pad, image2_pad, iters=10, test_mode=True)
            flow_up = padder.unpad(flow_up_pad)
            flow_image = process_flow(flow_up)
            flow_video.append(flow_image)
        return flow_video

def video_flow(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()
    input_dir = args.input_path
    all_videos = sorted(glob.glob(input_dir +'/**/**/*.mp4'))    
    video_subset = get_video_list_partition(all_videos, args.node_id, args.total_nodes)
    output_dir = args.output_path
    df = pd.DataFrame(([True] * len(all_videos)), index =all_videos,
                                              columns =['has_flow'])
    for video_fname in all_videos:
        #print(video_fname)
        rgb_frames, fps, fourcc = load_video(video_fname)
        ## Some videos have no RGB frames at all, skip those and track videos that have optical flow
        if len(rgb_frames) < 2:
            df.loc[video_fname,'has_flow']=False
            continue
        flow_video = compute_optical_flow()
        save_filename = video_fname.replace(input_dir, output_dir)
        save_dir = os.path.dirname(save_filename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        write_frame_list_to_video(flow_video, fps, save_filename)
    csv_name = 'video_has_flow_' + str(args.node_id) +'.csv'
    df.to_csv(csv_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--input_path', help="local dataset path")
    parser.add_argument('--output_path', help="local output path for optical flow videos ")
    parser.add_argument('--upload_path', help = "path for uploading local results")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--node_id', help="ID of node, starting from 0 to total_nodes-1")
    parser.add_arguemnt('--total_nodes', help="Total number of nodes for distributed job")
    args = parser.parse_args()

    video_flow(args)
