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
from azure.storage.blob import BlobServiceClient, BlobClient
import yaml

DEVICE='cpu'

def process_image(img):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)
    
def process_flow(flow,max_flow=20):
    "Function for converting flow to saveable images or videos, truncates all values of flow beyond +/- max_flow, normalizes all flow values to [0,255] and stores flow as 3D array of uint8"
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
            
def get_video_list_partition(video_list, node_id, total_nodes):
    video_list = sorted(video_list)
    video_list =np.array(video_list, dtype=object)
    splits = np.array_split(video_list, total_nodes)
    video_subset_list = list(splits[node_id])
    return video_subset_list

def compute_optical_flow(model, rgb_frames):
    with torch.no_grad():
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
    
def read_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        data = yaml.safe_load(stream)
    return data

def video_flow(args): 
    azure_info = read_yaml(args.azure_yaml_file)
    yaml_file = 'azure_info.yaml'
    with open(yaml_file, 'r') as stream:
        data = yaml.safe_load(stream)
    upload_conn_str = data['upload_conn_str']
    upload_container_name = data['upload_container_name']
    download_conn_str = data['download_conn_str']
    download_container_name = data['download_container_name']
    
    #### all functions that handle Azure credentials are within video_flow()
    def azcopy_upload(local_filename):
        """expects full video path names for local_filename
        i.e. 'official/train_256/zumba/-0_5tJuIrJA_000004_000014.mp4
        uses local filenames as upload name
        """
        blob = BlobClient.from_connection_string(conn_str=upload_conn_str, container_name=upload_container_name, blob_name=local_filename)
        try:
            with open(local_filename, "rb") as data:
                blob.upload_blob(data)
        except:
            print("Error uploading file (File may already exist in storage): " + local_filename)
    
    
    def azcopy_download(azure_video_name):
        """expects full video path names for azure_video_name
        i.e. 'official/train_256/zumba/-0_5tJuIrJA_000004_000014.mp4
        also saves local files based on exact directory structure in Azure storage
        """
        blob = BlobClient.from_connection_string(conn_str=download_conn_str, container_name=download_container_name, blob_name=azure_video_name)
        try:
            if not os.path.exists(os.path.dirname(azure_video_name)):
                os.makedirs(os.path.dirname(azure_video_name))
            with open(azure_video_name, "wb") as my_blob:
                blob_data = blob.download_blob()
                blob_data.readinto(my_blob)
        except:
            print("Error downloading file :" + azure_video_name)
    
    def check_if_video_has_been_processed(video_name):
        """expects full video path names for video_name
        i.e. 'official/train_256/zumba/-0_5tJuIrJA_000004_000014.mp4
        """
        blob = BlobClient.from_connection_string(conn_str=upload_conn_str, container_name=download_container_name, blob_name=video_name)
        exists = blob.exists()
        return exists
    
    def get_video_list(azure_dir):
        "videos are assumed to be individual blobs in Azure storage"
        connect_str = download_conn_str
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_name = download_container_name
        container_client = blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs(name_starts_with=azure_dir)
        video_list = [blob.name for blob in blob_list]
        return video_list
    
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()
    input_dir = args.azure_data_path
    all_videos = get_video_list(input_dir)
    video_subset = get_video_list_partition(all_videos, args.node_id, args.total_nodes)
    print(len(all_videos), len(video_subset))
    output_dir = args.output_path
    df = pd.DataFrame(([True] * len(all_videos)), index =all_videos,
                                              columns =['has_flow'])
    for i, video_fname in enumerate(video_subset):
        print(i, video_fname)
        save_filename = video_fname.replace(input_dir, output_dir)
        if not check_if_video_has_been_processed(save_filename):
            azcopy_download(video_fname)
            rgb_frames, fps, fourcc = load_video(video_fname)
            ## Some videos have no RGB frames at all, skip those and track videos that have optical flow
            if len(rgb_frames) < 2:
                df.loc[video_fname,'has_flow']=False
                continue
            flow_video = compute_optical_flow(model, rgb_frames)
            save_dir = os.path.dirname(save_filename)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            write_frame_list_to_video(flow_video, fps, save_filename)
            azcopy_upload(save_filename)
    csv_name = os.path.join(output_dir, 'video_has_flow_' + str(args.node_id) +'.csv')
    df.to_csv(csv_name)
    azcopy_upload(csv_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--azure_data_path', help="dataset path on Azure storage")
    parser.add_argument('--output_path', help="local output path for optical flow videos ")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--node_id', type=int, help="ID of node, starting from 0 to total_nodes-1")
    parser.add_argument('--total_nodes', type=int, help="Total number of nodes for distributed job")
    parser.add_argument('--azure_yaml_file', help="YAML file containing all the info for Azure storage")
    args = parser.parse_args()
    video_flow(args)
