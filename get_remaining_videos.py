import pandas as pd

vl = 'video_list.txt'
data_video = pd.read_csv(vl, sep=";", header=None)
data_video.columns = ["info", "header"]
data_video['info'] = data_video['info'].replace('INFO: ', '')
x = list(data_video['info'])
y = [name[6:] for name in x if '.mp4' in name]
z = [name for name in y if 'raft_strange' not in name]
flow = [name for name in y if 'raft' in name]
video = [name for name in y if 'raft' not in name]
remaining_videos = sorted(list(set(video) - set(flow)))
textfile = open("remaining_videos.txt", "w")
for element in remaining_videos:
    textfile.write(element + "\n")
textfile.close()

