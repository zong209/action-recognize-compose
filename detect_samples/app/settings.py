#-*-coding=utf8-*-

BASEHOST='detect-web-server'
PORT='5000'

# classes
CLASS=['others','shibiaoqian','2','3','4']

# logging files path
detect_result='./detect_result.txt'
statis_result='./statis_result.txt'
times_resilt='./times_result.txt'

# video stream path
# VIDEO_PATH='rtmp://193.112.88.179/hls/test3'
VIDEO_PATH='rtmp://193.112.88.179/hls/test'

tsn_label_path='./tsn_label.txt'
tsn_label_path_copy='./tsn_label_copy.txt'

# detect params
NUM=2           #检测视频段时间
fps=25          #视频流的帧率
TIME=1          #每隔2s检测一次
STEP=10         #采样间隔
THRESH=0.4      #判定阈值