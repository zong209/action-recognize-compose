# -*- coding=utf8 -*-
# USAGE
# python simple.py

# import the necessary packages
import requests
import cv2
import json
import numpy as np
import helpers
from collections import deque
import os
import time
from socketIO_client import SocketIO,LoggingNamespace
import settings
import multiprocessing

# 记录结果
detect_result=settings.detect_result
statis_result=settings.statis_result
times_resilt=settings.times_resilt

os.system("cat /dev/null > "+detect_result)
os.system("cat /dev/null > "+statis_result)
os.system("cat /dev/null > "+times_resilt)

detect_file=open(detect_result,'a')
statis_file=open(statis_result,'a')
times_file=open(times_resilt,'a')

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://"+settings.BASEHOST+":"+settings.PORT+"/predict"
print(KERAS_REST_API_URL)
# IMAGE_PATH = "../data/demo_data/clips/clips_0/000000.jpg"
# IMAGE_PATH = "../data/demo_data/clips/clips_0/000000.jpg"

# 构造队列
class my_queue(deque):
    def __init__(self,max_size):
        self.max_size = max_size        
    # keep the length of queue    
    def keep_queue(self,value):        
        if len(self) < self.max_size:            
            self.append(value)        
        else:            
            self.popleft()            
            self.append(value)    
    # print queue     
    def print_queue(self):  
        print (list(self))
        
# 发送检测请求
def post_detect(image,cnts,width,height):
    image=image.copy(order="C")
    image = helpers.base64_encode_image(image)
    payload={"image":image,'cnts':cnts,'width':width,'height':height}
    # submit the request
    r = requests.post(KERAS_REST_API_URL, json=json.dumps(payload)).json()
    # print (image.shape)
    if r["success"]:
        # loop over the predictions and display them
        return r["predictions"]
    else:
        return []

def statis_detect(pre,cur,total,duration,times,THRESH):
    """
    pre cur:上次检测结果 当前检测结果 
            Array [labels frames prob]
    total:当前总帧数
    duration:动作持续帧数
    times:动作出现次数
    """
    total=cur[1]
    # 判断当前置信度
    if cur[2]>THRESH and pre[2]>THRESH:
        if(cur[0]!=pre[0]):
            times+=1
        if(cur[0]==settings.CLASS[1]):
            duration+=cur[1]-pre[1]
    return total,duration,times

# 获取时间戳
def get_times(pre,cur,timestamp):
    """
    pre cur:上次检测结果 当前检测结果 
        Array [labels frames prob]
    cnts:当前帧数
    times: Array [start,end]
    """
    start=timestamp[0]
    end=timestamp[1]
    # print(pre[0],cur[0])
    if(pre[0]!=settings.CLASS[1] and cur[0]==settings.CLASS[1] and end == '' and start==''):
        start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if(pre[0]==settings.CLASS[1] and cur[0]!=settings.CLASS[1] and start != '' and end==''):
        end=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return [start,end] 

# socket发送结果
def emit_sk(label,interval,label_flag,interval_flag):
    while True:
        with SocketIO(settings.BASEHOST,settings.PORT,LoggingNamespace) as socketIO:
            if label_flag['label_flag']:
                socketIO.emit('result',{'name':'label','result':label['label']})
                label_flag['label_flag']=False
            if interval_flag['interval_flag']:
                socketIO.emit('result',{'name':'interval','result':interval['timestamp']})
                interval_flag['interval_flag']=False
            # socketIO.wait(seconds=1)

def drawtxt(THRESH,frame_cnt,class_label,class_possible):
    # 判断文件是否可写
    tsn_label_path=settings.tsn_label_path
    tsn_label_path_copy=settings.tsn_label_path_copy
    if(class_possible>THRESH):
        file=open(tsn_label_path,'w')
        file.write("{:5d}".format(frame_cnt)+': '+class_label+'  '+"{:.3f}".format(class_possible))
        file.close()
        helpers.mycopyfile(tsn_label_path,tsn_label_path_copy)

def start_detect(label,interval,label_flag,interval_flag):
    # 初始化参数
    fps=settings.fps          #视频流的帧率
    TIME=settings.TIME        #每隔1s检测一次
    STEP=settings.STEP        #采样间隔
    THRESH=settings.THRESH    #判定阈值
    NUM=settings.NUM          #检测n秒的片段

    # 函数内全局变量
    frames=[]       #用于存储用于检测的视频帧
    cnt=0           #用于稀疏取帧
    frame_cnt=0     #用于统计视频帧
    MAX=int(fps)*NUM  #检测n秒的片段
    INTEVAL=int(MAX/STEP)
    frames_queue=my_queue(MAX)
    total=0         #总帧数
    duration=0      #动作持续帧数
    times=0         #动作出现次数
    pre=['',0,0]    #上一次检测结果
    timestamp=['','']   #上一次动作的时间区间
    global vs
    if not vs.open(VIDEO_PATH):
        print("can not open the video")
    else:
    #     print "FPS:",vs.get(cv2.CAP_PROP_FPS)
    #     fps=vs.get(cv2.CAP_PROP_FPS)
        while True:
            grabbed,frame=vs.read()
            if grabbed:
                frame_cnt+=1
                frame=cv2.resize(frame,(340,256))
                (width,height,chn)=frame.shape
                frames_queue.keep_queue(frame)           
                # print width,height
                # if len(frames)==0:
                #     frames=frame
                # else:
                #     frames=np.concatenate((frames,frame))
                cnt+=1
                if len(frames_queue)==MAX and cnt%(TIME*fps)==0:
                    # print "start detect ..."
                    frames=list(frames_queue)
                    #随机初始帧
                    t0=np.random.randint(STEP)
                    frames_reg=np.array([frames[t0+STEP*i] for i in range(0,INTEVAL)]).reshape((-1,height,3))
                    result=post_detect(frames_reg,INTEVAL,width,height)
#                     print("result",result)
                    class_label=settings.CLASS[eval(result)[0]['label']]
                    class_possible=eval(result)[0]['probability']
                    # 统计功能
                    cur=[class_label,frame_cnt,class_possible]
                    total,duration,times=statis_detect(pre,cur,total,duration,times,THRESH)
                    # 获取时间
                    if(class_possible>THRESH):
                        timestamp=get_times(pre,cur,timestamp)
                        if(timestamp[0]!='' and timestamp[1]!=''):
                            print ('上次动作时间:',timestamp)
                            interval['timestamp']=timestamp
                            interval_flag['interval_flag']=True
                            times_file.write('上次动作时间:'+str(timestamp)+'\n')
                            times_file.flush()
                            timestamp=['','']
                    # print ('总帧数:',total,'撕标签总帧数:',duration,'动作次数:',times)
                    statis_file.write('总帧数:'+str(total)+'\t撕标签总帧数:'+str(duration)+'\t动作次数:'+str(times)+'\n')
                    statis_file.flush()

                    print ('class:',class_label,'frame_cnt:',frame_cnt,'possible:',class_possible)
                    detect_file.write('class:'+class_label+'\tframe_cnt:'+str(frame_cnt)+'\tpossible:'+str(class_possible)+'\n')
                    detect_file.flush()
                    #print 'class:',class_label,';frame_cnt:',frame_cnt,';result:',result
                    if(class_possible<THRESH):
                        cur[0]=pre[0]
                    # go to next loop
                    label['label']=cur[0]
                    label_flag['label_flag']=True
                    pre=cur
                    frames=[]
                    cnt=0
                    # 生成用于显示标签的文档
                    # drawtxt(THRESH,frame_cnt,class_label,class_possible)
            else:
                break

        detect_file.close()
        statis_file.close()
        times_file.close()

if __name__=='__main__':

    VIDEO_PATH=settings.VIDEO_PATH
    print (VIDEO_PATH)
    vs=cv2.VideoCapture()

    with multiprocessing.Manager() as manager:
        label=manager.dict({'label':''})
        interval=manager.dict({'timestamp':[]})
        label_flag=manager.dict({'label_flag':False})
        interval_flag=manager.dict({'interval_flag':False})
        
        process=multiprocessing.Process(target=start_detect,args=(label,interval,label_flag,interval_flag))
        process.start()
        emit_sk(label,interval,label_flag,interval_flag)
