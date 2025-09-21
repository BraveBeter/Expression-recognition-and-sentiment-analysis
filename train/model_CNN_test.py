import cv2
import torch
import torch.nn as nn
import numpy as np
from statistics import mode
from collections import Counter
import time
import matplotlib.pyplot as plt
from tkinter import messagebox
import tkinter
# 人脸数据归一化,将像素值从0-255映射到0-1之间
def preprocess_input(images):
    """ preprocess input by substracting the train mean
    # Arguments: images or image of any shape
    # Returns: images or image with substracted train mean (129)
    """
    images = images / 255.0
    return images


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


class FaceCNN(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(FaceCNN, self).__init__()

        # 第一次卷积、池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.BatchNorm2d(num_features=64),  # 归一化
            nn.RReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大值池化
        )

        # 第二次卷积、池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三次卷积、池化
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 参数初始化
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=8),
        )

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 数据扁平化
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y

def showmap(sum_time,emotion_num,last_time):
    rects = plt.bar(emotion_num, last_time)
    if sum_time <= 300:  #小于5分钟
        for rect in rects:  # rects 是三根柱子的集合
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(format(height, '.1f')), size=15, ha='center',
                     va='bottom')
            plt.ylabel("时间（秒）")
    elif sum_time <= 7200:  # 小于2小时采用分钟计数
        for rect in rects:  # rects 是三根柱子的集合
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(format(height / 60, '.1f')), size=15, ha='center',
                     va='bottom')
            plt.ylabel("时间（分钟）")
    else:
        for rect in rects:  # rects 是三根柱子的集合
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(format(height / 3600, '.1f')), size=15,
                     ha='center',
                     va='bottom')
            plt.ylabel("时间（小时）")
    plt.xlabel("表情")
    plt.title("表情状态持续时间")
    plt.show()
    tip(last_time)

def tip(last_time):
    timex=max(last_time)
    maxindex=last_time.index(timex)
    root=tkinter.Tk()
    root.withdraw()
    if maxindex==0:
        messagebox.showinfo('提示','不要生气了，放轻松!')
    elif maxindex==1:
        messagebox.showinfo('提示','不要厌恶，保持乐观!')
    elif maxindex==2:
        messagebox.showinfo('提示','不要害怕，保持自信!')
    elif maxindex==3:
        messagebox.showinfo('提示','这段时间你很开心!')
    elif maxindex==4:
        messagebox.showinfo('提示','这段时间很自然!内心平静!')
    elif maxindex==5:
        messagebox.showinfo('提示','不要难过，开心点!')
    elif maxindex==6:
        messagebox.showinfo('提示','这段时间时常惊喜，有什么事值得分享!')
    else:
        messagebox.showinfo('提示','这段时间相对较困，找个合适的时间休息一下吧!')


# Opencv自带的一个面部识别分类器
detection_model_path = 'model/haarcascade_frontalface_default.xml'
classification_model_path = '../model/model_cnn8.pkl'

# 加载人脸检测模型
face_detection = cv2.CascadeClassifier(detection_model_path)

# 加载表情识别模型
emotion_classifier = torch.load(classification_model_path)

frame_window = 10
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise',7: 'tired'}
emotion_num=('angry','digust','fear','happy','neutral','sad','surprise','tired')
emotion_window = []
emotion_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,7:0}
emotion_start_time = 0
emotion_total_duration = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,7:0}
last_time=[]
sum_time=0
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['toolbar']='None'
# 调起摄像头，0是笔记本自带摄像头
video_capture = cv2.VideoCapture(0)
# 视频文件识别
#video_capture = cv2.VideoCapture("video/example_dsh.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.startWindowThread()
cv2.namedWindow('window_frame')

while True:
    flag, frame = video_capture.read()
    #print(flag)
    if flag == True :
        frame = frame[:, ::-1, :]
        frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            faces = face_detection.detectMultiScale(gray, 1.3, 5)
        except:
            break

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (84, 255, 159), 2)
            face = gray[y:y+h, x:x+w]

            try:
                # 将人脸大小调整为(48,48)
                face = cv2.resize(face, (48, 48))
            except:
                continue

            face = np.expand_dims(face, 0)
            face = np.expand_dims(face, 0)
            face = preprocess_input(face)
            new_face = torch.from_numpy(face)
            new_new_face = new_face.float().requires_grad_(False)

            emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().numpy())
            emotion = emotion_labels[emotion_arg]
            emotion_window.append(emotion)
            emotion_count[emotion_arg] += 1

            if len(emotion_window) >= frame_window:# 从情绪窗口中移除最早的情绪，确保窗口大小不超过指定的值。
                emotion_window.pop(0)
                emotion_mode = Counter(emotion_count).most_common(1)[0][0]#确定在给定的情绪计数中出现次数最多的情绪，`Counter`类，它用于对可迭代对象进行计数，返回一个字典，其中包含每个元素的计数。`most_common()`方法用于返回出现次数最多的元素，参数1表示我们只需要找出一个最常见的元素。[0]`: 由于`most_common()`返回一个列表，包含元素及其计数，我们通过索引0来获取第一个元素，即出现次数最多的元素及其计数。`[0]`: 最后一个索引0用于获取元素本身，因为我们只需要获取出现次数最多的元素，而不需要其计数。
                emotion_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,7:0}

                # 情绪窗口更新时计算持续时间
                if emotion_start_time != 0:
                    duration = time.perf_counter() - emotion_start_time
                    emotion_total_duration[emotion_mode] += duration
                    #if emotion_total_duration[7] >2:
                        #messagebox.showinfo('提示', '休息一会吧，已经不在状态了')

                emotion_start_time = time.perf_counter()#记录当前时间作为情绪开始时间。这行代码将当前的性能计数器值（即当前时间）赋值给`emotion_start_time`变量，以便后续用于计算情绪持续时间。
                #print(emotion_start_time)

                # 打印所有情绪的总持续时间
                #for emotion, duration in emotion_total_duration.items():
                    #print(f"{emotion_labels[emotion]} Total Duration: {duration} seconds")
                cv2.putText(frame, emotion_labels[emotion_mode], (x, y-30), font, .7, (0, 0, 255), 1, cv2.LINE_AA)

        try:
            cv2.imshow('window_frame', frame)
        except:
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else :
        for emotion, duration in emotion_total_duration.items():
            print(f"{emotion_labels[emotion]} Total Duration: {duration} seconds")
            last_time.append(duration)
        sum_time=sum(last_time)
        showmap(sum_time, emotion_num, last_time)
        break
video_capture.release()
cv2.destroyAllWindows()
if flag==True:
    for emotion, duration in emotion_total_duration.items():
        print(f"{emotion_labels[emotion]} Total Duration: {duration} seconds")
        last_time.append(duration)
    rects=plt.bar(emotion_num, last_time)
    sum_time = sum(last_time)
    showmap(sum_time,emotion_num,last_time)