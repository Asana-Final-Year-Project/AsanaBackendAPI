import numpy as np
import os
import mediapipe as mp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2 as cv2
import pandas as pd
import os
import glob as glob


globals = []
target = ['bhujan', 'padmasan', 'shav', 'tadasana','trik', 'vriksh' ]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNN(nn.Module):

  def __init__(self,hidden_dim=80, input_dim=99, sequence_num=16, n_layers=1):
    super(RNN, self).__init__()
    self.input_dim = input_dim
    self.sequence_num = sequence_num
    self.n_layers = n_layers
    self.hidden_dim = hidden_dim
    self.rnn = nn.LSTM(input_dim,hidden_dim, n_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, 64)
    self.fc1 = nn.Linear(64,20)
    self.fc2 = nn.Linear(20,6) 

  def forward(self, input):
    h0 = torch.zeros(self.n_layers,  self.hidden_dim)
    c0 = torch.zeros(self.n_layers,  self.hidden_dim)
    out,( _,_) = self.rnn(input,(h0, c0))
    pred = self.fc(out)
    pred = self.fc1(pred)
    pred = self.fc2(pred)
    output = nn.Softmax( dim=1)(pred)
    return output

class YogaDataset(Dataset):
  def __init__(self, df):
    self.features = df
    # self.target = df['1.1']

  def __len__(self):
    return len(self.features)
  
  def __getitem__(self, index):
    features = self.features.loc[index]
    # target = self.target[index]
    return torch.tensor(features.tolist()).float().to(device)

def landmark_det(image):
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_pose = mp.solutions.pose

  BG_COLOR = (192, 192, 192)
  with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
    a = []
    image_height, image_width, _ = image.shape
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i in range(33):
      a.append((results.pose_landmarks.landmark[mp_pose.PoseLandmark(i)].x ,results.pose_landmarks.landmark[mp_pose.PoseLandmark(i)].y ,results.pose_landmarks.landmark[mp_pose.PoseLandmark(i)].z))
    b = []
    for i in a:
      for x in i:
        b.append(x)
    df = pd.DataFrame(b).T
    df = df.to_numpy()
    return df

def listframe(path):
    local = []
    video = cv2.VideoCapture(path)
    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    f_img = 0
    count = 0
    while (count<frame_num): 
        try:
            success, cap = video.read()
            if count % (int(frame_num / 256)) == 0 and (f_img<256):
                df  = landmark_det(cap)
                local.append(df)
                f_img =f_img+1
                count = count+1
            else:
                count = count+1

        except:
          count = count+1
    print(f'done dong doing {path}')
    return local


def pre_processing(videopath):
    data = np.array(listframe(videopath))
    for i in range(len(data)):
        for j in range(len(data[i])):
            globals.append(data[i][j])

def prediction_main(videopath):
  PATH = "yoga_fulldatamodel_test.pth"
  model = RNN().to(device)
  model.load_state_dict(torch.load(PATH))
  pre_processing(videopath)
  df_test = pd.DataFrame(globals)
  tad = YogaDataset(df_test)
  tad_dl = DataLoader(tad, batch_size=16,drop_last=False)
  for batch in tad_dl:
    pred=(model(batch))
    _, is_correct = torch.max(pred.data,1)
  array = is_correct.numpy()
  values, counts = np.unique(array, return_counts=True)
  ind = np.argmax(counts)
  return target[values[ind]]

