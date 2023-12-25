from torch.utils.data import Dataset
import os
from PIL import Image
import dataset.handinfo_dataset
import os.path
import os
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
class PalamPrintDataset(Dataset):
    def __init__(self,csv_path,img_dir,transform=None):
        #self.label_name={'0':0,"4":1,"8":2,"10":3,"12":4,"13":5,"19":6,"23":7,"25":8,"27":9,"28":10}
        self.data_info = self.get_img_info(img_dir,csv_path)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform
    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('L')  # 0~255
        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等
        return img, label
    def __len__(self):
        return len(self.data_info)
    def get_img_info(self,data_dir,csv_path):
        data_info = list()
        df=pd.read_csv(csv_path)
        img_names=df['imageName'].values
        #img_name=data_dir+"\\"+img_name
        img_names=list(img_names)
        for i in range(len(img_names)):
            img_name=img_names[i]
            path_img=os.path.join(data_dir,img_name)
            tmp=df[df['imageName'] == img_name]
            label=tmp.at[tmp.index[0],'id']
            data_info.append((path_img,int(label)))
        return data_info


