import pandas as pd
import torch.utils.data
from dataset.dataset import PalamPrintDataset
from dataset.handinfo_dataset import test_dir,train_dir,csv_path_train,csv_path_test
import torchvision.transforms as transforms
import os.path


#测试集文件夹路径
dir_path=os.path.dirname(os.path.abspath(__file__))
#测试集预处理和加载
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))
                              ])
test_set=PalamPrintDataset(csv_path_test,test_dir,transform=transform)
test_dataloader=torch.utils.data.DataLoader(test_set,batch_size=4,shuffle=False,num_workers=0)
#准确率
acc_data=[]
#开始不同网络的测试
for k in range(7):
     correct_num=0
     dataiter = iter(test_dataloader)
     #加载不同网络
     net=torch.load("D:\PalamPrintRegulation\ALEXNet_model_"+str(k*20+20)+".pth")
     #测试
     for i in range(20):
          images,labels=dataiter.__next__()
          labels=labels.numpy().tolist()
          print("real labels:",labels)
          outputs=net(images)
          _,predicted=torch.max(outputs.data,1)
          predicted=predicted.numpy().tolist()
          print("predicted labels:",predicted)
          print('\n')
          for i in range(4):
            if labels[i]==predicted[i]:
                correct_num=correct_num+1
     print("accuracy:",correct_num/80)
     print('\n')
     #准确率添加
     acc_data.append(correct_num/80)
#准确率生成csv
acc_data=pd.DataFrame(acc_data)
acc_data.to_csv("D:\\PalamPrintRegulation\\acc_data.csv")



''',transforms.Normalize((0.5),(0.5))'''
