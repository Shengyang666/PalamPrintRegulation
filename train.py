import pandas as pd
import torch.utils.data
from torchvision.transforms import ToPILImage
from dataset.dataset import PalamPrintDataset
from dataset.handinfo_dataset import test_dir,train_dir,csv_path_train,csv_path_test
from Net import criterion,optimizer,net
import torchvision.transforms as transforms
from multiprocessing import freeze_support
import os.path
if __name__ == '__main__':
    freeze_support()
#数据的处理
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5),(0.5))])
#加载训练集
train_set=PalamPrintDataset(csv_path_train,train_dir,transform=transform)
train_dataloader=torch.utils.data.DataLoader (train_set,batch_size=4,shuffle=False,num_workers=0)
#loss记录
loss_data=[]
#加载经过训练的网络
net=torch.load('D:\PalamPrintRegulation\ALEXNet_model_120.pth')
#训练100轮
for epoch in range(100):
        running_loss=0.0
        for i,data in enumerate(train_dataloader,0):
            inputs,labels=data
            optimizer.zero_grad()
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if i%30==29:
                print('[%d, %5d] loss: %f' \
                  % (epoch + 1, i + 1, running_loss / 2000))

                loss_data.append((epoch+1,running_loss/2000))
                running_loss = 0.0
        #每20轮保存一次网络
        if epoch%20==0:
            torch.save(net,"D:\PalamPrintRegulation\ALEXNet_model_"+str(epoch)+".pth")
print('Finished Training')

#将loss记录到csv文件
loss_data=pd.DataFrame(loss_data)
dir_path=os.path.dirname(os.path.abspath(__file__))
loss_data.to_csv(dir_path+"\\loss_data.csv",index=False)






