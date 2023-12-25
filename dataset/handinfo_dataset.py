import os.path
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/handinfo.pkl"
train_dir=dataset_dir+"\\dataset_train"
test_dir=dataset_dir+"\\dataset_test"
csv_path_train=dataset_dir+"\\train.csv"
csv_path_test=dataset_dir+"\\test.csv"

def get_train_path(filename):
    file_path = dataset_dir + "\\dataset_train\\" + filename
    return file_path

def get_test_path(filename):
    file_path = dataset_dir + "\\dataset_test\\" + filename
    return file_path


def _load_label(file_name):  # 提取标签
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    labels = pd.read_csv(filepath_or_buffer=file_path, header=0)
    print("Done")
    return labels


##清洗数据集，删除不必要的图片
def delete_img_and_csv(dir_name, labels):  ##图片文件夹，标签
    imgs = os.listdir(dir_name)
    df = pd.DataFrame(labels)
    selected_rows = df[df['aspectOfHand'].isin(['dorsal right', 'dorsal left'])]
    selected_cols = selected_rows['imageName'].values
    for img in imgs:
        if img in selected_cols:
            delete_path = dir_name + "/" + img
            os.remove(delete_path)

            print("Remove" + img + '\n')
    labels = df[df['aspectOfHand'].isin(['palmar left', 'palmar right'])]
    labels.to_csv(dataset_dir + "//handinfo.csv", index=False)
    print("Done")

##从图片名得到具体标签
def _get_label(filename, labels):
    df = pd.DataFrame(labels)
    selected_row = df[df['imageName'] == filename]
    if selected_row.empty:
        print("Image Not Exist\n")
    else:
        print("Get Image Label")
        return selected_row

def img_resize(file_path):
  #  file_path = get_test_path(file_name)
    #  待处理图片路径
    img_path = Image.open(file_path)
    #  resize图片大小，入口参数为一个tuple，新的图片的大小
    img_size = img_path.resize((224, 224))
    #  处理图片后存储路径，以及存储格式
    img_size.save(file_path, 'JPEG')
    print("Resize " + file_path + '\n')

#灰度图
def img_to_GREY(file_path):
    img = Image.open(file_path).convert('L')
    img.save(file_path)
   
def img_processing():
    for dir in ['\\dataset_test','\\dataset_train']:
        imgs = os.listdir(dataset_dir+dir)
        for img in imgs:
            img_resize(dataset_dir+dir+'\\'+img)
            img_to_GREY(dataset_dir+dir+'\\'+img)
    print("Done")

def csv_processing():
    file='\\test.csv'
    dir='\\dataset_test'
    file_path=dataset_dir+file
    df = pd.read_csv(file_path)
    selected_rows=df[df['id']<30]
    selected_rows.to_csv(file_path,index=False)
    imgs = os.listdir(dataset_dir + dir)
    for img in imgs:
        if img not in selected_rows['imageName'].values:
            os.remove(get_test_path(img))
    file='\\train.csv'
    dir='\\dataset_train'
    file_path=dataset_dir+file
    df = pd.read_csv(file_path)
    selected_rows=df[df['id']<30]
    selected_rows.to_csv(file_path,index=False)
    imgs = os.listdir(dataset_dir + dir)
    for img in imgs:
        if img not in selected_rows['imageName'].values:
            os.remove(get_train_path(img))
    print("Done")
''''''


def split_dataset_csv(csv_name):
    csv_path = dataset_dir + '/' + csv_name
    train_path = dataset_dir + '/' + "train.csv"
    test_path = dataset_dir + '/' + "test.csv"

    # 读取CSV文件
    data = pd.read_csv(csv_path)
    # 划分训练集和测试集（比例为70%训练集，30%测试集）
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    # 保存划分后的训练集和测试集为CSV文件
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

'''
def split_dataset_img():
    train_dir_path = dataset_dir + "\\dataset_train\\"
    test_dir_path = dataset_dir + "\\dataset_test\\"
    train_path = dataset_dir + "\\train.csv"
    test_path = dataset_dir + "\\test.csv"

    data = pd.read_csv(train_path)
    for value in data['imageName'].values:
        img_path = get_img_path(value)
        img = cv.imread(img_path, 0)
        p = train_dir_path + value
        cv.imwrite(p, img=img)

    data = pd.read_csv(test_path)
    for value in data['imageName'].values:
        img_path = get_img_path(value)
        img = cv.imread(img_path)
        cv.imwrite(test_dir_path + value, img=img)

    print("Split Imgs Done\n")

'''
if __name__ == '__main__':
    '''
#删除不需要的图片和csv行
delete_img_and_csv(dataset_dir+"/Hands",labels=labels)
#划分训练集，测试集
split_dataset_csv('handinfo.csv')
split_dataset_img()

img_to_GREY('Hand_0000043.jpg')_load_img('')
labels=_load_label('test.csv')
p=_load_img('dataset_test',labels)
init_dataset()
#csv的处理
csv_processing()

train_path=dataset_dir+'/train.csv'
#删除不需要的列
df=pd.read_csv(csv_path_train)
df=df.drop("age",axis=1)
df=df.drop("gender",axis=1)
df=df.drop("skinColor",axis=1)
df=df.drop("accessories",axis=1)
df=df.drop("nailPolish",axis=1)
df=df.drop("aspectOfHand",axis=1)
df=df.drop("irregularities",axis=1)
# 使用 map() 函数进行映射
label_name = {0: 1, 4: 2, 8: 3, 10: 4, 11:5,12: 6, 13: 7, 19: 8, 23: 9, 25: 10, 27: 11, 28: 12}
df['id'] = df['id'].map(label_name).astype(int)
df.to_csv(csv_path_train,index=False)'''