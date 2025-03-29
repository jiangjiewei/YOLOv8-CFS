import os
import torch
from torch.utils import data
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import argparse
# from test.models import *
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--data', metavar='DIR', default="/data/jiangjiewei/dk/data/nocropped_slitlamp/",
#                     help='path to dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='SimpleNet',
                    help='model architecture (default: SimpleNet)')

args = parser.parse_args()



class SimpleNetWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.5):
        super(SimpleNetWithDropout, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # 添加 Dropout 层
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        x=self.dropout(x)
        x=self.relu(x)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x


# 模拟一个简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
class SimpleNet_v2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet_v2, self).__init__()  # 修正此处的super调用
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 第一个额外的全连接层
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # 第二个额外的全连接层
        self.fc4 = nn.Linear(hidden_size, num_classes)  # 输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)  # 添加激活函数
        out = self.fc3(out)
        out = self.relu(out)  # 添加激活函数
        out = self.fc4(out)   # 最终输出层
        return out

# 读取单个txt文件的特征数据
def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        features = [list(map(float, line.strip().split(' '))) for line in lines]
        # features = [list(map(float, line.strip().split(',')[:-1])) for line in lines]
        return torch.tensor(features)
# 在 read_txt_file 函数中对特征进行缩放
def read_txt_file_agument(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        features = [list(map(float, line.strip().split(','))) for line in lines]

        # 对特征进行缩放（在0.9到1.1之间乘以随机比例系数）
        # for i in range(len(features)):
        #     # 选择要缩放的特征索引
        #     selected_indices = np.random.choice(len(features[i]), int(0.2 * len(features[i])), replace=False)
        #
        #     # 对所选的特征乘以随机比例系数
        #     for idx in selected_indices:
        #         scale_factor = np.random.uniform(0.7, 1.3)
        #         features[i][idx] *= scale_factor

        return torch.tensor(features)

# 其余代码保持不变

# 读取整个文件夹的特征数据
def read_folder(folder_path,agument=False):

    files = os.listdir(folder_path)
    data = []
    labels = []
    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        if agument:
            features = read_txt_file_agument(file_path)
            print('采用数据增强')
        else:
            features = read_txt_file(file_path)
        data.append(features)
        # 添加标签
        labels.extend([i] * len(features))

    return torch.cat(data), torch.tensor(labels, dtype=torch.long)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 这里可能需要调整
        return output


# 创建数据集和数据加载器
class MyDataset(data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

# 定义超参数和数据路径
num_classes = 3  # 三类图像
learning_rate = 0.001
num_epochs = 60
file_path = '/home/jiangjiewei/dingke_gpu/test_linear/features/feature_pca'

# 初始化模型、损失函数和优化器

if args.arch=='SimpleNet':
    input_size = 1024  # 输入特征维度
    hidden_size = 1024
    model = SimpleNet(input_size, hidden_size, num_classes)
elif args.arch == 'SimpleNet_v2':
    input_size = 1024  # 输入特征维度
    hidden_size = 1024
    model = SimpleNet_v2(input_size, hidden_size, num_classes)

elif args.arch == 'SimpleNetWithDropout':
    input_size = 1024  # 输入特征维度
    hidden_size = 2048
    model = SimpleNetWithDropout(input_size, hidden_size, num_classes)
elif args.arch == 'LSTM':
    input_size = 1024
    hidden_size = 2048
    num_classes = 3
    model = LSTM(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 定义保存模型权重的路径
weights_path = './models/model_完整图像'
def train():

    loss_file_path = os.path.join(weights_path, 'loss_record.txt')
    loss_file = open(loss_file_path, 'w')  # 创建文件用于保存损失记录
    # Creating data loaders for training and validation
    train_path = os.path.join(file_path, 'train')
    train_data, train_labels = read_folder(train_path,agument=False)
    train_dataset = MyDataset(train_data, train_labels)
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_path = os.path.join(file_path, 'val')
    val_data, val_labels = read_folder(val_path,agument=False)
    val_dataset = MyDataset(val_data, val_labels)
    val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    best_validation_loss = float('inf')
    best_weight_path = os.path.join(weights_path, 'best_weight.pth')

    for epoch in range(num_epochs):
        # Training the model
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, input_size)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%')

        # Validating the model
        model.eval()
        total_validation_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(-1, input_size)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                total_validation_loss += val_loss.item()

        avg_validation_loss = total_validation_loss / len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_validation_loss:.4f}')


        # Saving the model weights if validation loss decreases
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            torch.save(model.state_dict(), best_weight_path)

        # Saving the weights for each epoch
        epoch_weight_path = os.path.join(weights_path, f'densenet121_inception_v3_{epoch}.pth')
        torch.save(model.state_dict(), epoch_weight_path)
        loss_file.write(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}\n')

    loss_file.close()  # 关闭文件

def test():
    test_path=os.path.join(file_path,'val')
    #test_path = '/data/jiangjiewei/dk_test/features/DenseNe121_inception_v3/test'
    # 创建数据集和数据加载器
    all_data, all_labels = read_folder(test_path,agument=False)
    dataset = MyDataset(all_data, all_labels)
    test_loader = data.DataLoader(dataset, batch_size=64, shuffle=False)
    # 加载已训练好的模型权重
    weights_path = '/home/jiangjiewei/dingke_gpu/test_linear/models/model_完整图像/best_weight.pth'

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print(model)
        print("Model weights loaded successfully.")
    else:
        print("No model weights found.")
    # 测试模型
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(labels.numpy())


    # 生成混淆矩阵
    print(all_targets)
    print(all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(conf_matrix)
if __name__ == '__main__':
    #train()
    test()