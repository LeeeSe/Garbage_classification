import os
from pprint import pprint
from random import random, shuffle

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# 定义transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 定义垃圾分类数据集
class Garbage_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.data = []
        self.label = []
        self.read_data()
        # pprint(self.data[:10])
        # pprint(self.label[:10])

    # 从txt中获取图像名称和标签
    # 将data与label同步打乱
    def read_data(self):
        # 获取所有txt文件
        file_list = os.listdir(os.path.join(self.root))
        txt_list = [txt for txt in file_list if txt.endswith('.txt')]
        for txt in txt_list:
            # 获取txt文件中的图像名称和标签
            with open(os.path.join(self.root, txt), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(',')
                    self.data.append(os.path.join(self.root, line[0]))
                    self.label.append(int(line[1]))
        # 将data与label同步打乱
        data_label = list(zip(self.data, self.label))
        shuffle(data_label)
        self.data, self.label = zip(*data_label)

        # 根据是否是训练集，将data与label分为训练集和验证集
        if self.train:
            self.data = self.data[:int(len(self.data) * 0.8)]
            self.label = self.label[:int(len(self.label) * 0.8)]
        else:
            self.data = self.data[int(len(self.data) * 0.8):]
            self.label = self.label[int(len(self.label) * 0.8):]

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape, self.label[index])
        return img, self.label[index]

    def __len__(self):
        return len(self.data)


# 定义数据集并split为训练集和测试集
train_dataset = Garbage_Dataset(root='datasets/garbage_classify/train_data', transform=transform, train=True)
test_dataset = Garbage_Dataset(root='datasets/garbage_classify/train_data', train=False)

# 定义训练集和测试集的加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

# 展示训练集与测试集数量
print('train_dataset:', len(train_dataset))
print('test_dataset:', len(test_dataset))

# 展示训练集中的第一张图和标签
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(f'example data: {images.shape}')
print(f'example label: {labels}')

# 定义网络,并修改为num_classes=40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b0(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 40)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器
criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数


# 定义训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# 定义测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # loss 加和
            pred = output.argmax(dim=1, keepdim=True)  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 边训练边测试
for epoch in range(1, 10):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)

