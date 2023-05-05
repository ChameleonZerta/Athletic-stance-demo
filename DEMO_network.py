import scipy.io as scio
import numpy as np
from sklearn.model_selection import *
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

feature = scio.loadmat('./Data/BufferFeatures.mat')
acc = scio.loadmat('./Data/BufferedAccelerations.mat')
feat = feature['feat']
actnames = acc['actnames']
actid = acc['actid']
atx = acc['atx']
aty = acc['aty']
atz = acc['atz']

actid_label = actid.reshape(1, -1)[0]-1

feature_size = np.shape(feat)[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
feat_minmax = scaler_minmax.fit_transform(feat)

feat_train, feat_test, actid_train, actid_test = \
    train_test_split(feat_minmax, actid_label, test_size=0.5)
feat_train = torch.from_numpy(feat_train).to(torch.float).to(device)
feat_test = torch.from_numpy(feat_test).to(torch.float).to(device)
actid_train = torch.from_numpy(actid_train).to(torch.int64).to(device)
actid_test = torch.from_numpy(actid_test).to(torch.int64).to(device)

batch_size = 30
train_dataset = TensorDataset(feat_train, actid_train)
test_dataset = TensorDataset(feat_test, actid_test)
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 模型构建
num_inputs, num_outputs, num_hiddens= feature_size, 6, 100
net = nn.Sequential(
    nn.Linear(num_inputs, num_hiddens),
    # nn.Sigmoid(),
    # nn.Linear(num_hiddens, num_hiddens),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(num_hiddens, num_outputs)
    # torch交叉熵的损失函数自带softmax运算，这里就没有再加一层激活函数
    ).to(device)

# 参数初始化
for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.005)


def acc(test_iter, net, device):
    acc_sum, n = 0.0, 0
    for x, y in test_iter:
        acc_sum += (net(x.to(device)).argmax(dim=1) == y.to(device)).sum().item()
        n += y.shape[0]
    return acc_sum / n


K = acc(test_iter, net, device)
print(K)

# 交叉熵损失函数
loss = torch.nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(params=net.parameters(), lr=0.02, weight_decay=0.0001)

train_allacc = []
test_allacc = []
loss_all = []
num_epochs = 250
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for x, y in train_iter:
        y_hat = net(x)
        lo = loss(y_hat, y).sum()
        optimizer.zero_grad()  # 优化器的梯度清零
        lo.backward()  # 反向传播梯度计算
        optimizer.step()  # 优化器迭代
        train_l_sum += lo.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    test_acc = acc(test_iter, net, device)
    train_allacc.append(train_acc_sum / n)
    test_allacc.append(test_acc)
    loss_all.append(batch_size * train_l_sum / n)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
          % (epoch + 1, batch_size * train_l_sum / n, train_acc_sum / n, test_acc))
plt.plot(loss_all, label='loss')
plt.plot(train_allacc, label='train')
plt.plot(test_allacc, label='test')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.show()
