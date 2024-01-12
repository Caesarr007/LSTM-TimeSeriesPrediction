import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from matplotlib.widgets import CheckButtons
import logging
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description='Script description')

# 添加参数
parser.add_argument('--draw', action='store_true', help='绘制图像')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

# 解析命令行参数
args = parser.parse_args()


# 读取Excel文件 并且忽略前五行 读取6402行
df = pd.read_excel('./Baltic Exchange Panamax Index(3).xlsx',skiprows=5,nrows=6402)

# 转换'Date'列为日期
df['Date'] = pd.to_datetime(df['Date'])
# print(df.head())
# print(df.tail())

# 设置时间为索引
df.set_index('Date', inplace=True)

if args.draw:
    # 1998-2023 绘制全部数据随时间变化的图像
    # 绘制图像
    plt.plot(df['Index'])
    plt.xlabel('Date')
    plt.ylabel('Index')

    # 保存图片
    # plt.savefig('./images/1998-2023.png')
    plt.title('1998-2023')
    plt.show()
    

    # 对每年的数据进行绘制
    # 获取年份的唯一值
    years = df.index.year.unique()

    # 对每个年份的数据绘制一个图像
    for year in years:
        df_year = df[df.index.year == year]
        plt.figure(figsize=(10, 6))
        plt.plot(df_year['Index'])
        plt.xlabel('Date')
        plt.ylabel('Index')
        plt.title(f'{year}')
        
        # 保存图像
        plt.savefig(f'./images/{year}.png')
        # plt.show()
        plt.close()
    

# 考虑经济危机，剔除2007-2008年的数据
df = df[(df.index.year < 2007) | (df.index.year > 2008)]

# 数据预处理
data = df['Index'].values
# print(data.shape)
data = data.reshape(-1, 1)
# print(data.shape)

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
# print(data[:5])
# print(data[-5:])

# 划分训练集和测试集
train_size = int(len(data) * 0.8)
train = data[:train_size]
test = data[train_size:]
# print(train.shape)
# print(test.shape)

# 转换数据集为LSTM的输入格式
def create_dataset(dataset, look_back=10):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        y.append(dataset[i+look_back])
    return torch.from_numpy(np.array(X)), torch.from_numpy(np.array(y))

look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# print(trainX.shape)
# reshape数据集为[样本数，时间步，特征数]的格式
trainX = trainX.view(trainX.shape[0], 1, -1)
testX = testX.view(testX.shape[0], 1, -1)
# print(trainX.shape)

# 创建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.linear(lstm_out[-1])
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(look_back, 32, 1).to(device)
trainX = trainX.float().to(device)
trainY = trainY.float().to(device)
testX = testX.float().to(device)
testY = testY.float().to(device)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# 配置日志记录器
logging.basicConfig(filename='lstm.log', level=logging.INFO)

if args.train:
    print('Start training...')
    # 训练模型
    for epoch in range(100):
        for i in range(len(trainX)):
            model.zero_grad()
            output = model(trainX[i].view(1, 1, -1))
            loss = loss_fn(output, trainY[i])
            # 打印信息并保存到日志文件
            if i % 100 == 0:
                logging.info('epoch:%s, batch:%s, loss:%s', epoch, i, loss.item())
                print('epoch:{},batch:{},loss:{}'.format(epoch, i, loss.item()))
            loss.backward()
            optimizer.step()
        # 每5个epoch保存一次模型
        if epoch % 5 == 0:
            torch.save(model.state_dict(), './checkpoints/lstm_{}.pth'.format(epoch))
else:
    # 加载模型
    model.load_state_dict(torch.load('./checkpoints/lstm_95.pth'))

# 使用模型进行预测
# print(trainX.shape)
trainPredict = []
testPredict = []
for i in range(len(trainX)):
    trainPredict.append(model(trainX[i]))
# print(trainPredict.__len__())
for i in range(len(testX)):
    testPredict.append(model(testX[i]))

# 从list转换为tensor
trainPredict = torch.tensor(trainPredict)
testPredict = torch.tensor(testPredict)
# 反归一化
trainPredict = scaler.inverse_transform(trainPredict.cpu().detach().numpy().reshape(-1, 1))
trainY = scaler.inverse_transform(trainY.cpu().detach().numpy())
testPredict = scaler.inverse_transform(testPredict.cpu().detach().numpy().reshape(-1, 1))
testY = scaler.inverse_transform(testY.cpu().detach().numpy())

# print(trainPredict[:5])
# print(trainY[:5])

# 绘制训练集预测结果和真实结果的对比曲线
plt.plot(trainPredict, label='Predict')
plt.plot(trainY, label='True')
plt.xlabel('Date')
plt.ylabel('Index')
plt.legend()
plt.title('Train Dataset Prediction vs True')
plt.savefig('./images/Train Dataset Prediction vs True.png')
plt.show()

# 绘制测试集预测结果和真实结果的对比曲线
plt.plot(testPredict, label='Predict')
plt.plot(testY, label='True')
plt.xlabel('Date')
plt.ylabel('Index')
plt.legend()
plt.title('Test Dataset Prediction vs True')
plt.savefig('./images/Test Dataset Prediction vs True.png')
plt.show()


