import os
import torch
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
from test import evaluate
from models import (
    ModelTextCNNForSequenceClassification,
    ModelRNNForSequenceClassification,
    ModelRCNNForSequenceClassification,
    ModelForSequenceClassification, 
    TextCNNModel, 
    FastTextModel,
    RNNModel)

from transformers import AutoTokenizer
from data_processing import MyData, getDataLoader, tokenize_textCNN


def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setSeed(seed=42)

# tensorboard --logdir=training/

from tensorboardX import SummaryWriter
import time
from datetime import timedelta

def get_time_dif(start_time):  # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(model, device, model_name, lr, train_dataloader, dev_dataloader):
    start_time = time.time()  # 记录起始时间
    model.train()  # 设置model为训练模式
    optimizer = torch.optim.Adam(model.parameters(), lr=lr )  # 定义优化器
    total_batch = 0  # 记录进行到多少batch
    dev_best_acc = float('-inf')  # 记录验证集上的最好损失
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    if not os.path.exists('./log'):
        os.makedirs('./log', exist_ok=True)
    if not os.path.exists(f'./checkpoint/{model_name}'):
        os.makedirs(f'./checkpoint/{model_name}', exist_ok=True)
    writer = SummaryWriter(log_dir='./log/%s.'%(model_name) + time.strftime('%m-%d_%H.%M', time.localtime()))  # 实例化SummaryWriter
    num_epochs = 5  # 设置训练次数
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (trains, labels) in enumerate(train_dataloader):
            outputs = model(trains.to(device))  # 
            model.zero_grad()  # 模型梯度清零
            loss = F.cross_entropy(outputs, labels.to(device))  # 计算交叉熵损失
            loss.backward()  # 梯度回传
            optimizer.step()  # 更新参数
            if total_batch % 100  == 0:
                # 每100轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)  # 训练集精确度
                # 调用函数得到测试集精确度
                dev_acc, dev_loss, _, _ = evaluate(model, device, dev_dataloader)
                # 记录验证集当前的最优损失，并保存模型参数
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    checkpoint_path = f"./checkpoint/{model_name}/{model_name}_best_model"
                    torch.save(model, checkpoint_path)
                    improve = '*'  # 设置标记
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)  # 得到当前运行时间
                msg = 'Iter:{0:>4}, Tr-Loss:{1:>6.4}, Tr-Acc:{2:>6.2%}, Va-Loss:{3:>6.4}, Va-Acc:{4:>6.2%}, Time:{5}{6}'
                # 打印训练过程信息
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 写入tensorboardX可视化用的日志信息
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
            total_batch += 1
            if total_batch - last_improve > 1000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()  # 关闭writer对象

def trainDNN():
    MODEL_LIST = [FastTextModel, TextCNNModel, RNNModel]
    for MODEL_NAME in MODEL_LIST:
        print("----------------------\n", MODEL_NAME.__name__)
        train_dataset = MyData(tokenize_fun=tokenize_textCNN, filename='data/train.txt')
        dev_dataset = MyData(tokenize_fun=tokenize_textCNN, filename='data/dev.txt')
        train_dataset, dev_dataset = getDataLoader(train_dataset, dev_dataset)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        model = MODEL_NAME().to(device)
        lr = 1e-3  # Adam优化器学习率
        train(model, device, MODEL_NAME.__name__, lr, train_dataset, dev_dataset)  # 开始


def trainBert():
    MODEL_LIST = ["bert-base-chinese", "hfl/rbt3", "hfl/chinese-xlnet-base", "nghuyong/ernie-3.0-base-zh"]
    for MODEL_NAME in MODEL_LIST:
        print("----------------------\n", MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # 实例化分词器
        # 定义分词函数
        def tokenize_BERT(s):
            return tokenizer.encode(s, max_length=32, truncation=True, padding="max_length")
        # 得到数据集
        train_dataset = MyData(tokenize_fun=tokenize_BERT, filename='data/train.txt')
        dev_dataset = MyData(tokenize_fun=tokenize_BERT, filename='data/dev.txt')
        # 得到数据加载器
        train_dataset, dev_dataset = getDataLoader(train_dataset, dev_dataset)
        # 定义模型
        model = ModelForSequenceClassification(MODEL_NAME, num_classes=10)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        model = model.to(device)
        lr = 1e-4  # 设置Adam优化器学习率
        model_name = MODEL_NAME.split('/')[1] if len(MODEL_NAME.split('/')) > 1 else MODEL_NAME.split('/')[0]
        train(model, device, f"{model_name}", lr, train_dataset, dev_dataset)


def trainBertDNN():
    MODEL_LIST = [ModelTextCNNForSequenceClassification, ModelRNNForSequenceClassification, ModelRCNNForSequenceClassification]
    PLM = "nghuyong/ernie-3.0-base-zh"
    for MODEL_NAME in MODEL_LIST:
        print("----------------------\n", MODEL_NAME.__name__)
        tokenizer = AutoTokenizer.from_pretrained(PLM)  # 实例化分词器
        # 得到数据集
        def tokenize_BERT(s):
            return tokenizer.encode(s, max_length=32, truncation=True, padding="max_length")
        train_dataset = MyData(tokenize_fun=tokenize_BERT, filename='data/train.txt')
        dev_dataset = MyData(tokenize_fun=tokenize_BERT, filename='data/dev.txt')
        # 得到数据加载器
        train_dataset, dev_dataset = getDataLoader(train_dataset, dev_dataset)
        # 定义模型
        model = MODEL_NAME(PLM, num_classes=10)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        model = model.to(device)
        lr = 1e-4  # 设置Adam优化器学习率
        train(model, device, MODEL_NAME.__name__, lr, train_dataset, dev_dataset)


if __name__ == '__main__':
    trainDNN()
    trainBert()
    trainBertDNN()