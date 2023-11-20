import os
import torch
from tqdm import tqdm
import pickle as pkl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setSeed(seed=42)

labels = ['体育','娱乐','家居','教育','时政','游戏','社会','科技','股票','财经']
LABEL2ID = { x:i for (x,i) in zip(labels,range(len(labels)))}

vocab_file = "./vocab.pkl"
word_to_id = pkl.load( open(vocab_file, 'rb'))  #加载词典


def tokenize_textCNN(s):  # 输入句子s
    max_size = 32  # 句子分词最大长度
    ts = [w for i, w in enumerate(s) if i < max_size]  # 得到字符列表，最多32个
    ids = [word_to_id[w] if w in word_to_id.keys() else word_to_id['[UNK]'] for w in ts]  # 根据词典，将字符列表转换为id列表
    ids += [0 for _ in range(max_size-len(ts))]  # 若id列表达不到最大长度，则补0
    return ids

class MyData(Dataset):  # 继承Dataset
    def __init__(self, tokenize_fun, filename):
        self.filename = filename  # 要加载的数据文件名
        self.tokenize_function = tokenize_fun  # 实例化时需传入分词器函数
        print("Loading dataset "+ self.filename +" ...")
        self.data, self.labels = self.load_data()  # 得到分词后的id序列和标签
    #读取文件，得到分词后的id序列和标签，返回的都是tensor类型的数据
    def load_data(self):
        labels = []
        data = []
        with open(self.filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc='Loading data', colour="green"):
                fields  = line.strip().split('\t')
                if len(fields) != 2 :
                    continue
                labels.append(LABEL2ID[fields[0]])  #标签转换为序号
                data.append(self.tokenize_function(fields[1]))  # 样本为词id序列
        f.close()
        return torch.tensor(data), torch.tensor(labels)
    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)
    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        return self.data[index], self.labels[index]

def getDataLoader(train_dataset, dev_dataset):
    batch_size = 128
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,  # 从数据集合中每次抽出batch_size个样本
        shuffle=True,  # 加载数据时打乱样本顺序
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        shuffle=False,  # 按原始数据集样本顺序加载
    )
    return train_dataloader, dev_dataloader