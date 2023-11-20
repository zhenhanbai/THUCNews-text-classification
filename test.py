import torch
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader
from data_processing import MyData, tokenize_textCNN
from transformers import AutoTokenizer

labels = ['体育','娱乐','家居','教育','时政','游戏','社会','科技','股票','财经']

def evaluate(model, device, dataload):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    failed_predictions = []  # 存储预测失败的数据

    with torch.no_grad():
        for texts, labels in dataload:
            outputs = model(texts.to(device))
            loss = F.cross_entropy(outputs, labels.to(device))
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

            # 检查预测是否正确，如果预测错误则将数据存储到 failed_predictions 中
            for i in range(len(predic)):
                if predic[i] != labels[i]:
                    failed_predictions.append((texts[i], labels[i], predic[i]))

    acc = metrics.accuracy_score(labels_all, predict_all)
    model.train()
    return acc, loss_total / len(dataload), (labels_all, predict_all), failed_predictions


def gotestModel(model_file, test_file):
    # 得到数据集
    test_dataset = MyData(tokenize_fun=tokenize_textCNN, filename=test_file)
    batch_size = 128
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,  # 从数据集合中每次抽出batch_size个样本
        shuffle=False,  # 加载数据时不打乱样本顺序
    )
    model = torch.load(model_file, map_location=lambda s, l: s )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设
    model.to(device).eval()
    acc, loss, allInfo = evaluate(model, device, test_dataloader)
    report = metrics.classification_report(allInfo[0], allInfo[1], target_names=labels, digits=4)
    confusion = metrics.confusion_matrix(allInfo[0], allInfo[1])
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(loss, acc))
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)


def testModel(model_file, test_file):
    MODEL_NAME = 'nghuyong/ernie-3.0-base-zh'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # 实例化分词器
    # 定义分词函数
    def tokenize_BERT(s):
        return tokenizer.encode(s, max_length=32, truncation=True, padding="max_length")

    # 得到数据集
    test_dataset = MyData(tokenize_fun=tokenize_BERT, filename=test_file)
    batch_size = 128
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,  # 从数据集合中每次抽出batch_size个样本
        shuffle=False,  # 加载数据时不打乱样本顺序
    )
    model = torch.load(model_file, map_location=lambda s,l:s) # 不改变原始位置, 即还是与原来一样的位置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    model.to(device).eval()

    acc, loss, allInfo, failed_predictions = evaluate(model, device, test_dataloader)

    with open("failed_predictions.txt", mode='w', encoding='utf-8') as f:
        f.write("data\ttrue_label\tpre_label\n")
        for item in failed_predictions:
            data, label, predic = tokenizer.decode(item[0], skip_special_tokens=True), labels[item[1]], labels[item[2]]
            data = data.replace(" ", "")
            line = f"{data}\t{label}\t{predic}\n" 
            f.write(line)
    f.close()

    report = metrics.classification_report(allInfo[0], allInfo[1], target_names=labels, digits=4)
    confusion = metrics.confusion_matrix(allInfo[0], allInfo[1])
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(loss, acc))
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)

if __name__ == '__main__':
    testModel(model_file="checkpoint/ModelRCNNForSequenceClassification/ModelRCNNForSequenceClassification_best_model", test_file="data/test.txt")