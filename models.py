from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

class FastTextModel(nn.Module):
    def __init__(self, embedding_file='embedding_SougouNews.npz', class_num=10):
        super(FastTextModel, self).__init__()
        self.embedding_size = 300
        embedding_pretrained = torch.tensor(
            np.load(embedding_file)["embeddings"].astype('float32'))
        # 定义词嵌入层
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        self.fc = nn.Linear(self.embedding_size, class_num)

    def forward(self, x):
        # text = [batch size,sent len]
        embedded = self.embedding(x).float()
        # embedded = [batch size, sent len, emb dim]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        logits = self.fc(pooled)
        # pooled = [batch size, embedding_dim]
        return logits

class TextCNNModel(nn.Module):  # 定义模型
    def __init__(self, embedding_file='embedding_SougouNews.npz'):
        super(TextCNNModel, self).__init__()
        # 加载词向量文件
        embedding_pretrained = torch.tensor(
            np.load(embedding_file)["embeddings"].astype('float32'))
        # 定义词嵌入层
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        # 定义三个卷积
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, 300)) for k in [2, 3, 4]])
        # 定义dropout层
        self.dropout = nn.Dropout(0.3)
        # 定义全连接层
        self.fc = nn.Linear(256 * 3, 10)

    def conv_and_pool(self, x, conv):  # 定义卷积+激活函数+池化层构成的一个操作块
        x = conv(x)  # N,1,32,300 -> N,256,31/30/29,1
        x = F.relu(x).squeeze(3)  # x -> N,256,31/30/29
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # x -> N,256,1 -> N,256
        return x

    def forward(self, x):  # 前向传播
        out = self.embedding(x)  # N,32 -> N,32,300
        out = out.unsqueeze(1)  # out -> N,1,32,300
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # out ->N,768
        out = self.dropout(out)
        out = self.fc(out)  # N,768 -> N,10
        return out
    

class RNNModel(nn.Module):
    def __init__(self, embedding_file='embedding_SougouNews.npz', \
                 rnn_type="lstm", hidden_dim=256, class_num=10, n_layers=2, bidirectional=True, dropout=0.3, batch_first=True):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.embedding_size = 300
        embedding_pretrained = torch.tensor(
            np.load(embedding_file)["embeddings"].astype('float32'))
        # 定义词嵌入层
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        """
            输出序列和最后一个时间步的隐藏状态
        if batch_first:
            output, hidden = [bitch_size, max_seq, hidden_size * bidirectional] \ 
            [num_layers * bidirectional, batch_size, hidden_size]
        else:
            output, hidden = [bitch_size, max_seq, hidden_size * bidirectional] \ 
            [num_layers * bidirectional, max_seq, hidden_size]
        """
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
            
        self.dropout = nn.Dropout(dropout)
        if self.bidirectional:
            self.fc = nn.Linear(self.hidden_dim * 2, class_num)
        else:
            self.fc = nn.Linear(self.hidden_dim, class_num)

    def forward(self, x):
        x = self.embedding(x)
        self.rnn.flatten_parameters() # 扁平化
        if self.rnn_type in ['rnn', 'gru']:
            output, hidden = self.rnn(x)
        else:
            output, (hidden, cell) = self.rnn(x)
        x = output[:, -1, :]
        x = self.dropout(x)
        logits = self.fc(x)

        return logits

"""
PLM models
"""
class ModelForSequenceClassification(nn.Module):
    def __init__(self, model_Name, num_classes):
        super(ModelForSequenceClassification, self).__init__()
        self.model_Name = model_Name
        self.model = AutoModel.from_pretrained(model_Name)
        if model_Name != "hfl/chinese-xlnet-base":
            self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
            self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        else:
            self.dropout = nn.Dropout(self.model.config.summary_last_dropout)
            self.classifier = nn.Linear(self.model.config.d_model, num_classes)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):

        outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        
        if self.model_Name == "hfl/chinese-xlnet-base":
            pooled_output = torch.sum(outputs[0], dim=1)
        else:
            pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
    
class ModelTextCNNForSequenceClassification(nn.Module):
    def __init__(self, model_Name, num_classes):
        super(ModelTextCNNForSequenceClassification, self).__init__()
        self.model_Name = model_Name
        self.num_classes = num_classes
        self.model = AutoModel.from_pretrained(model_Name)

        if model_Name != "hfl/chinese-xlnet-base":
            self.convs = nn.ModuleList(
                [nn.Conv2d(1, 256, (k, self.model.config.hidden_size)) for k in [2, 3, 4]])
        else:
            self.convs = nn.ModuleList(
                [nn.Conv2d(1, 256, (k, self.model.config.d_model)) for k in [2, 3, 4]])
        # 定义dropout层
        self.dropout_cnn = nn.Dropout(0.3)
        # 定义全连接层
        self.classifier = nn.Linear(256 * 3, self.num_classes)

    def conv_and_pool(self, x, conv):  # 定义卷积+激活函数+池化层构成的一个操作块
        x = conv(x)  # N,1,32,300 -> N,256,31/30/29,1
        x = F.relu(x).squeeze(3)  # x -> N,256,31/30/29
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # x -> N,256,1 -> N,256
        return x
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):

        outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        
        seq_output = outputs[0] # [batch_size, seq_len, hidden_size]
        out = seq_output.unsqueeze(1) # [batch_size, 1, seq_len, hidden_size]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout_cnn(out)
        logits = self.classifier(out)    
        return logits
    
class ModelRNNForSequenceClassification(nn.Module):
    def __init__(self, model_Name, num_classes):
        super(ModelRNNForSequenceClassification, self).__init__()
        self.model_Name = model_Name
        self.num_classes = num_classes
        self.rnn_type = "lstm"
        self.hidden_dim = 256
        self.n_layers = 2
        self.droprate = 0.3
        self.bidirectional = True
        self.batch_first = True

        self.model = AutoModel.from_pretrained(model_Name)
        if model_Name != "hfl/chinese-xlnet-base":
            self.embedding_size = self.model.config.hidden_size
        else:
            self.embedding_size = self.model.config.d_model

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               self.hidden_dim,
                               num_layers=self.n_layers,
                               bidirectional=self.bidirectional,
                               batch_first=self.batch_first,
                               dropout=self.droprate)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=self.hidden_dim,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.droprate)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=self.hidden_dim,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.droprate)
            
        self.dropout= nn.Dropout(p=0.3)

        if self.bidirectional:
            self.classifier = nn.Linear(self.hidden_dim * 2, num_classes)
        else:
            self.classifier = nn.Linear(self.hidden_dim, num_classes)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):

        outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        seq_output = outputs[0]
        self.rnn.flatten_parameters() # 扁平化
        if self.rnn_type in ['rnn', 'gru']:
            output, hidden = self.rnn(seq_output)
        else:
            output, (hidden, cell) = self.rnn(seq_output)
        x = output[:, -1, :]
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits
    

class ModelRCNNForSequenceClassification(nn.Module):
    def __init__(self, model_Name, num_classes):
        super(ModelRCNNForSequenceClassification, self).__init__()
        self.model_Name = model_Name
        self.num_classes = num_classes
        self.rnn_type = "lstm"
        self.hidden_dim = 256
        self.n_layers = 2
        self.droprate = 0.3
        self.bidirectional = True
        self.batch_first = True

        self.model = AutoModel.from_pretrained(model_Name)
        if model_Name != "hfl/chinese-xlnet-base":
            self.embedding_size = self.model.config.hidden_size
        else:
            self.embedding_size = self.model.config.d_model

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               self.hidden_dim,
                               num_layers=self.n_layers,
                               bidirectional=self.bidirectional,
                               batch_first=self.batch_first,
                               dropout=self.droprate)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=self.hidden_dim,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.droprate)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=self.hidden_dim,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.droprate)
            
        self.dropout= nn.Dropout(p=0.3)
        self.maxpool = nn.MaxPool1d(32)
        self.ReLU = nn.ReLU()

        if self.bidirectional:
            self.classifier = nn.Linear(self.hidden_dim * 2 + self.embedding_size, num_classes)
        else:
            self.classifier = nn.Linear(self.hidden_dim + self.embedding_size, num_classes)

    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):

        outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        seq_output = outputs[0]
        self.rnn.flatten_parameters() # 扁平化
        if self.rnn_type in ['rnn', 'gru']:
            output, hidden = self.rnn(seq_output)
        else:
            output, (hidden, cell) = self.rnn(seq_output)
        x = torch.cat([seq_output, output], dim=2) # 连接Bertmodel 和 rnnmodel 的输出
        x = self.ReLU(x) # 非线性变化
        x = x.permute(0, 2, 1) # [batch_size, embedding_dim, max_seq]
        x = self.maxpool(x).squeeze(2) # [batch_size, embedding_dim]
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits