import numpy as np
import pickle as pkl

max_vocab_size = 10000
train_file = "embedding_data.txt"
orig_embedding_file  = "../glove.6B.300d.txt"
vocab_file = "vocab.pkl"
filename_trimmed_file = "embedding_zh" 

emb_dim = 300  # 嵌入向量长度
min_freq = 1  # 最小出现频率，小于该值的字将被忽略
np.random.seed(42)  # 设置随机数种子

word_count = {}  # 存储每个词的数量
with open(train_file, 'r', encoding='UTF-8') as f:
    for line in f:
        lin = line.strip()
        if not lin: continue
        for word in lin:  # 对于新闻标题，以字为单位构建词表
            word_count[word] = word_count.get(word, 0) + 1
    # 得到词表，按词频有序，最多max_vocab_size个
    vocab_list = sorted([_ for _ in word_count.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_vocab_size]
    # 对词从0开始编号
    word_to_id = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    # 添加特殊的Token
    word_to_id.update({'[UNK]': len(word_to_id), '[PAD]': len(word_to_id) + 1})
    # 存储词及其id到文件
    pkl.dump(word_to_id, open(vocab_file, 'wb'))
f.close()

embeddings = np.random.rand(len(word_to_id), emb_dim)  # 随机初始化每个词id对应的嵌入向量
with open(orig_embedding_file, "r", encoding='UTF-8') as f:  # 读取预训练词向量文件
    for i, line in enumerate(f.readlines()):
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:  # 如果字在word_to_id列表中
            idx = word_to_id[lin[0]]  # 得到该字的id
            emb = [float(x) for x in lin[1:emb_dim+1]]  # 得到嵌入向量
            embeddings[idx] = np.asarray(emb, dtype='float32')  # 替换初始化的嵌入向量
f.close()
np.savez_compressed(filename_trimmed_file, embeddings=embeddings)  # 保存训练文件中各词的嵌入向量，在训练时直接读取，以加快处理速度
