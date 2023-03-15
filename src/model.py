#首先，需要准备无错别字的新闻数据集和基于无错别字数据集随机生成的错别字数据集。
#可以使用 Python 中的 Pandas 库来读取数据集，
#然后使用 Scikit-learn 库的 train_test_split 函数将数据集分成训练集和测试集。
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取无错别字的新闻数据集
news_data = pd.read_csv('news_data.csv', encoding='utf-8')

# 读取错别字数据集
typo_data = pd.read_csv('typo_data.csv', encoding='utf-8')

# 合并数据集
data = pd.concat([news_data, typo_data], ignore_index=True)

# 划分训练集和测试集
train_data, test_data, train_label, test_label = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)


#接下来需要对数据进行预处理，包括中文分词、建立词典和将文本转换为数字序列。
# 这里使用了 Python 中的 Jieba 库进行中文分词，
# 并使用 Keras 库的 Tokenizer 类将文本转换为数字序列。
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 中文分词
train_data = train_data.apply(lambda x: ' '.join(jieba.cut(x)))
test_data = test_data.apply(lambda x: ' '.join(jieba.cut(x)))

# 建立词典
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data)

# 将文本转换为数字序列
train_seq = tokenizer.texts_to_sequences(train_data)
test_seq = tokenizer.texts_to_sequences(test_data)

# 对数字序列进行填充
max_len = 100
train_seq_pad = pad_sequences(train_seq, maxlen=max_len, padding='post')
test_seq_pad = pad_sequences(test_seq, maxlen=max_len, padding='post')

#搭建 LSTM 模型
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# 搭建 LSTM 模型
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 128, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#接下来，需要训练模型。这里使用了 Keras 库的 fit 函数进行模型训练，并使用验证集进行模型调优。
# 训练模型
history = model.fit(train_seq_pad, train_label, batch_size=128, epochs=10, validation_split=0.2)

# 在测试集上评估模型性能
test_loss, test_acc = model.evaluate(test_seq_pad, test_label, batch_size=128)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

#预测并输出错别字
#接下来，使用训练好的模型对一段文本进行预测，并输出其中的错别字。
import numpy as np

# 预测一段文本的标签
text = '小明在公园散步。'
text_seq = tokenizer.texts_to_sequences([text])
text_seq_pad = pad_sequences(text_seq, maxlen=max_len, padding='post')
pred = model.predict(np.array(text_seq_pad))

# 输出其中的错别字
if pred[0] > 0.5:
    print('这段文本中可能有错别字。')
    text = list(text)
    for i, c in enumerate(text):
        text_seq = tokenizer.texts_to_sequences([''.join(text[:i]+text[i+1:])])
        text_seq_pad = pad_sequences(text_seq, maxlen=max_len, padding='post')
        pred = model.predict(np.array(text_seq_pad))
        if pred[0] < 0.5:
            print('第', i+1, '个字', c, '可能是错别字。')
else:
    print('这段文本中没有错别字。')
