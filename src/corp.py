import random

def reader():
    ...


# 生成错别字
def generate_errors(word):
    # 错误概率
    error_probability = 0.2
    # 错误类型及对应的概率
    error_types = [("replace", 0.4), ("insert", 0.3), ("delete", 0.2), ("transpose", 0.1)]
    # 随机选择错误类型
    error_type = random.choices([e[0] for e in error_types], [e[1] for e in error_types])[0]
    # 替换错误
    if error_type == "replace":
        # 随机选择一个字符进行替换
        index = random.randint(0, len(word) - 1)
        return word[:index] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[index + 1:]
    # 插入错误
    elif error_type == "insert":
        # 随机选择一个位置插入一个字符
        index = random.randint(0, len(word))
        return word[:index] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[index:]
    # 删除错误
    elif error_type == "delete":
        # 随机选择一个字符删除
        index = random.randint(0, len(word) - 1)
        return word[:index] + word[index + 1:]
    # 交换错误
    elif error_type == "transpose":
        # 随机选择两个相邻字符交换
        index = random.randint(0, len(word) - 2)
        return word[:index] + word[index + 1] + word[index] + word[index + 2:]
    else:
        return word

# 生成错别字训练集
def generate_error_corpus(sentence):
    # 训练集大小
    corpus_size = 10000
    # 生成训练集
    corpus = []
    for i in range(corpus_size):
        # 将句子分成单词
        words = sentence.split(" ")
        # 随机选择一个单词生成错别字
        index = random.randint(0, len(words) - 1)
        words[index] = generate_errors(words[index])
        # 将句子重新拼接起来
        new_sentence = " ".join(words)
        # 添加到训练集中
        corpus.append(new_sentence)
    return corpus

