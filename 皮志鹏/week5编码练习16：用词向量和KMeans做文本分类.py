# 我要用gensim.models.Word2Vec训练词向量，我有一些句子和训练好的词向量，再用sklearn.cluster.Kmeans对句子分类
'''
加载词向量
用词向量把所有句子转成句子向量
调用Kmeans把句子分成若干类
对每类句子求中心点平均距离，输出平均距离最短的十个类别，每个类别打印十个句子
'''
from gensim.models import Word2Vec
import jieba
from sklearn.cluster import KMeans
import math
import numpy as np
model = Word2Vec.load('model.w2v')

sentences = []
with open(r"E:\BaiduNetdiskDownload\week5 词向量和文本向量\titles.txt",encoding='utf-8') as f:
    for line in f:
        sentence = " ".join(jieba.lcut(line))
        sentences.append(sentence)

vectors = []
for sentence in sentences:
    vector = np.zeros(model.vector_size)
    words = sentence.split()
    for word in words:
        try:
            vector += model.wv[word]
        except KeyError:
            vector += np.zeros(model.vector_size)
    vectors.append(vector/len(words))

n_clusters = int(math.sqrt(len(sentences)))
kmeans = KMeans(n_clusters)
kmeans.fit(vectors)

# 已有数据：语句数组，语句向量的数组，（分组完成后）每条语句的组号的数组，每个组号的中心点坐标的数组
# 求数据：所有分组中，组内平均距离最小的前十个组，每个组打印十条语句


def calc_distance(v1,v2):
    return np.sqrt(sum(np.square(v1-v2)))
#向量余弦距离
def cosine_distance(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.sum(np.square(vec1)))  #A/|A|
    vec2 = vec2 / np.sqrt(np.sum(np.square(vec2)))  #B/|B|
    return np.sum(vec1 * vec2)

labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

from collections import defaultdict
label_vectors_dict = defaultdict(list)
for index in range(len(sentences)):
    label_vectors_dict[labels[index]].append(vectors[index])

label_avgdistance =[]
for label,label_vectors in label_vectors_dict.items():
    distance_total = 0
    center_vector = cluster_centers[label]
    for vector in label_vectors:
        distance_total += cosine_distance(center_vector,vector)
    avg_distance = distance_total/len(label_vectors)
    label_avgdistance.append([label,avg_distance])

good_classes = sorted(label_avgdistance,key=lambda x:x[1])[:10]

label_sentences_dict = defaultdict(list)
for index in range(len(sentences)):
    label_sentences_dict[labels[index]].append(sentences[index])

for label,avg_distance in good_classes:
    print("-------------")
    print(f"cluster{label}，组内平均距离 {avg_distance}")
    sentences = label_sentences_dict[label][:10]
    for i in range(10):
        print(sentences[i].replace(" ","").strip())

