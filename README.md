# Keras-BertFc
将bert预训练模型（chinese_L-12_H-768_A-12）放到当前目录下
基于bert句向量的文本分类：基于Dense的微调
输入：
1. uncased_L-12_H-768_A-12 bert预训练好的模型
2. wos_labelcontent_label_content2 格式：[label1,label2,label3,....][content1,content2,content3....][label1+'\t'+content1,label2+'\t'+content2,.....]
3. _, wosy2_to_id,_, _ = pl.load(open(r'D:\01zjp\代码\keras_bert_classification\data\label2idclean', 'rb'))  其中wosy2_to_id将label转变为id

输出：
1.model:bertweights_fc.h5

数据预处理：
 wos_labelcontent_label_content2 由来方式：源数据：parent1，parent2,leaf_label,content 处理成：[label1,label2,label3,....][content1,content2,content3....][label1+'\t'+content1,label2+'\t'+content2,.....]
#######################################################
import csv
import pickle as pl
infile = r"D:\01zjp\代码\keras_bert_classification\dataprocess20210106\data\train.csv"
label3s = []
contents = []
label3_contents = []
with open(infile, 'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    # count = 0
    for row in reader:
        # count = count+1
        # if count>20 :
        #     break
        label3s.append(row[2])
        contents.append(row[3])
        label3_cont = row[2]+'\t'+row[3]
        label3_contents.append(label3_cont)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
count = 0
for line in contents:
    count=count+1
    print(count)

with open(r'D:\01zjp\代码\keras_bert_classification\dataprocess20210106\data\label3_content_labelContent20210106', 'wb') as f:
    pl.dump((label3s,contents,label3_contents), f)


######################################################################################################
