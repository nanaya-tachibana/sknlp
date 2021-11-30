![CI badge](https://img.shields.io/github/workflow/status/nanaya-tachibana/sknlp/CI)
![coverage badge](https://img.shields.io/codecov/c/github/nanaya-tachibana/sknlp)

*sknlp*是一个使用 Tensorflow 实现的仿[scikit-learn](https://scikit-learn.org/stable/)接口设计的 NLP 深度学习模型库, 训练的模型可直接使用 tensorflow serving 部署.

写这个库的原因, 一是搭建一个模型实现的基础模版, 方便业余时间实验新新技术;
二是简化一些常用模型的训练部署流程, 方便工作中快速上线原型服务, 有更多时间去关心业务和数据.

# 安装

```shell
pip install sknlp
```

# 功能

1. 分类, 单标签和多标签
   - BertClassifier, Bert 微调
   - RNNClassifier, bilstm + attention
   - RCNNClassifier, [Recurrent Convolutional Neural Network for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745)
   - CNNClassifier, 膨胀卷积
2. 序列标注, crf 解码和[global pointer](https://spaces.ac.cn/archives/8373)解码
   - BertTagger, Bert 微调
   - BertRNNTagger, Bert + bilstm
   - RNNTagger, bilstm
   - CNNTagger, 膨胀卷积
3. 语义向量检索, [SimSCE](https://arxiv.org/abs/2104.08821), 无监督和有监督
   - BertRetriever, Bert 微调
   - RNNRetriever, bilstm
4. 文本生成
   - BertGenerator, [unilm](https://arxiv.org/abs/1905.03197)

# 使用

下面是一个使用 BertClassifier 构建分类器, 和 scikit-learn 一样直接通过 X 和 y 提供数据训练和预测的示例

```python
from sknlp.module.classifiers import BertClassifier
from sknlp.moduel.text2vec import Bert2vec, BertFamily

b2v = Bert2vec.from_tfv1_checkpoint(BertFamily.BERT, "RoBERTa-tiny3L768-clue")
clf = BertClassifier(["letter", "digit"], is_multilabel=False, text2vec=b2v)
clf.fit(
    X=["aa", "bb", "cc", "dd", "11", "22", "33", "44"],
    y=["letter", "letter", "letter", "letter"],
    n_epochs=5,
    learning_rate=1e-4
)
clf.predict(X=["aa", "88"])
clf.score(X=["xx", "11"], y=["letter", "digit"])
```

也可以通过特定格式的文件构建 dataset 来进行训练和预测

```python
from sknlp.module.classifiers import RNNClassifier
from sknlp.moduel.text2vec import Word2vec

t2v = Word2vec.from_word2vec_format("data/jieba/vec.txt", segmenter="jieba")
clf = RNNClassifier(["letter", "digit"], is_multilabel=False, text2vec=t2v)
training_dataset = clf.create_dataset_from_csv("data/train.txt")
clf.fit(
    dataset=training_dataset,
    n_epochs=5,
    learning_rate=1e-4
)
test_dataset = clf.create_dataset_from_csv("data/test.txt", evaluation=True)
clf.predict(dataset=test_dataset)
clf.score(dataset=test_dataset)
```

sknlp.module 中所有模型都需要一个 text2vec(即预训练模型)来初始化, 初始化后均提供了 fit, predict 和 score 等统一功能的方法.

下面两小节详细介绍, 不同任务的数据格式要求和现在支持预训练模型格式.

## 数据格式

数据文件均采用 csv 格式, 列之间使用`\t`分隔, 列顺序固定, 需要表头, 不使用 quoting 而使用`\`做转义.
模型定义后使用 create_dataset_from_csv 方法读取数据文件.

```python
dataset = some_model.create_dataset_from_csv("data/train.txt")
```

如果数据中没有标签, `has_label`参数需要传入`False`.

```python
dataset = some_model.create_dataset_from_csv("data/no_label.txt", has_label=False)
```

对于一些训练和预测要求的数据格式不同的任务, `evaluation`传入`True`表示预测用, `False`表示训练用

```python
training_dataset = some_model.create_dataset_from_csv("data/train.txt") # 默认evaluation=False
evaluation_dataset = some_model.create_dataset_from_csv("data/test.txt", evaluation=True)
```

### 分类

单标签

```
text    label
a   letter
b   letter
1   digit
2   digit
```

多标签

```
text    label
a1   letter|digit
b   letter
1   digit
2b   digit|letter
```

对句对分类时需要增加为三列

```
text_1    text_2    label
aa  bb  1
cc  11  0
22  99  1
33  ii  0
```

使用 X 和 y 的数据格式如下

```python
X = ["a", "b", "1", "2"] # 单句
X = [["aa", "bb"], ["cc", "11"], ["22", "99"], ["33", "ii"]] # 句对
y = ["letter", "digit"] # 单标签
y = [["letter"], ["letter", "digit"]] # 多标签
```

### 序列标注

```
text    label
浙江诸暨市暨阳八一新村00幢	[[0, 1, "prov"], [2, 4, "district"], [5, 6, "town"], [7, 10, "poi"], [11, 13, "houseno"]]
杭州五洲国际	[[0, 1, "city"], [2, 5, "poi"]]
天气热  []
```

每一个标签使用三元组表示, 三个值分别表示起始位置(包含), 结束位置(包含)和类别

```python
X = ["杭州五洲国际", "天气热"]
y = [[[0, 1, "city"], [2, 5, "poi"]], []]
```

### 语义向量检索

语义向量检索任务的训练接口(fit)和 预测接口(predict 和 score)要求的数据格式不一样.
有监督训练, 同时有正样本和负样本时, 数据为三列, 依次为文本, 正例和反例.

```
text    positive    negative
今天天气真好    天气不错    今天天气很糟糕
很开心  很高兴  好难过
```

仅有正样本时, 数据为两列, 每行一对文本对

```
text    positive
今天天气真好    天气不错
很开心  很高兴
```

无监督训练, 数据为一列

```
text
今天天气真好
很开心
```

```python
X =  ["今天天气真好", "很开心"]
y = ["天气不错", "很高兴"] # 仅有正样本
y = [["天气不错", "今天天气很糟糕"], ["很高兴", "好难过"] # 正样本和负样本
```

预测时的数据格式

```
text1   text2   label
今天天气真好    天气不错    1
很开心  很高兴  1
今天天气真好    今天天气很糟糕  0
很开心  好难过  0
```

```python
X =  [["天气不错", "今天天气很糟糕"], ["天气不错", "很高兴"]]
y = [0, 1]
```

### 文本生成

```
text    label
1234    4321
abcd    dcba
```

```python
X = ["1234", "abcd"]
y = ["4321", "dcba"]
```

## 预训练模型

预训练模型分为两类:

1. 词向量, 由 RNN 类和 CNN 类模型使用
2. BERT 类

词向量目前支持加载 word2vec 和 glove 的文本文件格式

```shell
4 3 # 可以没有第一行
1 0 0 0
2 0.1 0.1 0.1
3 0 0.1 0
4 0.1 0 0
```

通过 Word2vec 类加载.

```python
from sknlp.moduel.text2vec import Word2vec
t2v = Word2vec.from_word2vec_format("data/word2vec/vec.txt")
```

BERT 类预训练模型支持加载 google 最初公布的 BERT checkpoint 格式.

```
chinese_roberta_wwm_ext_L-12_H-768_A-12
├── bert_config.json
├── bert_model.ckpt.data-00000-of-00001
├── bert_model.ckpt.index
├── bert_model.ckpt.meta
└── vocab.txt
```

除了初版模型结构的 BERT 外, 还支持结构略有调整的 ALBERT 和 ELECTRA, 可以通过 Bert2vec 类指定模型类型加载对应模型.

```python
from sknlp.moduel.text2vec import Bert2vec, BertFamily
b2v = Bert2vec.from_tfv1_checkpoint(BertFamily.BERT, "chinese_roberta_wwm_ext_L-12_H-768_A-12")
ab2v = Bert2vec.from_tfv1_checkpoint(BertFamily.ALBERT, "albert_base")
b2v = Bert2vec.from_tfv1_checkpoint(BertFamily.ELECTRA, "electra_180g_base")
```

测试支持的模型如下

1. BERT
   - [Google BERT](https://github.com/google-research/bert)
   - [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)
   - [RoBERTa(CLUEPretrainedModels)](https://github.com/CLUEbenchmark/CLUEPretrainedModels)
   - [MacBERT](https://github.com/ymcui/MacBERT)
2. ALBERT
   - [Google ALBERT](https://github.com/google-research/albert)
   - [brightmart Google version](https://github.com/brightmart/albert_zh)
3. ELECTRA
   - [Chinese-ELECTRA](https://github.com/ymcui/Chinese-ELECTRA)

## 具体示例

参考 examples

# 部署

每个模型类都提供了一个 export 方法, 可以导出一个可用 tensorflow serving 部署的模型 pb 文件及相关辅助文件.

```shell
crf_tagger/0
├── assets
├── keras_metadata.pb
├── meta.json
├── saved_model.pb
├── variables
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── vocab.json
```

meta.json 和 vocab.json 中包含了预处理和后处理所需的信息, [sknlp-server](https://github.com/nanaya-tachibana/sknlp-server)可以作为实现的参考示例.

得益于[tensorflow text](https://github.com/tensorflow/text)的 Bert Tokenizer 的实现, Bert 类模型的预处理放到模型的前向计算中, 即 tensorflow serving 可以直接以文本作为输入.

```shell
The given SavedModel SignatureDef contains the following input(s):
  inputs['text_input'] tensor_info:
      dtype: DT_STRING
      shape: (-1)
      name: serving_default_text_input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['bert_tokenize'] tensor_info:
      dtype: DT_INT64
      shape: (-1, -1)
      name: StatefulPartitionedCall_1:0
  outputs['bert_tokenize_1'] tensor_info:
      dtype: DT_INT64
      shape: (-1, -1)
      name: StatefulPartitionedCall_1:1
  outputs['input.to_tensor'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: StatefulPartitionedCall_1:2
Method name is: tensorflow/serving/predict
```
