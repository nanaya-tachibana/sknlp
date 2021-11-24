![CI badge](https://img.shields.io/github/workflow/status/nanaya-tachibana/sknlp/CI)
![coverage badge](https://img.shields.io/codecov/c/github/nanaya-tachibana/sknlp)

*sknlp*是一个使用 Tensorflow 实现的仿[scikit-learn](https://scikit-learn.org/stable/)接口设计的 NLP 深度学习模型库, 训练的模型可直接使用 tensorflow serving 部署.

写这个库的原因, 一是搭建一个模型实现的基础模版, 方便自己实验新模型; 二是标准化各类任务中一些常用模型, 方便工作中快速上线原型服务.

# 安装

```shell
pip install sknlp
```

# 功能

1. 分类, 单标签和多标签
   - BertClassifier, Bert 微调
   - RNNClassifier, bilstm + attention
   - RCNNClassifier, [Recurrent Convolutional Neural Network for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745)
   - CNNClassifier, 空洞卷积
2. 序列标注, crf 解码和[global pointer](https://spaces.ac.cn/archives/8373)解码
   - BertTagger, Bert 微调
   - BertRNNTagger, Bert + bilstm
   - RNNTagger, bilstm
   - CNNTagger, 空洞卷积
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
clf = BertClassifier(["letter", "digit"], text2vec=b2v)
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

b2v = Word2vec.from_word2vec_format("data/jieba/vec.txt", segmenter="jieba")
clf = RNNClassifier(["letter", "digit"], text2vec=b2v)
training_dataset = clf.create_dataset_from_csv("data/train.txt")
test_dataset = clf.create_dataset_from_csv("data/test.txt")
clf.fit(
    dataset=training_dataset,
    n_epochs=5,
    learning_rate=1e-4
)
clf.predict(dataset=test_dataset)
clf.score(dataset=test_dataset)
```
