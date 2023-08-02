---
created: "2021-11-24 22:46"
title: "从三元Loss到文本排序"

lastmod: {{ .Date }}
author: ["Sewens"]
description: ""
weight:
draft: false # 是否为草稿
comments: true # 本页面是否显示评论
math: true # 是否开启KaTex渲染页面公式
# mermaid: true #是否开启mermaid
tags:
    - text ranking
    - contrastive learning
    - learning to rank
# keywords: 
# - 
# categories:
# - 
# cover:
#     image: "" #图片路径例如：posts/tech/123/123.png
#     zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
#     caption: "" #图片底部描述
#     alt: ""
#     relative: false

---

# 背景由来
之前结合几个博客对 `triplet-loss` 进行了一定程度的学习，但是留下了一些尾巴没有搞清楚。这次集中梳理一遍。

首先明确一些基本概念，`anchor` 标准样本，`postive-sample` 正例样本即与 `anchor` 具有相同标签的样本，`negative-sample` 负例样本即与 `anchor` 标签不同的样本。

![[triplet损失示意图.png]]

从深度模型训练过程看待这个问题，如图所示为一个 siamese 网络结构，图片输入之后会通过同一个主干网络进行表示，以获取表示向量（embedding）。`triplet-loss` 的训练目标在于通过对损失的优化，对主干网络参数进行调整，实现相似的样本之间在高维空间中具有相近的距离，从而在给定标准样本前提下，能够识别输入的样本是否为当前的样本。

三者构成一个三元组 `(i, j, k)`，其中 `i` 为 `anchor`，`j` 为 `postive` 而 `k` 为 `negative`。

`triplet-loss` 形式如下如下所示：
$$
loss = max(margin - d(a,p) + d(a,n), 0)
$$
其中 `d(·,·)` 表示计算两个样本之间的向量空间上的距离，即 distance，满足一下性质的函数都可称为距离。

<!-- ![[距离#定义]] -->

因此可以看到，`triplet-loss` 结合了样本之间的距离度量和 `margin-loss`，计算标准样本和正例以及负例样本之间的分数，之后margin 为界对不同“情况”的正负样本进行区分。

因此可以看到，组成 `triplet-loss` 的基本形式包含三个部分：

* 一个三元组 `(i,j,k)`，包含标准样本、正例样本和负例样本相关的表示向量；
* 一个距离度量函数 $d$ 用来衡量样本之间的向量距离；
* 一个 `margin-loss` 最小化损失的最终目的在于使得正例样本和标准样本的空间距离尽可能小，反之负例样本和标准样本之间的距离尽可能大；

当完成以上三点的构造之后即可计算 `triplet-loss`。

# 两种策略
可以看到，三个要素中，`margin-loss` 的基本形式是确定的，距离度量 $d$ 也基本是确定的，因此三元组的产生就是 `triplet-loss` 构造的关键。

三元组的构建过程有两种策略，离线方法（offline）和在线方法（online）。

离线方法中，针对一个标准样本，通过数据采样等方式构造正例和负例，形成训练数据集。在模型训练时直接输入三元组，通过模型表示之后可以直接计算 $d(a,p)$ 和 $d(a,n)$，之后计算具体损失。
离线方式优势在于对于数据处理较为灵活，能够更加多样的策略构造三元组，同时一些额外的信息也可以在三元组基础上来对齐引入。其劣势在于会有额外的运算开销，假定 `B` 个标准样本，每个对应着 `B` 个正例样本，同时负例样本也有 `B` 个，则构造得到的三元组的量级为 `B*B*B` ，训练集三元组的采样和存储开销过大。

在线方式则是在一个 batch 数据内对正例和负例进行采样，通过输入标签区分具有相同标注的数据。例如一个 batch `B` 中共有 `P` 个人的人脸数据，每个人包含 `K` 张不同的照片。同一个人的不同照片之间互为正例，而不同人之间的照片互为负例，以此为基础构造三元组。在线方式的优势在于开销更少，更加“端到端”，只要做好数据分组即可进行学习，一定程度上类似于聚类方法。

# 在线学习的快速采样
在线构建三元损失关键在于如何找到学习目标相关的正例和负例样本，在此基础上再结合损失函数进行学习。

为表述清晰此处给出 `pytorch` 下的示例代码。
设一个 batch 内包含四个样本的表示向量，为 `B=[a,b,c,d].T`，各自代表一个 `p` 维的特征向量。其中 `a,b` 同属一个类别，`c,d` 同属一个类别，因此标签空间为 `[1,1,2,2]`。

首先计算当前 batch 内样本两两之间的相似程度，直接进行向量的转置相乘，即 `BB=B*B.T`，可以得到：

```python
p = 300
a = torch.rand(1,p)
b = torch.rand(1,p)
c = torch.rand(1,p)
d = torch.rand(1,p)

B = [a,b,c,d]
mm = lambda x,y:torch.matmul(x,y.T)

pairwise_score = mm(B,B)
```

为了版面清晰，此处约定 `xy=torch.matmul(x,y.T)`，即两个向量之间做转置乘法得到内积。则样本之间的相似分数可被简写为：

```
pairwise_score = [
    [aa, ab, ac, ad],
    [ba, bb, bc, bd],
    [ca, cb, cc, cd],
    [da, db, dc, dd],
]
```

在此基础上，做这样的一个构造：

```python

margin = 0.361433

anchor_positive_score = pairwise_score.unsqueeze(dim = 2)
anchor_negative_score = pairwise_score.unsqueeze(dim = 1)

raw_triplet_with_margin = anchor_positive_score - anchor_negative_score + margin

```

此处对相似分数矩阵做维度扩展，可以得到标准样本和正例样本之间的分数 `anchor_positive_score` 以及标准样本和负例样本之间的分数 `anchor_negative_score`，二者相减则可以凑出 `d(a,p)-d(a,n)` 形式。

**P. S.**：这一步操作依赖 torch 框架中向量运算的传播（broadcast）特性，做加减法的两个矩阵会首先被扩展为相同 shape 对应维度上的元素会被复制，之后对应元素相加。

因此可以得到以下结果：
```
anchor_positive_score = [
    [[aa], [ab], [ac], [ad]],
    [[ba], [bb], [bc], [bd]],
    [[ca], [cb], [cc], [cd]],
    [[da], [db], [dc], [dd]],
]

anchor_negative_score = [
    [[aa, ab, ac, ad]],
    [[ba, bb, bc, bd]],
    [[ca, cb, cc, cd]],
    [[da, db, dc, dd]],
]

# 此处需要注意为何是pos-neg 因为以向量内积形式做为distance时
# distance越大 表明 样本之间越相似 严格意义上来讲不符合距离的定义
# 但在实践中是如此来用的

raw_triplet = anchor_positive_score - anchor_negative_score 

raw_triplet = [
    [
        [aa-aa,aa-ab,aa-ac,aa-ad],
        [ab-aa,ab-ab,ab-ac,ab-ad],
        [ac-aa,ac-ab,ac-ac,ac-ad],
        [ad-aa,ad-ab,ad-ac,ad-ad]
    ],
    [
        [ba-ba,ba-bb,ba-bc,ba-bd],
        [bb-ba,bb-bb,bb-bc,bb-bd],
        [bc-ba,bc-bb,bc-bc,bc-bd],
        [bd-ba,bd-bb,bd-bc,bd-bd]
    ],
    [
        [ca-ca,ca-cb,ca-cc,ca-cd],
        [cb-ca,cb-cb,cb-cc,cb-cd],
        [cc-ca,cc-cb,cc-cc,cc-cd],
        [cd-ca,cd-cb,cd-cc,cd-cd]
    ],
    [
        [da-da,da-db,da-dc,da-dd],
        [db-da,db-db,db-dc,db-dd],
        [dc-da,dc-db,dc-dc,dc-dd],
        [dd-da,dd-db,dd-dc,dd-dd]
    ]
]
```

`triplet-loss` 基本的三元组形式构造完成后，需要筛选其中有效部分。以三元组 `(i,j,k)` 为例。有效的 `d(a,p)-d(a,n)` 形式中包含 `a!=p` 即独特性（distinct）及根据标签推出的三元组有效性（valid）一共两个条件。

首先看独特性条件，样本自身做内积的情况不予考虑。

```
valid_raw_triplet = [
    [
        [0,0,0,0],
        [0,0,ab-ac,ab-ad],
        [0,ac-ab,0,ac-ad],
        [0,ad-ab,ad-ac,0]
    ],
    [
        [0,0,ba-bc,ba-bd],
        [0,0,0,0],
        [bc-ba,0,0,bc-bd],
        [bd-ba,0,bd-bc,0]
    ],
    [
        [0,ca-cb,0,ca-cd],
        [cb-ca,0,0,cb-cd],
        [0,0,0,0],
        [cd-ca,cd-cb,0,0]
    ],
    [
        [0,da-db,da-dc,0],
        [db-da,0,db-dc,0],
        [dc-da,dc-db,0,0],
        [0,0,0,0]
    ]
]

distinct_mask = [
        [[0, 0, 0, 0],
         [0, 0, 1, 1],
         [0, 1, 0, 1],
         [0, 1, 1, 0]],

        [[0, 0, 1, 1],
         [0, 0, 0, 0],
         [1, 0, 0, 1],
         [1, 0, 1, 0]],

        [[0, 1, 0, 1],
         [1, 0, 0, 1],
         [0, 0, 0, 0],
         [1, 1, 0, 0]],

        [[0, 1, 1, 0],
         [1, 0, 1, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0]]
]

valid_raw_triplet = raw_triplet * distinct_mask
```

上述过程具体实现方式如下：
```python

indices_equal = torch.eye(4).bool()
indices_not_equal = ~indices_equal

i_not_equal_j = indices_not_equal.unsqueeze(2)
i_not_equal_k = indices_not_equal.unsqueeze(1)
j_not_equal_k = indices_not_equal.unsqueeze(0)

distinct_mask =((i_not_equal_j & i_not_equal_k) & j_not_equal_k)

```
同样依赖矩阵运算的 broadcast 特性，构造 `i!=j;j!=k;i!=k` 的 distinct mask。

之后再看标签的有效性条件，当前的标签为 `[1, 1, 2, 2]`，首先按照标签将数据分为两组，之后剔除掉无效的样本组合，具体代码如下：

```python
label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
i_equal_j = label_equal.unsqueeze(2)
i_equal_k = label_equal.unsqueeze(1)

valid_mask = ~i_equal_k & i_equal_j
```

中间结果如下：

```
label_equal = [
    [1,1,0,0],
    [1,1,0,0],
    [0,0,1,1],
    [0,0,1,1],
]

i_equal_j = [
    [[1],[1],[0],[0]],
    [[1],[1],[0],[0]],
    [[0],[0],[1],[1]],
    [[0],[0],[1],[1]],
]

~i_equal_k = [
    [[0,0,1,1]],
    [[0,0,1,1]],
    [[1,1,0,0]],
    [[1,1,0,0]],
]

valid_mask = [
    [
        [0,0,1,1]
        [0,0,1,1],
        [0,0,0,0],
        [0,0,0,0]
    ],
    [
        [0,0,0,0]
        [0,0,0,0],
        [1,1,0,0],
        [1,1,0,0]
    ],
    [
        [0,0,0,0]
        [0,0,0,0],
        [1,1,0,0],
        [1,1,0,0]
    ],
]

```

最后 distinct mask 和 valid mask 二者复合构成筛选有效三元组的最终 `triplet_batch_all_mask`，与 `raw_triplet_with_margin` 相乘之后得到有效的 `triplet-loss`。

```
triplet_batch_all_mask = distinct_mask & valid_mask

triplet_batch_all_mask = [
        [[0, 0, 0, 0],
         [0, 0, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[0, 0, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 1, 0, 0]],

        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0]]
]
```

# 在线采样的两种策略

如图所示，三元组中存在不同的样本类型，一些简单一些困难，均根据标准样本和正例样本以及负例样本之间的 distance 进行区分，其中：

* Hard postive：`d(a,p)`非常大，即正例样本和标准样本之间的距离很大，与负例样本之间混淆
* Hard negative `d(a,n)<d(a,p)`即负例样本和标准样本之间距离很小，与正例样本之间混淆
* Semi-Hard negative：`d(a,p)<d(a,n)<d(a,p)+margin`负例样本与标准样本之间距离虽然有一定距离，但是仍处于margin之内，没有明确的区分开

重新看一下 loss 可以写作：
$$
loss = max(margin - d(a,p) + d(a,n),0)
$$

根据值不同可以划分为不同难度的三元组。

![[margin示意图.png|500]]

再回顾上面所述的例子，有 `P` 个不同人的照片，每个人有 `K` 张，因此一个 batch `B` 内包含着 `B=P*K` 个照片样本。即类似于 few-shot learning 中的，P-way，K-shot 概念。

此处的有效三元组是指，假定三元组`(i, j, k)`为一个三元组，各自代表一个样本。如果其中`i`为anchor即标准图片，`j`为正例样本即和`i`同属于一类，`k`为负例样本即与`i`属于不同类别，并且有`i!=j`即`i`和`j`不是同一个样本，则`(i, j, k)`为一个有效的三元组，可参与训练。

## 策略1：batch all

即 batch 内的所有样本，只要是有效的三元组，都参与到训练中，包含 hard negative、hard positive、semi-hard negative 等。总的样本数目计算下来为 `PK*(K-1)*(P-1)K`，即包含 `PK` 个 anchor，每个 anchor 对应一共 `K-1` 个正例样本（每个组中首个样本被作为 anchor，同时样本和自身是无效配对），剩余的 `P-1` 组中，每组 `K` 个样本，均作为负例。

此处需要注意，easy positive 样本会被剔除不用，因为其贡献的 loss 为 0（通过 `max (d(a,p)-d(a,n)+margin,0)` 计算之后，soft positive 会被归零），如果考虑在内的话，在做 per sample loss 平均时会导致 loss 结果偏低。可以考虑在输出损失结果之前做一次 mask 操作，将归零的样本不计入统计中。之后针对 per-sample loss 进行平均时候就不会有偏小的情况。

具体的选择策略在上面已经详细分析过了，样本 `B=4`，标签为 `[1,1,2,2]`。更多样本和标签的情况使用上述代码可以直接求得。

## 策略2：batch hard

即 batch 内只选择最难的样本作为训练候选。只考虑每一个 batch 内最强的正例样本与最强的负例样本进行训练。

与 batch all 策略相比，batch hard 方法首先需要得到标准样本与正例样本和负例样本之间的分数。之后选出 hard sample，作为有效的搭配进行计算。

延续上述例子，首先我们计算一个 pairwise score，之后获取可以作为正例和负例的样本：

```
pairwise_score = [
    [aa, ab, ac, ad],
    [ba, bb, bc, bd],
    [ca, cb, cc, cd],
    [da, db, dc, dd],
]

positive_sample = [
    [0, ab, 0, 0],
    [ba, 0, 0, 0],
    [0, 0, 0, cd],
    [0, 0, dc, 0],
]

negative_sample = [
    [0, 0, ac, ad],
    [0, 0, bc, bd],
    [ca, cb, 0, 0],
    [da, db, 0, 0],
]
```

生成对应 mask 的代码如下：

```python

def _get_anchor_positive_triplet_mask(labels):

    # 标签相同的样本可以作为正例
    # 但是要去除样本自身和自身相乘的
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()

    indices_not_equal = ~indices_equal

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal

def _get_anchor_negative_triplet_mask(labels):
    
    # 标签不相同的所有样本组合都是有效的负例
    
    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))
```

计算完成得到所有的正例样本和负例样本，之后求取正例样本中距离分数最大的，以及负例样本中距离分数最小的样本。作为最终的 hard sample 三元组，参与训练。

# 距离还是分数
再回顾一次公式，可以发现与代码中的损失函数表述相互冲突了，代码中计算了`pos-neg+margin`。
$$
loss = max(margin - d(a,p) + d(a,n),0)
$$
首先直观上分析 margin loss 函数，目的在于指导模型学习样本的表述能力，以达到高维空间中正例样本和标准样本之间的距离更小，负例样本和标准样本距离更大的效果，如下图所示，实际类似于聚类。

![[embeddings.gif]]

此处的`d(·,·)`的选用是自由的，如果使用了标准的距离度量，例如欧氏距离（euclidean），则损失的形式为`max(margin+d(a,p)-d(a,n),0)`，因为距离越大意味着样本之间相似程度越小。
而如果使用其他的衡量样本之间距离分数的方法，例如余弦公式（cosine），则损失的形式为`max(margin-d(a,p)+d(a,n),0)`，因为余弦分数越大，说明向量在空间中的距离越小，进而相似程度越高。为保证一致性，其实也可以用 1 减去余弦分数，将分数形式转换为满足距离定义的格式。

**P. S.**：向量内积方法类似于余弦分数，在使用时候应当注意。

# 文本排序任务
上文中以图像分类为例子展示了如何以在线方式进行三元组挖掘并对模型进行学习。可以看到，隐含在其中的一个重要部分是，基础的深度模型为孪生网络结构（siamese network），即用于输入样本特征抽取的表示层结构共享参数（实际就是同一个网络），获取样本特征表示之后再进行三元组的挖掘。

文本检索排序任务与之类似，双塔结构中查询（query）和文档（document）分别通过两个独立结构被表示为特征向量，之后计算二者的相关性。

此时的三元组`(i, j, k)`分别表示查询文本（query），相关文档（relevant document）以及不相关文档（irrelevant document）。如果借助上述`triplet-loss`方法，则输入数据需要做少许调整：
$$
B = [q_0,d_0^0,d_1^0,...d_k^0,q_1,d_0^1,d_1^1,......,q_p,d_0^p,d_1^p,...,d_k^p]
$$
即输入数据中包含查询文本和文档文本，分为`P`组，每组以查询文本为首个元素，后面跟`K`个相关文档，这样构造数据时对应的输入标签如下，每组标签包含`P+1`个样本。
$$
L = [0,0,...,0,1,1,...,1,....,p-1]
$$
使用上述方法进行训练时，不仅查询和文档之间会构成训练的三元组，同时`P`个查询之间也会构成训练样本组合，因此相对更适用于查询文本之间差异较大的场合，或者强制对查询之间的差异进行学习。
另外一点的是，这种方法只能对相关文档进行学习，但是相关文档中会被一视同仁作为相同等级的文档来使用，并不能学习到相关性的大小，只能学习到相关与不相关文档之间的差异。而且这种方式构造输入 batch 略显复杂，要求 batch size 必须为`P+1`的整数倍。

上述过程可以稍作简化，降低构造 batch 的难度。给定训练查询和相关文档的 pair`(q,d)`，batch size 为`B`，则一个 batch 包含着 B 个这样的查询-文档对。通过网络获取表示之后可以得到两个表示向量矩阵：$Q=[q_{0},q_{1},...q_{B-1}]$和$D=[d_{0},d_{1},...d_{B-1}]$，二者做转置相乘得到查询和文档表示向量之间的内积，即相似度矩阵（类似于余弦分数）：
```
Q*D. T = [
    [q1d1,q1d2,q1d3,...,q1dB],
    [q2d1,q2d2,q2d3,...,q2dB],
    ......
    [qBd1,qBd2,qBd3,...,qBdB],
]
```
可以看到，相似度矩阵的对角线即为所有的正例样本的相似度分数，其余部分为非相关片段的分数。
与上面相对应，使用 batch all 策略时，所有的负例样本按行求和并取平均得到正例样本相对的所有负例样本的平均相似分数。
而使用 batch hard 策略时，针对每个查询，正例样本只有一个因此不提。负例样本中，按照每行进行处理，取出其中分数最大的一个作为 hard negative sample，与正例构成三元组，进而求取损失。

代码如下：
```python

def batch_all_triplet_loss(
                       query_feature_tensor:torch.Tensor,
                    doc_feature_tensor:torch.Tensor):
    
    device = query_feature_tensor.device
    batch_size = query_feature_tensor.shape[0]
    neg_mask = (torch.ones(batch_size) -  
                                torch.eye(batch_size)).to(device)

    distance = torch.matmul(query_feature_tensor,doc_feature_tensor)
    distance = torch.softmax(distance, dim=1)

    pos_tensor = torch.diag(distance)
    neg_mat = distance * neg_mask

    neg_tensor = torch.mean(neg_mat,1)

    per_sample_loss = torch.maximum(self.margin + pos_tensor - neg_tensor, 
                                    torch.zeros(batch_size).to(device))

    return torch.mean(per_sample_loss)
    
def batch_hard_triplet_loss(
                  query_feature_tensor:torch.Tensor,
                doc_feature_tensor:torch.Tensor):
    
    device = query_feature_tensor.device
    batch_size = query_feature_tensor.shape[0]

    neg_mask_mat = torch.diag(torch.tensor([float('-inf') 
                        for i in range(batch_size)])).to(device)

    distance = torch.matmul(query_feature_tensor,doc_feature_tensor)

    pos_score = torch.diag(distance)
    neg_score = torch.max(distance+neg_mask_mat,-1)[0]

    per_sample_loss = torch.maximum(self.margin - pos_score + neg_score,
                                    torch.zeros(batch_size).to(device))

    return torch.mean(per_sample_loss)
```

再进一步考虑，实际上这种基于查询-文档对的训练目标，在构建相似度分数矩阵`Q*D.T`之后，优化的模型目标就在于使得对角线上的分数尽量大，其余位置尽量小。
所以为何不直接用`cross entropy`呢：
```python

def binary_cross_entropy_pairwise_ranking(
                query_feature_tensor:torch.Tensor,
               doc_feature_tensor:torch.Tensor):
    
    device = query_feature_tensor.device
    batch_size = query_feature_tensor.shape[0]

    distance = torch.matmul(query_feature_tensor,doc_feature_tensor.T)
    distance = torch.softmax(interact_matrix, dim=1)

    dummy_label = torch.diag(torch.ones((batch_size))).to(device)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                        distance,dummy_label)

    return loss
```

测试效果之后发现，`BCE`的训练效果和收敛速度也一般都优于`margin`类的损失。
# 参考链接
* https://omoindrot.github.io/triplet-loss
* https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
* https://gombru.github.io/2019/04/03/ranking_loss/
* https://github.com/adambielski/siamese-triplet/blob/master/losses.py
* https://kevinmusgrave.github.io/pytorch-metric-learning/
* https://github.com/UKPLab/sentence-transformers/tree/master/sentence_transformers/losses