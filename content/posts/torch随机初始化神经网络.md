---
author: sewen
created: "2022-01-18 17:29"
tags: ["#lawbda-kb"]
title: "torch随机初始化神经网络"
---
# 背景
最近有个实际需求，对比尝试直接在 Bert 和 Roberta 架构下直接进行模型训练。所以要求在加载原有模型的架构和权重基础上，将权重全部初始化。尝试了最简单的方案，即通过`state_dict`来修改模型内部的权重参数，结果模型跑起来直接 loss 变为 `nan`。看样子这种简单粗暴方案不太行。
代码如下：
```python
if 'bert_initial' in self.train_args and self.train_args['bert_initial']:
    old_state_dict = self.BertModel.state_dict()
    new_state_dict = {}
    # 使用rand进行随机初始化
    for name in old_state_dict:
        new_state_dict[name] = torch.rand(old_state_dict[name].shape)
    # 更新模型参数
    self.BertModel.load_state_dict(new_state_dict)
```

查了一下这里用的`torch.rand`的实际原理：https://pytorch.org/docs/stable/generated/torch.rand.html。即从 `[0,1)`区间内的均匀分布上取一个数，测试之后发现完全不行，输出的 loss 直接变`nan`。

# 初始化方法
搜了搜 torch 初始化网络的方法，可以用下面代码，以卷积层为例：
```python
conv1 = torch.nn.Conv2d(...)
torch.nn.init.xavier_uniform(conv1.weight)
```
即调用`torch.nn.init`中相关的方法对权重进行初始化。这一类方法是 in place 的，即直接修改对象的权重，不用重新再赋值了。

如果要对整个`torch.nn.Module`进行初始化，或者对指定名称（已注册）的对象进行初始化，可以使用：
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)
```

# bert 初始化
进一步搜了搜，在 hugging face 的`transformers`库中模型的初始化方案。issue-4701 对这个问题进行了讨论，原因是提问者发现了模型在加载过程中首先会调用`init_weights`方法对权重进行初始化，对应代码如下，原始代码位置在：https://github.com/huggingface/transformers/blob/a9aa7456ac/src/transformers/modeling_bert.py#L520-L530
```python
def _init_weights(self, module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
```
对于这一部分的解答也很清晰：

>Have a look at the code for [`.from_pretrained()`](https://github.com/huggingface/transformers/blob/a9aa7456ac824c9027385b149f405e4f5649273f/src/transformers/modeling_utils.py#L490). What actually happens is something like this:
> -   find the correct base model class to initialise
> -   initialise that class with pseudo-random initialisation (by using the `_init_weights` function that you mention)
> -   find the file with the pretrained weights
> -   overwrite the weights of the model that we just created with the pretrained weightswhere applicable
> This ensure that layers were not pretrained (e. g. in some cases the final classification layer) _do_ get initialised in `_init_weights` but don't get overridden.

即模型在调用`from_pretrained`方法读取现有的预训练权重时，会首先调用`init_weights`对模型结构进行初始化，同时初始化一个权重。之后读取现有的预训练权重，再对初始化的权重覆盖就行。

# 最后方案
所以综上所述，在`transformers`库中想要从头预训练一个模型的话，可以直接调用`init_weights`方法对模型进行初始化。用于测试的代码如下：
```python
test_config = AutoConfig.from_pretrained('roberta-base')

test_tkm = RobertaTokenizer.from_pretrained('roberta-base')
test_model = RobertaModel.from_pretrained('roberta-base',config=test_config)

o_test_model = copy.deepcopy(test_model)

test_model.init_weights()

o_state_dict = dict(o_test_model.named_parameters())
t_state_dict = dict(test_model.named_parameters())

for name,param in test_model.named_parameters():
    print(name,o_state_dict[name]==t_state_dict[name])
```
最后可以看到二者的权重已经不一样了。

# 相关资源
* https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
* https://github.com/huggingface/transformers/issues/4701