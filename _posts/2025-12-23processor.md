---
title: '深挖训练模型时dataset,dataloader和processor的关系，以及WebDataset的机制'
date: 2025-12-23
permalink: /posts/2025/12/processor-dataset-dataloader/
tags:
  - learning note
---

---

## 遇到的问题

自定义了一个dataset，准备用来微调Qwen3-VL

这类开源模型一般都带有一个processor，将image（支持PIL.Image，numpy，tensor等等）,text等数据向量化为模型的输入。

输入的格式非常混乱，且没有一个统一的标准（至少我没找到文档...），似乎是一个约定俗成的东西。

例如，如果你设置了return_dict=True，返回值将为一个字典，大概长这样
```python
input={
    "input_ids": tensor, # 在NLP中常用，文本tokenize后的id
    "attention_mask": tensor, 
    "pixel_values": tensor, # CV中常用，图像的像素值（当然，是分patch之后的）
    "pixel_attention_mask": tensor,
    "spatial_shapes": tensor,
}
```
或者是一个叫做``BatchFeature``的类，长这样
```python
class BatchFeature(BaseModel):
    data: Dict[str, Any]
    tensor_type: Optional[str] = None
```
在[hf官方文档](https://huggingface.co/docs/transformers/en/main_classes/data_classes#transformers.BatchFeature)中对这个类的描述是：
>参数
>>data (dict, *可选*) — 由 __call__/pad 方法返回的列表/数组/张量字典（‘input_values’、‘attention_mask’等）。
>>
>>tensor_type (Union[None, str, TensorType], *可选*) — 你可以在此处指定一个 tensor_type，以便在初始化时将整数列表转换为 PyTorch/TensorFlow/Numpy 张量。
>>
>该类派生自 Python 字典，可以像字典一样使用。

也是个字典。

这些键值可以说是奇奇怪怪，也没有文档说明，暗藏在这些库最底层的代码中，你要是想中间拆开，插入一点自己的东西，比登天还难。

这就遇到了一个问题（从上个blog扩展而来：[Transformers SigLip2 processor的一个坑](https://bajimh.github.io/posts/2025/12/siglip2/)）：

processor本身一般都很慢（一个大for循环，没有任何并行优化），
对于大语言模型来说，processor还包含了tokenizer。，如果先dataloader读出一个batch，再processor处理，那么效率非常低。要么就整个重写，这又很麻烦，要么就外面套成并行，怎么搞？

正确的姿势似乎应该是这样的（我找了很久没找到相关的教程和文档）：

dataset类往往支持map函数，即对每一个item，都调用一个函数进行处理。

>需要注意的是webdataset和dataset的表现似乎并不一样（webdataset似乎都是惰性处理的,**只有迭代的时候会真正读取**）
>
>而dataset则不是，当加载数据集时指定streaming=True（流式数据集），map会变为完全惰性执行，与 WebDataset 的map行为一致：
```python
from webdataset.compat import WebDataset
def process_function(sample):
    print("processing sample")
    return sample
def main():
    tar_path = "XXX.tar"
    dataset = WebDataset([tar_path], shardshuffle=None).decode("pil").map(process_function)
    print("YES")
    for sample in dataset:
        print(sample)
        break
```
大家可以试一下这个程序，它会先输出YES，再输出processing sample。意味着map只有在迭代的时候才会执行。

根据和豆包反复对话得来的结论，我们似乎应该写一个process函数，在process函数中完成模型的预处理，然后对整个dataset调用map函数（把每个sample都提前处理一遍，而不是训练时一遍遍处理），然后保存到一块可持久的存储空间（例如硬盘）。训练的时候直接加载（dataset类也支持打包整个数据集到硬盘/读取）
- 对于流式数据集和webdataset而言，只是构建了一个数据集的映射，并没有真正读取、处理数据

但是这样很容易产生另一个问题：sample是如何聚合到batch的？

## 读底层代码真的很烦人

这是dataloader的核心逻辑之一，默认不需要你实现

torch.utils.data.dataloader.py里面有这么一句：
```python
default_collate: _collate_fn_t = _utils.collate.default_collate
```
这个函数为常见的格式都做了兼容，注释中介绍了一些例子

```python
def default_collate(batch):
"""
    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`
"""
```

其中这个Mapping就是我们前面提到的东西，如果它是一个字典/BatchFeature，那么它会分别的聚合每个value。而前面的value是processor处理完的结果，一般是一个torch.Tensor，在第一维加个B，变成(B, ...)，这就是我们喜闻乐见的东西了。

如果你用过webdataset，还有一个torch dataloader的平替叫WebLoader，它和dataloader接口非常相似。

这样以来，我们的数据加载流程就通顺了：

1. 自定义dataset类做读入，实现get_items/或者用webdataset之类的东西读入
2. 实现process函数，将sample处理为模型可接受的格式
3. 构建webdataset pipeline（不会立刻执行，只是告诉他以后要这么做）
4. 用dataloader类封装dataset类，设置好batchsize，shuffle，num_workers等参数
5. 模型直接用dataloader类并行加载数据，在加载的时候真正执行pipeline，不需要接触任何底层代码

