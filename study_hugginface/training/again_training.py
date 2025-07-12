# 加载tokenizer
from datasets import load_from_disk
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')
print(tokenizer)

# 试用
tokenizer.batch_encode_plus(['明月装饰了你的窗子', '你装饰了别人的梦'], truncation=True)

# 从磁盘加载中文数据集

dataset = load_from_disk('./data/ChnSentiCorp/')
# 缩小数据规模, 便于测试.
dataset['train'] = dataset['train'].shuffle().select(range(2000))
dataset['test'] = dataset['test'].shuffle().select(range(100))
print(dataset)


def f(data, tokenizer):
    return tokenizer.batch_encode_plus(data['text'], truncation=True)


dataset = dataset.map(f, batched=True,
                      batch_size=1000,
                      num_proc=4,
                      remove_columns=['text'],
                      fn_kwargs={'tokenizer': tokenizer})

print(dataset)


# 删掉太长的句子
def f(data):
    return [len(i) <= 512 for i in data['input_ids']]


dataset = dataset.filter(f, batched=True, batch_size=1000, num_proc=4)

# 根据任务类型加载模型
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('hfl/rbt3', num_labels=2)
# 统计模型参数量
sum([i.nelement() for i in model.parameters()])
# 模型试算
import torch

# 模拟一条数据
data = {
    'input_ids': torch.ones(4, 10, dtype=torch.long),
    'token_type_ids': torch.ones(4, 10, dtype=torch.long),
    'attention_mask': torch.ones(4, 10, dtype=torch.long),
    'labels': torch.ones(4, dtype=torch.long)
}

out = model(**data)
