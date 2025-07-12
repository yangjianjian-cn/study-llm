# bert-base-chinese
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',  # bert-base-chinese模型在huggingface上的路径或名称，这里用的模型名称
    cache_dir=None,  # 使用默认路径来缓存模型文件
    force_download=False  # 不是每次都强制下载，从缓存路径拿
)

sents = [
    '你站在桥上看风景',
    '看风景的人在楼上看你',
    '明月装饰了你的窗子',
    '你装饰了别人的梦'
]

# 批量编码函数
out = tokenizer.batch_encode_plus(
    # 句子对
    # 如果要对单句子编码, batch_text_or_text_pairs=[sents[0], sents[1], sents[2], ...]
    batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],
    truncation=True,
    padding='max_length',
    max_length=25,
    add_special_tokens=True,
    return_tensors=None,
    return_token_type_ids=True,
    return_attention_mask=True,
    return_special_tokens_mask=True,
    return_length=True
)
# print(out)

# 查看模型使用的词典
#bert_base_chinese模型，把每个中文字当成一个词.
vocab = tokenizer.get_vocab()

# 词典添加词
tokenizer.add_tokens(new_tokens=['明月', '装饰', '窗子'])
# 添加特殊字符
tokenizer.add_special_tokens({'eos_token': '[EOS]'})
for word in ['明月', '装饰', '窗子', '[EOS]']:
    print(tokenizer.get_vocab()[word])


# 用新词表去编码
out = tokenizer.encode(text='明月装饰了你的窗子[EOS]',
                text_pair=None,
                truncation=True,
                padding='max_length',
                add_special_tokens=True,
                max_length=10,
                return_tensors=None)

# 解码
text = tokenizer.decode(out)

# tokenizer.encode()
# 编码器进阶版
tokenizer.encode_plus(
    text=sents[0],
    text_pair=sents[1],
    truncation=True,
    padding='max_length',
    max_length=25,
    add_special_tokens=True,
    return_tensors=None,
    return_token_type_ids=True,
    return_attention_mask=True,
    return_special_tokens_mask=True,
    return_length=True
)
# 批量编码函数
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],
    truncation=True,
    padding='max_length',
    max_length=25,
    add_special_tokens=True,
    return_tensors=None,
    return_token_type_ids=True,
    return_attention_mask=True,
    return_special_tokens_mask=True,
    return_length=True
)
print(out)

vocab = tokenizer.get_vocab()
