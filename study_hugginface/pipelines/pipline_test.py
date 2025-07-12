from transformers import pipeline

# 1.分本分类 、情感分析
text_classification = pipeline("sentiment-analysis")
classification = text_classification("I Love You")
print(classification)

# 2.阅读理解
context = ""
question_answering = pipeline("question-answering")
answering = question_answering(question="What is extractive question answering?", context=context)
print(answering)

# 3.完形填空 <mask>
sentence = ""
fill_mask = pipeline("fill-mask")
mask = fill_mask(sentence)
print(mask)

# 4.文本生成
text_generation = pipeline("text-generation")
text = text_generation("As far as I am concerned, I will", max_length=50, do_sample=False)

# 5.命名实体识别
# 一段文本中找出实体；地名、人名等名词
sequence = ""
ner = pipeline("ner")
entity = ner(sequence)
for e in entity:
    print(e)

# 6.文本摘要
APICLE = ""
summarization = pipeline("summarization")
abstract = summarization(APICLE, max_length=130, min_length=30, do_sample=False)
print(abstract)

# 7.翻译任务 translation_xx_to_yy
sentence = ""
translator = pipeline("translation_en_to_de")
translation = translator(sentence, max_length=400)
print(translation)

# 替换模型，执行任务
# 如果默认的pipeline任务模型不满足要求，可以自己指定模型
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# pip install sentencepiece
sentence = "我叫萨拉，我住在伦敦"
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
translator_cus = pipeline(task="translation_zh_to_en", model=model, tokenizer=tokenizer)
translation = translator_cus(sentence, max_length=20)
print(translation)


