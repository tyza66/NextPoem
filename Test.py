from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-poem")

model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-poem")

text_generator = TextGenerationPipeline(model,tokenizer)

print(text_generator("[CLS] 万 丈 高 楼 平 地 起",max_length=50,do_sample=True))