from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline

dataset = "jawerty/html_dataset"
new_model = "Llama-2-finetune"
dataset = load_dataset(dataset, split="train")
dataset.train_test_split(test_size=0.2, shuffle=True)

#Loading the pretrained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(new_model)
tokenizer = AutoTokenizer.from_pretrained(new_model)

prompt = "Generate an HTML code for Simple Landing Page"
pipe = pipeline(task = "text-generation", model = model, tokenizer = tokenizer, max_length = 200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
generated_text = list(result[0]['generated_text'])

metric = load_metric("bleu")

bleu_score = metric.compute(predictions=[generated_text], references=[dataset['train']])
print(f"Bleu Score Metric when references is train dataset: {bleu_score}")

bleu_score = metric.compute(predictions=[generated_text], references=[dataset['test']])
print(f"Bleu Score Metric when references is train dataset: {bleu_score}")


