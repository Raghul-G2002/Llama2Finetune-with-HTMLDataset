from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = "jawerty/html_dataset"
new_model = "Llama-2-finetune"

#loading the dataset
dataset = load_dataset(dataset, split="train")
dataset.train_test_split(test_size=0.2, shuffle=True)

#Loading the pretrained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(new_model)
tokenizer = AutoTokenizer.from_pretrained(new_model)

#Tokenizing the dataset and Generate Predictions
def tokenize_function(examples):
    return tokenizer(examples, return_tensors="pt")


#Tokenize and Generate Prediction
def evaluation(data):
    references = []
    predictions = []
    for example in data:
        input_text = example['label']
        reference_text = example['html']

        input_ids = tokenize_function(input_text)['input_ids']
        output_ids = model.generate(input_ids)

        predicted_text = tokenizer.decode(output_ids[0],clean_up_tokenization_spaces=True)
        predictions.append([predicted_text])
        references.append([reference_text])
        return predictions, references

##############################
# For Training Dataset
##############################
  
metric = load_metric("bleu")
predictions, references = evaluation(dataset['train'])
print(metric.compute(predictions=[predictions], references=[references]))


##############################
# For Testing Dataset
##############################

predictions, references = evaluation(dataset['test'])
print(metric.compute(predictions=[predictions], references=[references]))
