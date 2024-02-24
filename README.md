# Prefix_tuning
This repository is the implementation of prompt based parameter efficient finetuning technique called Prefix tuning

# What is Prefix tuning ?
Prefix tuning, akin to prompt tuning, enhances natural language generation (NLG) tasks on GPT models by appending task-specific vectors to the input. Unlike prompt tuning, which only modifies input embeddings, prefix tuning inserts parameters into all model layers. These parameters are optimized through a separate feed-forward network (FFN), avoiding instability caused by direct training on soft prompts. After updating the soft prompts, the FFN is discarded. Despite having significantly fewer parameters (1000x less), prefix tuning achieves comparable performance to full fine-tuning and outperforms in scenarios with limited data.

## Model
**t5-large:** T5-Large is a variant of the Text-To-Text Transfer Transformer (T5) model, renowned for its impressive performance across various natural language processing tasks. With its large architecture, T5-Large excels in understanding and generating text by leveraging a transformer-based architecture and fine-tuning on extensive datasets, making it a powerful tool for a wide range of language tasks.

## Dataset
**financial_phrasebank/sentence_all_agree:** It is a dataset of 4,840 financial news sentences categorized by sentiment and agreement rate among human annotators. 
sentence all agree is Number of instances with 100% annotator agreement. It consists of approximately 2.26k rows

## Libraries used

- peft: for model pruning and quantization
- transformers: transformers: For utilizing and fine-tuning the model.
- datasets: For handling and processing the data.
- numpy: For numerical computations.
- torch: For building and training neural networks.

## Hyper parameters

- learning rate = 1e-2
- num_epochs = 5
- batch_size = 8

## Usage

```
from peft import PeftModel, PeftConfig

peft_model_id = "likhith231/t5-large_PREFIX_TUNING_SEQ2SEQ"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

inputs = tokenizer(
    "The Lithuanian beer market made up 14.41 million liters in January , a rise of 0.8 percent from the year-earlier figure , the Lithuanian Brewers ' Association reporting citing the results from its members .",
    return_tensors="pt",
)

model.to(device)

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

```
