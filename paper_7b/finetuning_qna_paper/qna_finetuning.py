import pandas as pd
import torch
from datasets import Dataset
from random import randrange
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

qna_data = "coaxnn_paper_qna_train.csv"

df = pd.read_csv(qna_data)
print(df)

df['text'] = 'Question:\n' + df['Question'] + '\n\nAnswer:\n' + df['Answer']

#drop columns other than 'text'
df.drop(columns=['Question','Answer'], axis=1, inplace=True)

#convert to Huggingface Datasets format
train = Dataset.from_pandas(df)
test = Dataset.from_pandas(df)

print(train)

model_id = "ilufy/meta-llama2-7b-coaxnn-paper-domain-tuned-merged-180"

# Get the type
compute_dtype = getattr(torch, "float16")

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Load the pretrained model
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map="auto")

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
)

# Define the training arguments. For full list of arguments, check
#https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
batch_size = 8
output_dir_name = 'llama2-7b-coaxnn-paper-qna'
args = TrainingArguments(
    output_dir=output_dir_name,
    num_train_epochs=10, # adjust based on the data size
    max_steps=int( 10 * (308 / batch_size) ),
    per_device_train_batch_size=batch_size, # use 4 if you have more GPU RAM
    save_strategy="steps", #steps
    save_steps=30,# default = 500
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=9, #every half epoch
    learning_rate=2e-4,
    fp16=True,
    seed=42
)


# Create the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train,
    eval_dataset=test,
    dataset_text_field='text',
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=args,
    packing=True,
)


# train
trainer.train()


# Empty VRAM
del model
del trainer
import gc
gc.collect()

torch.cuda.empty_cache()
gc.collect()




torch.cuda.empty_cache()

gc.collect()

