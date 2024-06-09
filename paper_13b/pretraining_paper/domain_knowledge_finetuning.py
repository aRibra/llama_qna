import pandas as pd
import torch
from datasets import Dataset, load_dataset
from random import randrange
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

df = pd.read_csv("train_coaxnn_paper_text.csv")

train = Dataset.from_pandas(df)
test = Dataset.from_pandas(df)

# after getting access permissions from Meta:
# (Gated model You have been granted access to this model)
model_id = "meta-llama/Llama-2-13b-chat-hf"

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
batch_size = 2
output_dir_name = 'llama2-13b-coaxnn-paper'
args = TrainingArguments(
    output_dir=output_dir_name,
    num_train_epochs=10, # adjust based on the data size
    max_steps=int( 10 * (len(train) / batch_size) ),
    per_device_train_batch_size=batch_size, # use 4 if you have more GPU RAM
    save_strategy="steps", #steps
    save_steps=15,# default = 500
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

trainer.train()

# save model in local
trainer.save_model()

# Empty VRAM
del model
del trainer
import gc
gc.collect()
gc.collect()

torch.cuda.empty_cache()

gc.collect()


from peft import AutoPeftModelForCausalLM

new_model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)


# Merge LoRA and base model
merged_model = new_model.merge_and_unload()

# Save the merged model
model_name="meta-llama2-13b-coaxnn-paper-domain-tuned-merged"
merged_model.save_pretrained(model_name, safe_serialization=True)
tokenizer.save_pretrained(model_name)


# Test the model
prompt = "Question: What is CoAxNN framework?"
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
outputs = merged_model.generate(input_ids=input_ids,
                         max_new_tokens=200,
                        #  do_sample=True,
                        #  top_p=0.9,
                         temperature=0.6)

result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

print(result)


# push merged model to the hub
hf_model_repo = "ilufy/meta-llama2-13b-coaxnn-paper-domain-tuned-merged"
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)

