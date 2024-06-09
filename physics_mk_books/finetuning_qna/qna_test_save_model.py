# Reload the saved model and merge it then we can save the whole model

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from random import randrange
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from peft import AutoPeftModelForCausalLM

model_chkpt = 'llama2-7b-physics-finetuned-qna/checkpoint-800'

new_model = AutoPeftModelForCausalLM.from_pretrained(
    model_chkpt,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load the tokenizer
model_id = 'ilufy/meta-llama-2-7b-mk-physics-domain-tuned-2500'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#prompt = "Question: How does the ten-dimensional theory simplify the laws of nature?"
prompt = "Question: Is time travel possible?"

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()

outputs = new_model.generate(input_ids=input_ids,
                         max_new_tokens=200,
                        #  do_sample=True,
                        #  top_p=0.9,
                         temperature=0.6)

result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

print(result)

# Merge LoRA and base model
merged_model = new_model.merge_and_unload()

# Save the merged model
model_folder_name = "llama2-7b-physics-books-qna-800"
merged_model.save_pretrained(model_folder_name, safe_serialization=True)
tokenizer.save_pretrained(model_folder_name)

# push merged model to the hub
#hf_model_repo = "ilufy/meta-llama-2-7b-mk-physics-qna-finetuned"
hf_model_repo = "ilufy/meta-llama-2-7b-mk-physics-qna-finetuned-800"
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)


