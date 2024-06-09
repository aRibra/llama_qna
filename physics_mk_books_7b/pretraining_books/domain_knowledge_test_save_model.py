import pandas as pd
import torch
from datasets import Dataset, load_dataset
from random import randrange
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from peft import AutoPeftModelForCausalLM

# Load the tokenizer
#model_id = "meta-llama/Llama-2-7b-chat-hf"

model_id = "ilufy/meta-llama-2-7b-mk-physics-domain-tuned"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#finetuned_model_folder = 'llama2-7b-physics-books'
finetuned_model_folder = "llama2-7b-physics-books/checkpoint-2500"
new_model = AutoPeftModelForCausalLM.from_pretrained(
    finetuned_model_folder,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)


# Merge LoRA and base model
merged_model = new_model.merge_and_unload()

# Save the merged model
hf_model_name="meta-llama2-7b-tuned-merged-2500"
merged_model.save_pretrained(hf_model_name, safe_serialization=True)
tokenizer.save_pretrained(hf_model_name)


# Test the model
prompt = "ten dimensions in theoretical physics is"
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
outputs = merged_model.generate(input_ids=input_ids,
                         max_new_tokens=200,
                        #  do_sample=True,
                        #  top_p=0.9,
                         temperature=0.6)

result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

print(result)


# push merged model to the hub
hf_model_repo = "ilufy/meta-llama-2-7b-mk-physics-domain-tuned-2500"
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)

