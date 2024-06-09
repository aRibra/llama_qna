# Reload the saved model and merge it then we can save the whole model

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


model_chkpt = 'llama2-7b-coaxnn-paper-qna/checkpoint-270'

new_model = AutoPeftModelForCausalLM.from_pretrained(
    model_chkpt,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load the tokenizer
model_id = 'ilufy/meta-llama2-7b-coaxnn-paper-domain-tuned-merged-180'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# q = "How does CoAxNN improve the top-1 accuracy of the ResNet-50 model?"
# q = "What is the purpose of the early exiting strategy in CoAxNN?"
# q = "What is the advantage of CoAxNN compared to other methods?"
q = "What future studies will be explored in CoAxNN?"

prompt = "Question:\n" + q

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()

outputs = new_model.generate(input_ids=input_ids,
                         max_new_tokens=200,
                         temperature=0.6)

result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

print(result)

# Merge LoRA and base model
merged_model = new_model.merge_and_unload()

# Save the merged model
model_folder_name = "llama2-7b-coaxnn-paper-qna-360"
merged_model.save_pretrained(model_folder_name, safe_serialization=True)
tokenizer.save_pretrained(model_folder_name)

# push merged model to the hub
hf_model_repo = "ilufy/llama2-7b-coaxnn-paper-qna-fuinetuned-merged-270"

merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)

