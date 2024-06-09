

""" BLEU
from datasets import load_metric

bleu_metric = load_metric('bleu')

print(bleu_metric)

# Example reference
ground1 = 'the cat is on the mat'
# Example candidate sentence
candidate = 'the cat the cat on the mat'
# Compute the score with bleu.compute
bleu_scores = bleu_metric.compute(predictions=[candidate.split(' ')],references=[[ground1.split(' ')]])
print(bleu_scores)
"""

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from random import randrange
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

from tqdm import tqdm


df = pd.read_csv("train_coaxnn_paper_text.csv")
test = Dataset.from_pandas(df)
# Select random indices for metric evaluation
np.random.seed(42)
rand_ixs = np.random.randint(0, len(test), len(test))
test = test.select(rand_ixs)
print("Test data: ", test, rand_ixs)


test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

base_chkpnt_path = "llama2-13b-coaxnn-paper/checkpoint-"

model_id = "ilufy/meta-llama2-13b-coaxnn-paper-domain-tuned-merged-180"
#model_id = "meta-llama/Llama-2-13b-hf"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


for chkpnt_iter in list( range(15, 190, 15) ):
    chkpnt_path = base_chkpnt_path + str(chkpnt_iter)

    # Get the type
    compute_dtype = getattr(torch, "float16")

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )


    # Load the pretrained model
    model = AutoModelForCausalLM.from_pretrained(chkpnt_path,
                                             quantization_config=bnb_config,
                                             device_map="auto")


    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")


    # max_length = model.config.max_length
    max_length = 1024
    print("max_length: ", max_length)
    stride = 512
    seq_len = encodings.input_ids.size(1)
    device = "cuda"

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())

    print(f"\n\n[Preplexity/{chkpnt_path}]: ppl = ", ppl)
    print("\n\n")

    # Empty VRAM
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


