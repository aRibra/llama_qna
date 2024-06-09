

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
from tqdm import tqdm
import torch
import evaluate
from datasets import Dataset, load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


data_src = 'coaxnn' # 'wikitext' or 'coaxnn'
metric = 'rouge' # 'rouge' or 'perplexity'
top_n = False

if metric == 'rouge':
    eval_metric = evaluate.load("rouge")
elif metric == 'perplexity':
    eval_metric = load_metric("perplexity")

qna_data = "coaxnn_paper_qna_train.csv"


if data_src == 'coaxnn':
    coaxnn_df = pd.read_csv(qna_data)
    coaxnn_df['text'] = 'Question:\n' + coaxnn_df['Question'] + '\n\nAnswer:\n' + coaxnn_df['Answer']
    coaxnn_df['text_question'] = 'Question:\n' + coaxnn_df['Question']
    coaxnn_df['text_answer'] = 'Answer:\n' + coaxnn_df['Answer']
    #drop columns other than 'text'
    coaxnn_df.drop(columns=['Question','Answer'], axis=1, inplace=True)
    coaxnn_df = coaxnn_df[:256]
    
    print(coaxnn_df)
    # train = Dataset.from_pandas(df)
    # # Select random indices for metric evaluation
    # np.random.seed(42)
    # rand_ixs = np.random.randint(0, len(train), len(train))
    # sample_test = train.select(rand_ixs)

elif data_src == 'wikitext':
    sample_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")


base_chkpnt_path = "llama2-7b-coaxnn-paper-qna/checkpoint-"

#model_id = "meta-llama/Llama-2-7b-chat-hf"
model_id = 'ilufy/llama2-7b-coaxnn-paper-qna-fuinetuned-merged-270'

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




if metric == 'rouge' and data_src == 'coaxnn':
    device = "cuda"
    for chkpnt_iter in list( range(90, 360 + 30, 30) ):
        chkpnt_path = base_chkpnt_path + str(chkpnt_iter)

        # Load the pretrained model
        model = AutoModelForCausalLM.from_pretrained(chkpnt_path,
                                                quantization_config=bnb_config,
                                                device_map="auto")

        references = list(coaxnn_df['text_answer'].values)
        predictions = []

        for input_question, gt_answer in tqdm(zip(coaxnn_df['text_question'], coaxnn_df['text_answer'])):
        
            #list(df['text_question'].values)
        
            if top_n:
                N = 3
                rouge_type = 'rougeL'
                
                best_pred_str = None
                best_pred_rougeL = 0
                
                for i in range(N):
                    input_ids = tokenizer(input_question, truncation=True, return_tensors='pt').input_ids.to(device)
                    
                    with torch.no_grad():
                        outputs = model.generate(input_ids=input_ids,
                                            max_new_tokens=200,
                                            temperature=0.6)

                    # gen_tokens; exclude input tokens from the final decoded output
                    gen_tokens = outputs[:, input_ids.shape[1]:]
                    result = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

                    rougeL = eval_metric.compute(predictions=[result], references=[gt_answer])[rouge_type]

                    if rougeL > best_pred_rougeL:
                        best_pred_str = result

                predictions.append(best_pred_str)
            
            else:
                input_ids = tokenizer(input_question, truncation=True, return_tensors='pt').input_ids.to(device)
                
                with torch.no_grad():
                    outputs = model.generate(input_ids=input_ids,
                                        max_new_tokens=200,
                                        temperature=0.6)

                # gen_tokens; exclude input tokens from the final decoded output
                gen_tokens = outputs[:, input_ids.shape[1]:]
                result = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

                predictions.append(result)

        rougel = eval_metric.compute(predictions=predictions, references=references)
        
        print(f"\n\n[RougeL/{chkpnt_path}]: rougel = ", rougel)
        print("\n\n")

        # Empty VRAM
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        


elif metric == 'perplexity':

    for chkpnt_iter in list( range(30, 360 + 30, 30) ):
        chkpnt_path = base_chkpnt_path + str(chkpnt_iter)

        # Load the pretrained model
        model = AutoModelForCausalLM.from_pretrained(chkpnt_path,
                                                quantization_config=bnb_config,
                                                device_map="auto")

        encodings = tokenizer("\n\n".join(sample_test["text"]), return_tensors="pt")


        #max_length = model.config.max_length
        max_length = 1024
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


