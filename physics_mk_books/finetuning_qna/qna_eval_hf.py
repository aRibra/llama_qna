import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


# Get the type
compute_dtype = getattr(torch, "float16")

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)


#hf_model_repo = "meta-llama/Llama-2-7b-chat-hf"
hf_model_repo = "ilufy/meta-llama-2-7b-mk-physics-qna-finetuned-800"


# Get the tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_model_repo)

# Load the model
model = AutoModelForCausalLM.from_pretrained(hf_model_repo,
                                             quantization_config=bnb_config,
                                             device_map="auto")

# prompt = "Question: According to E. T. Bell, what sparked Riemann's early interest in number theory?"
# prompt = "Question: How many watts does the human brain use?"
#prompt = "Question: Who was Einstein's girlfriend?"
#prompt = "Question: What is the vision of Susumu Tachi regarding virtual and real reality?"
#prompt = "Question: Is there an unsolvable problem in Math?"
#prompt = "Question: How can membranes like a beach ball induce resonance frequencies?"
#prompt = "Question: Who is Michio Kaku, and what does he discuss in his book Hyperspace?"
#prompt = "Question: ccording to the author, how will teleconferencing be replaced by telepresence?"
#prompt = "Question: What are the two types of vibrations that exist in the heterotic string?"
#prompt = "Question: Is time travel possible?"
#prompt = "Question: Why time travel may not be possible?"
prompt = "Question: What are the different classes of physics impossibilites?"

# Generate response
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
outputs = model.generate(input_ids=input_ids,
                         max_new_tokens=200,
                         temperature=0.6)

print("outputs: ", outputs)

result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Print the result
print(f"Generated response:\n{result}")


use_transformers_pipeline = False

if use_transformers_pipeline:
    import transformers

    tokenizer = AutoTokenizer.from_pretrained(hf_model_repo,  trust_remote_code=True)
    pipeline = transformers.pipeline(
        "text-generation", #"question-answering",
        model=hf_model_repo,
        trust_remote_code=True
    )

    sequences = pipeline(
        prompt,
        temperature=0.6,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )

    for seq in sequences:
        print(seq['generated_text'])


