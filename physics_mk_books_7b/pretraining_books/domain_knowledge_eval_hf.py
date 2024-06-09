import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


# Get the type
compute_dtype = getattr(torch, "float16")

# Use Hugging Face for inference

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)


hf_model_repo = "ilufy/meta-llama-2-7b-mk-physics-domain-tuned-2500"

#hf_model_repo = "meta-llama/Llama-2-7b-chat-hf"

# Get the tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_model_repo)

# Load the model
model = AutoModelForCausalLM.from_pretrained(hf_model_repo,
                                             quantization_config=bnb_config,
                                             device_map="auto")

#prompt = """
#At this point, it's important to note that some people think that
#because consciousness determines existence, then consciousness can
#therefore control existence, perhaps by meditation. They think that
#"""

prompt = """
They think that we can create reality according to our wishes. This thinking, as attractive as it might sound, goes against quantum mechanics
"""

prompt = """
Their logic, however, was very compelling. Any object, when heated, gradually emits radiation. This is the reason why iron gets red hot when placed in a furnace. The hotter the iron, the higher the frequency of radiation it emits. A precise mathematical formula,
"""

# predict prompt completion
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
outputs = model.generate(input_ids=input_ids,
                         max_new_tokens=200,
                         temperature=0.6)

result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Print the result
print(f"predicted response:\n{result}")


#################################################
#################################################


use_transformer_pipeline = False
if use_transformer_pipeline:
    # Use Transformers Pipeline for Inference
    import transformers

    tokenizer = AutoTokenizer.from_pretrained(hf_model_repo,  trust_remote_code=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model="ilufy/meta-llama-2-7b-mk-physics-domain-tuned-2500",
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



