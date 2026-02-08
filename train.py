import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
from huggingface_hub import login


load_dotenv()
login(token=os.getenv("HF_TOKEN"))


MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading Model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BNB_CONFIG,
    torch_dtype=torch.bfloat16,             
    device_map="auto",
)

model.config.use_cache = False
model.config.pretraining_tp = 1

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# Set padding token to eos_token for batch processing if not already set
if tokenizer.pad_token is None:
    print("Setting pad_token to eos_token for batch processing.")
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

dataset = load_dataset("json", data_files="processed_data/nust_llama3_chat_format.jsonl", split="train")

def formatting_prompts_func(example):
    output_texts = []
    raw_messages = example['messages']
    
    # Handle Batch vs Single
    batch = raw_messages if isinstance(raw_messages[0], list) else [raw_messages]

    for conversation in batch:
        # Using the Universal Template applier is safer than hardcoding strings
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)
    return output_texts

# 7. TRAINER SETUP
sft_config = SFTConfig(
    output_dir="/workspace/nust_bank_adapter",  # SAVE TO PERSISTENT STORAGE
    max_length=2048,
    per_device_train_batch_size=4,        # Increased for RunPod GPUs
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    fp16=False,                           # OFF
    bf16=True,                            # ON (Modern GPU Standard)
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    report_to="none",
    remove_unused_columns=False,
    packing=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
)

print("Starting Training...")
trainer.train()

print("Saving Adapter...")
trainer.save_model("/workspace/nust_bank_adapter")
print("Done!")