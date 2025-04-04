import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from prompts import sft_prompt, sft_inst

DATA_PATH = './data/'

### Modify the path in each stage ###
model_path = 'C:/Users/PC/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590' # path to the LLM to be finetuned
save_path = './other_sft/checkpoint' # path to save the finetuned model

new_model = "./other_sft/sft_model"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 5
max_seq_length = None
packing = False
device_map = {"": 0}


# def split_data(data_file, train_file, test_file, train_ratio=0.9):

#     with open(data_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     train_size = int(len(lines) * train_ratio)
#     train_data = lines[:train_size]
#     test_data = lines[train_size:]

#     with open(train_file, 'w', encoding='utf-8') as f:
#         for item in train_data:
#             f.write(item)

#     with open(test_file, 'w', encoding='utf-8') as f:
#         for item in test_data:
#             f.write(item)

# split_data(DATA_PATH + 'train.jsonl', DATA_PATH + 'train.jsonl', DATA_PATH + 'valid.jsonl')


# Load datasets
train_dataset = load_dataset('json', data_files=DATA_PATH + 'train.jsonl', split="train")
valid_dataset = load_dataset('json', data_files=DATA_PATH + 'valid.jsonl', split="train")

# valid_actions = """Available actions:
# [
#     "idle(isCombat=<bool>)"
#     "walk(target=<str>)"
#     "run(target=<str>)"
#     "jump()"
#     "roll(direction=<int>)"
#     "punch(target=<str>)"
#     "melee(target=<str>)" ## only valid with melee weapon on hand
#     "shoot(target=<str>)" ## only valid with range weapon on hand
#     "magic(target=<str>)" ## only valid with magic weapon on hand
#     "block()"
#     "pickup(item=<str>)"
#     "consume(item=<str>)"
#     "talk(dialog=<str>)"
#     "open(item=<str>)"
#     "close(item=<str>)"
#     "check_status(target=<str>)"
#     "use(item=<str>)" ## only valid with household equipment on hand
# ]
# """
# valid_actions = ""

# Preprocess datasets
def format_prompt(example):
    return {
        'text': [
            sft_prompt.format(system_message=sft_inst, scene=scene, goal=goal, actions=actions, feedback=feedback)
            for scene, goal, actions, feedback in zip(example['scene'], example['goal'], example['actions'], example['feedback'])
        ]
    }

train_dataset_mapped = train_dataset.map(format_prompt, batched=True)
valid_dataset_mapped = valid_dataset.map(format_prompt, batched=True)
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="all",
    evaluation_strategy="steps",
    eval_steps=5  # Evaluate every 20 steps
)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_mapped,
    eval_dataset=valid_dataset_mapped,  # Pass validation dataset here
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()
trainer.model.save_pretrained(new_model)

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Save the merged model
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)