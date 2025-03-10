import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader

# Configurations
TRAIN_FILE = "./data/train.jsonl"
MODEL_DIR = "./sft_llm"
# REWARD_MODEL_DIR = "./reward_model"
OUTPUT_MODEL_DIR = "./rlhf_llm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl(TRAIN_FILE)

batch_size = 4

def collate_fn(batch):
    """Collates batch samples into tensors with correct tokenization & padding."""
    prompt = [example["prompt"] for example in batch]  # Truncate input
    response = [example["response"] for example in batch]
    return {
        "prompt": prompt, 
        "response": response
    }

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

# Quantization config for 4-bit or 8-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Change to load_in_8bit=True for 8-bit quantization
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # Normalized float-4 quantization
)

# Load SFT policy model with quantization
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map=DEVICE  # Automatically distribute layers to GPUs
)

# Load reward model (MLP)
class RewardModel(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Output a single score

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# reward_model = RewardModel(input_size=768).to(DEVICE)  # Assuming embeddings have size 768
# reward_model.load_state_dict(torch.load(f"{REWARD_MODEL_DIR}/reward_model.pth"))
# reward_model.eval()

# PPO Configuration
config = PPOConfig(
    model_name=MODEL_DIR,
    learning_rate=1e-6,
    batch_size=batch_size,
    mini_batch_size=2,
    gradient_accumulation_steps=4
)

# Initialize PPO Trainer
ppo_trainer = PPOTrainer(config, model, ref_model=None, tokenizer=tokenizer)
num_epochs = 3

# Training loop
for epoch in range(num_epochs):  # Train for 3 epochs
    random.shuffle(train_data)
    with tqdm(train_loader, unit="batch", total=len(train_loader)) as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

        for batch in train_loader:
            prompt = batch["prompt"]
            # print("----", len(prompt))

            # Encode prompt
            encoded_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
            query_tensor = encoded_prompt.input_ids.to(DEVICE)
            attention_mask = encoded_prompt.attention_mask.to(DEVICE)

            # logits = model(query_tensor, attention_mask)
            # # Check for NaN or Inf in logits
            # print("Logits:", logits)
            with torch.no_grad():  # No gradient computation for inference
                logits = model(
                    input_ids=query_tensor,
                    attention_mask=attention_mask,
                    return_dict=True  # Ensure outputs are returned as a dict
                )

            # print("---", query_tensor)
            # print("***", attention_mask)

            # Generate model response
            response_tensor = model.generate(query_tensor, attention_mask=attention_mask, max_length=512, pad_token_id=tokenizer.eos_token_id, do_sample=True)

            # response_text = tokenizer.decode(response_tensor[0], skip_special_tokens=True)

            # # Tokenize response for reward model
            # response_embedding = tokenizer(response_text, return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids.to(DEVICE).float()

            # Compute reward score
            with torch.no_grad():
                # reward_value = reward_model(response_embedding).item()
                reward_value = [torch.tensor([0.5]) for _ in range(batch_size)]

            # Optimize using PPO
            ppo_trainer.step(list(query_tensor), list(response_tensor), reward_value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            tepoch.update(1)

# Save trained model
model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print(f"RLHF fine-tuned model saved to {OUTPUT_MODEL_DIR}")
