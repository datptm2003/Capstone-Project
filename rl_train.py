import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader

from app.services import llm_service
from prompts import rl_prompt, sft_inst
from rw_model.predictor import ActionEvaluator
from rw_model.utils import embed_text

# Configurations
TRAIN_FILE = "./more_data/dataset.json"
MODEL_DIR = "./other_sft/checkpoint"
REWARD_MODEL_DIR = "./rw_model/checkpoint"
OUTPUT_MODEL_DIR = "./rlhf_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
# def load_jsonl(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return [json.loads(line) for line in f]

# train_data = load_jsonl(TRAIN_FILE)
with open("more_data/dataset.json", "r") as f:
    train_data = json.load(f)

batch_size = 4

def collate_fn(batch):
    """Collates batch samples into tensors with correct tokenization & padding."""
    scene = [example["scene"] for example in batch]
    goal = [example["goal"] for example in batch]
    actions = [example["actions"] for example in batch]
    feedback = [example["feedback"] for example in batch]
    score = [example["score"] for example in batch]
    return {
        "scene": scene,
        "goal": goal,
        "actions": actions,
        "feedback": feedback,
        "score": score
    }

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
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

reward_model = ActionEvaluator().to(DEVICE)  # Assuming embeddings have size 768
reward_model.load_state_dict(torch.load(f"{REWARD_MODEL_DIR}/action_evaluator.pth"))
reward_model.eval()

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
            scene = batch["scene"]
            goal = batch["goal"]
            actions = batch["actions"]

            # print(":", scene, len(scene))
            # print(":", goal, len(scene))
            # print(":", actions, len(scene))

            # Encode prompt
            encoded_prompt = tokenizer([rl_prompt.format(system_message=sft_inst, scene=s, goal=g, actions=a) for s, g, a in zip(scene, goal, actions)], return_tensors="pt", truncation=True, padding=True, max_length=512)
            query_tensor = encoded_prompt.input_ids.to(DEVICE)
            attention_mask = encoded_prompt.attention_mask.to(DEVICE)
            # print(encoded_prompt, len(encoded_prompt))

            # with torch.no_grad():  # No gradient computation for inference
            #     logits = model(
            #         input_ids=query_tensor,
            #         attention_mask=attention_mask,
            #         return_dict=True  # Ensure outputs are returned as a dict
            #     )

            # Generate model response
            response_tensor = model.generate(
                **encoded_prompt,
                max_length=512, 
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False,
                temperature=0.7,
            )
            # print("===", response_tensor)

            feedback = tokenizer.batch_decode(response_tensor, skip_special_tokens=True)
            # print(":::", feedback, len(feedback))
            new_actions = [(llm_service.call(u_scene, u_goal, u_actions, u_feedback.split('[/INST]')[-1])) for u_scene, u_goal, u_actions, u_feedback in zip(scene, goal, actions, feedback)]
            
            # print(":", new_actions, len(new_actions))

            # Compute reward score
            with torch.no_grad():
                # reward_value = reward_model(response_embedding).item()
                scene_goal_input = \
"""
**Scene**: {scene}
**Goal**: {goal}
"""
                # scene_goal_embedding = embed_text(scene_goal_input.format(scene=scene, goal=goal))
                # action_embedding = embed_text(actions)
                # reward_value = reward_model(scene_goal_embedding, action_embedding)
                reward_value, masks = reward_model(scene, goal, new_actions)
                reward_value = [torch.mean(reward) for i, reward in enumerate(reward_value)]
                # print("------", reward_value)

            # Optimize using PPO
            ppo_trainer.step(list(query_tensor), list(response_tensor), list(reward_value))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            tepoch.update(1)

# Save trained model
model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print(f"RLHF fine-tuned model saved to {OUTPUT_MODEL_DIR}")
