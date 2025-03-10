import torch
import torch.nn.functional as F

from predictor import ActionEvaluator
from utils import get_device, get_reward

device = get_device()
evaluator = ActionEvaluator().to(device)

checkpoint = torch.load("./checkpoint/checkpoint_ver1.pth", map_location=device)
evaluator.load_state_dict(checkpoint)

# init_state = "You are riding a bike."
actions = [
            "Turn around to face the stove.",
            "Pick up a potato from the counter next to the stove.",
            "Turn around and walk to the microwave.",
            "Place the potato in the microwave.",
            "Turn around and walk to the sink.",
            "Pick up the potato in the sink.",
            "Turn around and walk back to the microwave.",
            "Place the second potato in the microwave."
        ]
goal = "Place two potatoes in the microwave to cook."

# rewards = get_rewards(predictor, init_state, actions, goal)
reward = get_reward(evaluator, actions, goal)

print(f"Reward: {reward}")

# model.eval()