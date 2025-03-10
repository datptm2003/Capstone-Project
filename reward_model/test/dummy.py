import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingface_hub

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(device)

tokenizer.pad_token = tokenizer.eos_token

import time

start_time = time.time()

text = \
"""
## Role: You are a reasoning assistant tasked with evaluating and enhancing action sequences for achieving a predefined goal state. Below is the detailed description:

Current state description: "I am sitting on the ground."
Goal state description: "I got the ball."

Action sequence: ["Stand up", "Watch around to find where the ball is", "Go to the position of the ball", "Pick up the ball", "Return back"]
Reward values: [0.23, 0.47, 0.678, 0.98, 0.34]

## Task: Evaluate the quality of the provided action sequence. Consider how closely the resulting states from these actions approach the defined goal state, based on the provided reward values. Then, suggest improvements or alternative strategies to refine the sequence for achieving the goal state more efficiently.

## Output Requirements:

1. A detailed critique of the current action sequence, focusing on:
The effectiveness of individual actions in bringing the state closer to the goal.
Any unnecessary or suboptimal actions that could be replaced or removed.

2. Suggestions for improvement:
Specific actions that could be added, modified, or reordered.
General strategies to achieve higher reward values and a closer approach to the goal state.

## Note: Answer clearly, shortly and try not to repeat yourself.
"""

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)

# Get the model output (last hidden state)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=1024).to(device)

print(tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text):])

end_time = time.time()

print(end_time - start_time)
