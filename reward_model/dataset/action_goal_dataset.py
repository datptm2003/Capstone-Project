import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import json

class ActionGoalDataset(Dataset):
    def __init__(self, path, tokenizer_name="sentence-transformers/all-MiniLM-L6-v2", max_length=384):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.data = self.load_json_data(path, limit=21024)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        goal, actions = self.data[idx]
        return {
            "goal": goal,
            "actions": ". ".join(actions)
        }

        # action_text = ". ".join(actions)  # Flatten actions into a single string

        # action_tokens = self.tokenizer(action_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        # goal_tokens = self.tokenizer(goal, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        # return {
        #     "input_ids": action_tokens["input_ids"].squeeze(0),
        #     "attention_mask": action_tokens["attention_mask"].squeeze(0),
        #     "goal_ids": goal_tokens["input_ids"].squeeze(0),
        #     "goal_mask": goal_tokens["attention_mask"].squeeze(0),
        #     "goal_text": goal
        # }

    def load_json_data(self, file_path, limit=100000):
        data = []
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        for row in raw_data:
            if len(data) < limit:
                goal = row["goal"]
                actions = row["actions"]
                data.append((goal, actions))
        return data
