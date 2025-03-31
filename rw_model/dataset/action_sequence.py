from torch.utils.data import Dataset, DataLoader
import torch

class ActionSequenceDataset(Dataset):
    def __init__(self, data, max_length=128):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        scene = sample["scene"]
        goal = sample["goal"]
        actions = sample["actions"]
        score = sample["score"]

        n_actions = len(actions)
        k = int(score * n_actions)
        action_labels = [1] * k + [0] * (n_actions - k)

        return {
            "scene": scene,
            "goal": goal,
            "actions": actions,
            "action_labels": torch.tensor(action_labels, dtype=torch.float32),
            "score": torch.tensor(score, dtype=torch.float32),
            "n_actions": n_actions
        }