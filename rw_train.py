import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from rw_model.dataset import ActionGoalDataset
from rw_model.predictor import ActionEvaluator
from rw_model.utils import embed_text, tokenize, get_device
import random

# Load datasets
train_dataset = ActionGoalDataset("./data_rw/train.json")
valid_dataset = ActionGoalDataset("./data_rw/valid.json")
test_dataset = ActionGoalDataset("./data_rw/test.json")

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model, optimizer, and device
model = ActionEvaluator()
optimizer = optim.Adam(model.parameters(), lr=5e-3)
device = get_device()
model.to(device)

num_epochs = 15

def contrastive_loss(pos_sim, neg_sim, margin=0.5):
    """Compute contrastive loss given positive and negative similarities."""
    pos_loss = 1 - pos_sim  # We want positive similarity to be close to 1
    neg_loss = torch.clamp(neg_sim + margin, min=0)  # Negative similarity should be below -margin
    return torch.mean(pos_loss + neg_loss)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    with tqdm(train_loader, unit="batch", total=len(train_loader)) as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
        index = 1

        for batch in train_loader:
            goal = batch["goal"]
            action_text = batch["actions"]

            # Get embeddings (already in [-1, 1] range from Tanh)
            goal_embedding = embed_text(goal).to(device)
            action_embedding = embed_text(action_text).to(device)

            # Model prediction
            output = model(action_embedding, goal_embedding)  # Assume output is in [-1, 1]

            # Normalize embeddings for cosine similarity
            goal_embedding = F.normalize(goal_embedding, p=2, dim=-1)
            output = F.normalize(output, p=2, dim=-1)

            # Generate negative samples (random embeddings in [-1, 1])
            neg_goal_embedding = torch.tanh(torch.randn_like(goal_embedding)).to(device)
            neg_goal_embedding = F.normalize(neg_goal_embedding, p=2, dim=-1)

            # Compute cosine similarities (range [-1, 1])
            pos_sim = F.cosine_similarity(output, goal_embedding)
            neg_sim = F.cosine_similarity(output, neg_goal_embedding)

            # Calculate contrastive loss
            loss = contrastive_loss(pos_sim, neg_sim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=epoch_loss/index)
            tepoch.update(1)
            index += 1

torch.save(model.state_dict(), "./rw_model/checkpoint/checkpoint_ver1.pth")