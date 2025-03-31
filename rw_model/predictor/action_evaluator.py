import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, AutoModel, AutoTokenizer

class ActionEvaluator(nn.Module):
    def __init__(self, hidden_dim=1024, num_layers=6, num_heads=8, dropout=0.1):
        super(ActionEvaluator, self).__init__()
        
        self.bert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, 
                                                 dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, scene, goal, actions, mask=None):
        # Tokenize inputs
        scene_inputs = self.tokenizer(scene, return_tensors="pt", padding="max_length", 
                                    truncation=True, max_length=512).to(self.bert.device)
        goal_inputs = self.tokenizer(goal, return_tensors="pt", padding="max_length", 
                                   truncation=True, max_length=512).to(self.bert.device)
        masks = []
        
        # Get embeddings
        scene_emb = self.bert(**scene_inputs).last_hidden_state[:, 0, :]
        goal_emb = self.bert(**goal_inputs).last_hidden_state[:, 0, :]
        
        # Process actions (assuming actions is a list of text strings or list of lists for batched input)
        action_embs = []
        # print("======", actions)
        if isinstance(actions[0], str):  # Single batch case
            # Pad actions and create mask
            mask = torch.zeros(batch_size, 7, dtype=torch.long, device=self.bert.device)
            mask[0, :len(actions)] = 1

            for act in actions:
                act_inputs = self.tokenizer(act, return_tensors="pt", padding="max_length",
                                          truncation=True, max_length=512).to(self.bert.device)
                act_emb = self.bert(**act_inputs).last_hidden_state[:, 0, :]
                action_embs.append(act_emb)
            # print("======", len(action_embs))
            action_embs = torch.stack(action_embs, dim=0)
            # print("++++++", len(action_embs))
        else:  # Batched case
            batch_size = len(actions)
            
            # Pad actions and create mask
            mask = torch.zeros(batch_size, 7, dtype=torch.long, device=self.bert.device)

            for i, batch_actions in enumerate(actions):
                # print("&&&", batch_actions)
                mask[i, :len(batch_actions)] = 1
                batch_actions += [""] * (7 - len(batch_actions))
                act_inputs = self.tokenizer(batch_actions, return_tensors="pt", padding="max_length",
                                          truncation=True, max_length=512).to(self.bert.device)
                act_emb = self.bert(**act_inputs).last_hidden_state[:, 0, :]
                action_embs.append(act_emb)

            # print("======", action_embs)
            action_embs = torch.stack(action_embs, dim=0)
            # print("++++++", action_embs)
        
        # Combine context
        # print("======", len(action_embs))
        context = torch.cat([scene_emb, goal_emb], dim=-1)
        context = context.unsqueeze(1).expand(-1, action_embs.size(1), -1)
        
        combined_input = torch.cat([action_embs, context], dim=-1)[:, :, :1024]
        
        transformer_out = self.transformer_encoder(combined_input, src_key_padding_mask=~mask.bool() if mask is not None else None)
        
        action_probs = self.fc(transformer_out).squeeze(-1)
        
        return action_probs, masks

    def compute_score(self, action_probs, mask, theta=0.8):
        batch_size, seq_len = action_probs.shape
        scores = []
        
        for b in range(batch_size):
            cum_prob = 1.0
            k = 0
            total_actions = mask[b].sum().item()
            
            for t in range(seq_len):
                if mask[b, t] == 0:
                    break
                cum_prob *= action_probs[b, t].item()
                if cum_prob > theta:
                    k += 1
                else:
                    break
            
            score = k / total_actions if total_actions > 0 else 0.0
            scores.append(score)
        
        return torch.tensor(scores, dtype=torch.float32, device=action_probs.device)