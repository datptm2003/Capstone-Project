from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
max_length = 384

def embed_text(text):
    """Embed a text string into a vector using the model."""
    # Tokenize input text and convert to PyTorch tensors
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
    
    # Get the model output (last hidden state)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the mean pooling of the token embeddings as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def tokenize(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)

    return {
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
    }
        
def get_reward(evaluator, actions, goal):
    action_text = ", ".join(actions)  # Flatten actions into a single string

    action_tokens = tokenizer(action_text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    goal_logits = evaluator(embed_text(action_text), embed_text(goal))

    # return torch.sigmoid((goal_logits * embed_text(goal)).sum(dim=-1))
    # print(goal_logits)
    # print("---------------")
    # print(embed_text(goal))

    goal_logits = F.normalize(goal_logits, p=2, dim=-1)
    goal_embedding = F.normalize(embed_text(goal), p=2, dim=-1)

    pos_sim = F.cosine_similarity(goal_logits, goal_embedding)

    # return nn.MSELoss()(goal_logits, goal_embedding)
    return pos_sim



def get_device():
    return device

if __name__ == "__main__":
    print(model)