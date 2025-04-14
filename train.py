import os
import random
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.optim
from tqdm import tqdm

# ==== Config ====
BATCH_SIZE = 256
BERT_DIM = 300
LR = 6e-5
EPOCHS = 10000
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
MODEL_SAVE_PATH = "model"
CHECKPOINT_DIR = "checkpoint"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
s


with open("data/train.csv", "r") as f:
    for i, line in enumerate(f, start=1):
        if len(line.split(",")) != 8:  # Replace 8 with the expected number of fields
            print(f"Inconsistent row at line {i}: {line}")


# ==== Dataset ====
class TripletDataset(Dataset):
    def __init__(self, csv_path):
        # Use on_bad_lines='skip' to skip problematic rows
        self.df = pd.read_csv(csv_path, on_bad_lines='skip')
        self.prompts = self.df["prompt"].astype(str).tolist()
        self.responses = self.df["utterance"].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.prompts[idx]
        positive = self.responses[idx]
        while True:
            neg_idx = random.randint(0, len(self.df) - 1)
            if neg_idx != idx:
                negative = self.responses[neg_idx]
                break
        return anchor, positive, negative

# ==== Model ====
class BertRetrievalModel(nn.Module):
    def __init__(self, projection_dim=300):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.project = nn.Linear(self.bert.config.hidden_size, projection_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.project(cls_embedding)

# ==== Collator ====
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    texts = list(anchors + positives + negatives)
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    return (
        encodings["input_ids"][:len(anchors)],
        encodings["attention_mask"][:len(anchors)],
        encodings["input_ids"][len(anchors):2*len(anchors)],
        encodings["attention_mask"][len(anchors):2*len(anchors)],
        encodings["input_ids"][2*len(anchors):],
        encodings["attention_mask"][2*len(anchors):]
    )

# ==== Load Dataset ====
train_loader = DataLoader(TripletDataset("data/train.csv"), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ==== Initialize ====
model = BertRetrievalModel(projection_dim=BERT_DIM).to(DEVICE)
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ==== Optionally resume checkpoint ====
start_epoch = 0
if os.path.exists(f"{CHECKPOINT_DIR}/encoder.pt"):
    print("🔁 Loading checkpoint...")
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/encoder.pt"))
    optimizer.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/optimizer.pt"))
    with open(f"{CHECKPOINT_DIR}/epoch.txt", "r") as f:
        start_epoch = int(f.read().strip()) + 1

# ==== Training ====
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for a_ids, a_mask, p_ids, p_mask, n_ids, n_mask in progress:
        a_ids, a_mask = a_ids.to(DEVICE), a_mask.to(DEVICE)
        p_ids, p_mask = p_ids.to(DEVICE), p_mask.to(DEVICE)
        n_ids, n_mask = n_ids.to(DEVICE), n_mask.to(DEVICE)

        anchor_embed = model(a_ids, a_mask)
        positive_embed = model(p_ids, p_mask)
        negative_embed = model(n_ids, n_mask)

        loss = criterion(anchor_embed, positive_embed, negative_embed)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    print(f"✅ Epoch {epoch+1} - Loss: {total_loss:.4f}")

    # ==== Save checkpoint ====
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/encoder.pt")
    torch.save(optimizer.state_dict(), f"{CHECKPOINT_DIR}/optimizer.pt")
    with open(f"{CHECKPOINT_DIR}/epoch.txt", "w") as f:
        f.write(str(epoch))

    # Optionally: Save full model
    torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/epoch_{epoch+1}.pt")
