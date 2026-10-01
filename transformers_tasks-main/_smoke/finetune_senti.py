# -*- coding: utf-8 -*-
"""Quick fine-tune of a 2-class sentiment classifier used as the PPO reward proxy."""
import os
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          default_data_collator, get_scheduler)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "_smoke", "models", "senti_2cls")
DST = os.path.join(ROOT, "_smoke", "models", "senti_ft")
RM_TRAIN = os.path.join(ROOT, "_smoke", "data", "rm_train.tsv")
os.makedirs(DST, exist_ok=True)

MAX_SEQ = 64
BATCH = 8
EPOCHS = 3
DEVICE = "cpu"
LR = 3e-4

def load_pairs():
    texts, labels = [], []
    with open(RM_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                texts.append(parts[0]); labels.append(1)      # 排在最前 = 最正向
                texts.append(parts[-1]); labels.append(0)     # 排在最后 = 最负向
    return texts, labels

def main():
    tokenizer = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC, num_labels=2)
    model.to(DEVICE)
    texts, labels = load_pairs()
    print("pairs:", len(texts))
    enc = tokenizer(texts, truncation=True, max_length=MAX_SEQ, padding="max_length")
    enc["labels"] = labels
    dataset = [{k: torch.tensor(v[i]) for k, v in enc.items()} for i in range(len(texts))]
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=default_data_collator)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps_per_epoch = (len(dataset) + BATCH - 1) // BATCH
    sch = get_scheduler("linear", optimizer=opt,
                        num_warmup_steps=0, num_training_steps=EPOCHS * steps_per_epoch)
    crit = torch.nn.CrossEntropyLoss()
    model.train()
    gstep = 0
    for ep in range(EPOCHS):
        total = 0.0
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(DEVICE),
                        token_type_ids=batch["token_type_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE))
            loss = crit(out.logits, batch["labels"].to(DEVICE))
            loss.backward(); opt.step(); sch.step(); opt.zero_grad()
            total += float(loss)
            gstep += 1
            if gstep % 20 == 0:
                print("step", gstep, "loss", round(float(loss), 4))
        print("epoch", ep, "avg_loss", round(total / steps_per_epoch, 4))
    model.save_pretrained(DST)
    tokenizer.save_pretrained(DST)
    print("saved to", DST)

if __name__ == "__main__":
    main()
