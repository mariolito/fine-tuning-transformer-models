import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW
import os
import sys
import time
sys.path.append(os.path.join(".."))
from src.utils.data import NLIDataset


device = 'cpu'
EPOCHS = 3


def multi_acc(y_pred, y_test):
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
    return acc


def train(model, train_loader, val_loader, optimizer):
    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch_idx, sample in enumerate(train_loader):
            input_ids = sample.input_ids.to(device)
            attention_mask = sample.attention_mask.to(device)
            token_type_ids = sample.token_type_ids.to(device)
            labels = sample.labels.to(device)
            optimizer.zero_grad()

            loss, prediction = model(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask,
                                     labels=labels).values()

            acc = multi_acc(prediction, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += acc.item()

        train_acc = total_train_acc / len(train_loader)
        train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_val_acc = 0
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, sample in enumerate(val_loader):
                input_ids = sample.input_ids.to(device)
                attention_mask = sample.attention_mask.to(device)
                token_type_ids = sample.token_type_ids.to(device)
                labels = sample.labels.to(device)
                optimizer.zero_grad()

                loss, prediction = model(input_ids=input_ids,
                                         token_type_ids=token_type_ids,
                                         attention_mask=attention_mask,
                                         labels=labels).values()

                acc = multi_acc(prediction, labels)

                total_val_loss += loss.item()
                total_val_acc += acc.item()

        val_acc = total_val_acc / len(val_loader)
        val_loss = total_val_loss / len(val_loader)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        print(
            f'Epoch {epoch + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def run():
    model = AutoModelForSequenceClassification.from_pretrained('lighteternal/nli-xlm-r-greek', num_labels=3)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-2,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)

    train_dataset = NLIDataset(filename='train.xml')
    train_loader = DataLoader(train_dataset, batch_size=150,
                            collate_fn=train_dataset.collate_fn)

    val_dataset = NLIDataset(filename='val.xml')
    val_loader = DataLoader(val_dataset, batch_size=150,
                            collate_fn=val_dataset.collate_fn)

    train(model, train_loader, val_loader, optimizer)


if __name__ == '__main__':
    run()
