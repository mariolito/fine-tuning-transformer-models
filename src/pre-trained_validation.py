import torch
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import sys
import os
sys.path.append(os.path.join(".."))
from src.utils.data import NLIDataset


def main():
    model = AutoModelForSequenceClassification.from_pretrained('lighteternal/nli-xlm-r-greek')

    all_labels = []
    all_preds = []

    val_dataset = NLIDataset(filename='val.xml')
    val_loader = DataLoader(val_dataset, batch_size=100,
                            collate_fn=val_dataset.collate_fn)
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            input_ids = sample.input_ids
            attention_mask = sample.attention_mask
            labels = sample.labels.numpy()
            features = {"input_ids": input_ids, "attention_mask": attention_mask}
            scores = model(**features).logits
            preds = [score_max for score_max in scores.argmax(dim=1).numpy()]
            all_labels += list(labels)
            all_preds += list(preds)
    print("Accuracy : {:.2f}".format(accuracy_score(all_labels, all_preds)))


if __name__ == "__main__":
    main()
