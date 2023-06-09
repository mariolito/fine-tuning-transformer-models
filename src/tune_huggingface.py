from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import numpy as np
import evaluate
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained('lighteternal/nli-xlm-r-greek')
label_dict = {'contradiction': 0, 'entailment': 1, 'neutral': 2}


def parse(premise, hypothesis, label):
    premise_id = tokenizer.encode(premise, padding=True, truncation=True, return_tensors="pt").tolist()[0]
    hypothesis_id = tokenizer.encode(hypothesis, padding=True, truncation=True, return_tensors="pt").tolist()[0]
    pair_token_ids = [tokenizer.cls_token_id] + premise_id + [
        tokenizer.sep_token_id] + hypothesis_id + [tokenizer.sep_token_id]
    pair_token_ids = torch.tensor(pair_token_ids)
    premise_len = len(premise_id)
    hypothesis_len = len(hypothesis_id)
    # sentence 0 and sentence 1.
    # but RoBERTa doesn’t have token_type_ids so in this case Hypothesis also filled with [0]
    segment_ids = torch.tensor(
        [0] * (premise_len + 2) + [0] * (hypothesis_len + 1))
    attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

    pair_token_ids = torch.nn.functional.pad(pair_token_ids, (0, 167 - len(pair_token_ids)))
    segment_ids = torch.nn.functional.pad(segment_ids, (0, 167 - len(segment_ids)))
    attention_mask_ids = torch.nn.functional.pad(attention_mask_ids, (0, 167 - len(attention_mask_ids)))

    return {
        "input_ids": pair_token_ids,
        "token_type_ids": segment_ids,
        "attention_mask": attention_mask_ids,
        "labels": label_dict[label]

    }


def compute_acc_metrics(predictions):
    metric = evaluate.load("accuracy")
    preds = np.argmax(predictions.predictions, axis=-1)
    return metric.compute(predictions=preds, references=predictions.label_ids)


def train(model, train_data, val_data):
    training_args = TrainingArguments(
        output_dir='../model_results',  # output directory
        overwrite_output_dir=True,
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=100,  # batch size per device during training
        per_device_eval_batch_size=100,  # batch size for evaluation
        warmup_steps=1,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=None,  # directory for storing logs
        logging_steps=1
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_data,  # training dataset
        eval_dataset=val_data,
        compute_metrics=compute_acc_metrics
    )

    trainer.train()
    print(trainer.evaluate(val_data))


def process_samples(sample):
    return parse(
        premise=sample['Premise'],
        hypothesis=sample['Hypothesis'],
        label=sample['Label']
    )


def run():
    model = AutoModelForSequenceClassification.from_pretrained('lighteternal/nli-xlm-r-greek')
    datasets = load_dataset("../datasets")
    train_data = datasets["train"].map(process_samples)
    val_data = datasets["validation"].map(process_samples)
    train(model, train_data, val_data)


if __name__ == '__main__':
    run()
