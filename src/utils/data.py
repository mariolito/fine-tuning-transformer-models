from collections import namedtuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
from transformers import AutoTokenizer
import torch
import xml.etree.ElementTree as ET


dataset_sample_fields = "input_ids", "token_type_ids", "attention_mask", "labels"
NLIDatasetSample = namedtuple("NLIDatasetSample", ("id", *dataset_sample_fields))
dataset_batch_fields = (*dataset_sample_fields, "sample_lengths")
NLIDatasetBatch = namedtuple("NLIDatasetBatch", ("id", *dataset_batch_fields))


class NLIDataset(Dataset):
    """NLI-based dataset.
    """

    def __init__(self, filename):
        self.label_dict = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
        self.xml_label_dict = {"yes": "entailment", "no": "contradiction", "unknown": "neutral"}
        self.XmlRoot = ET.parse(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", filename)).getroot()
        self.tokenizer = AutoTokenizer.from_pretrained('lighteternal/nli-xlm-r-greek')
        self.samples = self.parse_xml(self.XmlRoot)

    def parse_xml(self, root):
        pairs = root.find('corpus').findall('pair')
        premise_list = []
        hypothesis_list = []
        label_list = []
        for pair in pairs:
            try:
                premice = pair.find('T').text
                hypothesis = pair.find('H').text
                label = self.xml_label_dict[pair.find('entailment').text.lower()]
                premise_list.append(premice)
                hypothesis_list.append(hypothesis)
                label_list.append(label)
            except:
                continue

        data_dict = {}
        c = 0
        for (premise, hypothesis, label) in zip(premise_list, hypothesis_list, label_list):
            data_dict[c] = self.parse(
                premise, hypothesis, label
            )
            c += 1
        return data_dict

    def parse(self, premise, hypothesis, label):
        premise_id = self.tokenizer.encode(premise, add_special_tokens=False)
        hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens=False)
        pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [
            self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
        premise_len = len(premise_id)
        hypothesis_len = len(hypothesis_id)
        # sentence 0 and sentence 1.
        # but RoBERTa doesn’t have token_type_ids so in this case Hypothesis also filled with [0]
        segment_ids = torch.tensor(
            [0] * (premise_len + 2) + [0] * (hypothesis_len + 1))
        attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

        return {
            "input_ids": torch.tensor(pair_token_ids),
            "token_type_ids": segment_ids,
            "attention_mask": attention_mask_ids,
            "labels": self.label_dict[label]

        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset.
        Args:
            idx (int): Index of the sample in the full dataset.
        Returns:
            NLIDatasetSample: Sample from the dataset.
        """
        data_dict = self.samples[idx]

        sample = self.load_sample(data_dict, idx)

        return sample

    def collate_fn(self, data):
        """Create a batch of variable length tensors.
        Args:
            data (list): List of NLIDatasetSample objects.
        Returns:
            NLIDatasetBatch: Batch of samples.
        """
        batch = dict(
            id=[sample.id for sample in data],
            sample_lengths=torch.IntTensor([len(sample.input_ids) for sample in data])
        )

        # pad data such that all samples in the batch have the same length
        for key in dataset_sample_fields:
            if key == "labels":
                batch[key] = torch.tensor([getattr(sample, key) for sample in data])
            else:
                # padding_value = 1. if key == "seg_ids" else 0.
                batch[key] = pad_sequence([getattr(sample, key) for sample in data],
                                      batch_first=True)
        return NLIDatasetBatch(**batch)

    @staticmethod
    def load_sample(data_dict, idx):
        """Loads a sample.

        Args:
            data_dict: Dict data of sample
            idx: index of sample
        Returns:
            NLIDatasetSample: Data converted to a NLIDatasetSample object.
        """
        sample = NLIDatasetSample(
            id=idx,
            input_ids=data_dict["input_ids"],
            token_type_ids=data_dict["token_type_ids"],
            attention_mask=data_dict["attention_mask"],
            labels=data_dict["labels"]
        )
        return sample

    @staticmethod
    def batch_to_device(batch, device):
        """Move batch to device.
        Args:
            batch (NLIDatasetBatch): Batch of samples.
            device (torch.device): Device to move the batch to.
        Returns:
            NLIDatasetBatch: Batch of samples on the device.
        """
        device_tensors = [getattr(batch, key).to(device) for key in dataset_batch_fields]
        return NLIDatasetBatch(batch.id, *device_tensors)
