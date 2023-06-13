# Fine-tuning Cross-Encoder for Greek Natural Language Inference (lighteternal/nli-xlm-r-greek)

This project includes a natural language inference (NLI) model, developed
by fine-tuning a Sentence-BERT pre-trained model on a Greek Textual Entailment Corpus dataset [link](https://inventory.clarin.gr/corpus/689). 

The purpose of this project is to present fine-tuning examples of HuggingFace models, using PyTorch or transformers libraries.   

## Dependencies
* Install anaconda3

* Activate virtual enviroment
```angular2
sudo pip install --upgrade virtualenv
mkdir venvs
virtualenv my_venv
source my_venv/bin/activate
```

* Install python libraries
```angular2
pip install -r requirements.txt
```



## Data
Greek Textual Entailment Corpus dataset can be found in `datasets`. 
Data is already split into train (507 samples) and validation (92 samples) data, where `xml` and `json` format is provided. 

## Model
Model documentation could be found [here](https://huggingface.co/lighteternal/nli-xlm-r-greek)

Usage example:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('lighteternal/nli-xlm-r-greek')
scores = model.predict(
    [
        ('Δύο άνθρωποι συναντιούνται στο δρόμο', 'Ο δρόμος έχει κόσμο'),
        ('Ένα μαύρο αυτοκίνητο ξεκινάει στη μέση του πλήθους.', 'Ένας άντρας οδηγάει σε ένα μοναχικό δρόμο'),
        ('Δυο γυναίκες μιλάνε στο κινητό', 'Το τραπέζι ήταν πράσινο'),
        ('Γιγάντια κύματα που προκλήθηκαν από ισχυρούς ανέμους έξω από'
         'τη Νότιο Αφρική έπληξαν μεγάλο μέρος της Ινδονησίας, των Μαλδίβων, της Ταϊλάνδης και της Δυτικής Αυστραλίας',
         'Γιγάντια κύματα έπληξαν τις ακτές της Νοτιοανατολικής Ασίας')
    ]
)


#Convert scores to labels
label_mapping = ['contradiction', 'entailment', 'neutral']
labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
print(scores, labels)
```

## Pre-trained Validation
* Calculate the accuracy of pre-trained model (without fine-tuning) on the validation dataset. 

Accuracy: 0.49

```shell script
cd src
python test_trained_model.py
```

## HuggingFace fine-tuning
* Fine-tuning model, using HuggingFace transformers libraries.
* Calculate the accuracy of fine-tuned model on the validation dataset. 

Accuracy: 0.56

```shell script
cd src
python tune_huggingface.py
```


## PyTorch fine-tuning
* Fine-tuning model, using PyTorch libraries.
* Calculate the accuracy of fine-tuned model on the validation dataset. 

Accuracy: 0.60

```shell script
cd src
python tune_huggingface.py
```