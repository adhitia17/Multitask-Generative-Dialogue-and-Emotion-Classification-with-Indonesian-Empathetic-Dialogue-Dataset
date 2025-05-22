# Multitask-Generative-Dialogue-and-Emotion-Classification-with-Indonesian-Empathetic-Dialogue-Dataset

This research provides a new refresher in the field of emotion-aware dialogue systems in Indonesian by creating the Indonesian Empathetic Dialogue Dataset and conducting multitask text generation and emotion classification training using pretrained idT5

## Original Dataset
EmpatheticDialogue Original Dataset
```
wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
```

## Translated Dataset
Dataset Translated Using NLBB-200-1.3B
```
xxxxxxxxx
```

## Training	Configuration
**Learning Rate**:	1e-5 \
**Weight Decay**:	0.01 \
**Token**:	512 \
**Batch**:	64 \
**Epochs**:	40 \
**Warm Up Steps**:	500 \
**Optimizer**:	Adam \
**Evaluation Metrics**: \
BLEU + ROGUE (text generation) \
Accuracy + F1 (emotion classification) 


## Commands
Download, Normalized, and Translate Dataset
```
python data_preprocessing.py
```

Transform into Dataaset Class
```
python dataaset_class.py
```

Finetune Model
```
python train.py
```

Evaluate Model
```
python eval.py
```
