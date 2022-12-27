# textual_module_CP

## Installation

```
conda create --name textual_module python==3.8
conda env update -n textual_module --file environment.yml
```

## Inference
Link to download pretrained weights for dailydialog dataset
- [Google drive](https://drive.google.com/file/d/1gnjsPCizidd3OXJMN4-cPa2KJ2bxFgQy/view?usp=share_link)
- Unpack model.tar.gz and replace it in {dataset}_models/roberta-large/pretrained/no_freeze/{class}/{sampling}/model.bin
    - dataset: dailydialog, EMORY, iemocap, MELD
    - class: "emotion" or "sentiment"
    - sampling: 0.0 ~ 1.0, default: 1.0
Check ```pipeline.py``` for logics of code usage
