# ESCF-CFT
## download training data
you can get training data from [this website.](https://drive.google.com/drive/folders/1aQmnPmYGoQIYPK5jgbv4K4PXYYNwqisH)  
this project will use these datasets below:
- empathetic_dialogues
- eSNL
- gigaword
- eli5
- wiki_auto

## prepare training data
Place the data in the corresponding folders according to the diagram structure.

![image1](images/image.png)

Run the Python script for preprocessing as follows:
``` python
cd dataset/rawdata/scripts/
python prepare_trainingdata.py
```

## train model with 5 datasets
Please cd to the root directory of the project and then run the script.
```shell
bash scripts/train-on-5-instruction-task.sh
```

## eval model on instruction tasks
generate answer with base model and tuned model
```shell
bash scripts/generate-instruction-task-on-base-model.sh
bash scripts/generate-instruction-task-on-tuned-model.sh
```

evaluate scores
TODO
