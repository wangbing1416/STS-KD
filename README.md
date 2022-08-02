## STS-KD
This released repo of our work on remote sensing change detection  
_"Siamese Teacher-Strict Knowledge Distillation for Change Detection"_

Note that, our code is based on https://github.com/likyoo/change_detection.pytorch

---

### requirement
- torchvision>=0.5.0
- pretrainedmodels==0.7.4
- efficientnet-pytorch==0.6.3
- timm==0.4.12
- albumentations==1.0.3  

To install requirements, run `pip install -r requirements.txt`.

### dataset preparing

you can refer to README in https://github.com/likyoo/change_detection.pytorch  
datasets should lie in `./data/DATASET_NAME` and are cut into `./data/DATASET_NAME/train`, `./data/DATASET_NAME/val`
and `./data/DATASET_NAME/test`

### training and eval

`cd STS-KD`  
`bash ./run.sh`

### logs and checkpoints

your logs will be recorded in `./log/MODEL_NAME` and checkpoint models are saved in `./checkpoint/bestmodel_UUID`