# spinal_cord_segmentation


## Information
### Measures

- DICE Coefficient
- Jaccard Coefficient (IoU)
- Average Contour Distance (ACD)
- Average Surface Distance (ASD)

### Data Description 
- Spinal-cord Dataset
  - https://www.sciencedirect.com/science/article/pii/S1053811917302185#s0070


### Logger
- Train Logger       : epoch, loss, IoU, Dice, ACD, ASD
- Test Logger        : epoch, loss, IoU, Dice, ACD, ASD


## Getting Started
### Requirements
- Python3 (3.6.8)
- PyTorch (1.2)
- torchvision (0.4)
- NumPy
- pandas
- matplotlib
- medpy
- AdamP
- opencv


### Segmentation - Train Examples
* python3 main_unet.py  --server server_A --exp exp_test --arch unet --source-dataset site2 --optim adam --weight-decay 5e-4 --loss-function bce_logit --batch-size 8  --lr 0.1 --lr-schedule 100 120 --aug-mode True --aug-range aug6 --train-size 0.7 

* python3 main_proposed_embedding.py  --server server_A --exp exp_test --source-dataset site2 --seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss mse --embedding-alpha 1 --optim adam --weight-decay 5e-4 --batch-size 8 --lr 0.1 --lr-schedule 100 120 --aug-mode True --aug-range aug6 --train-size 0.7 

| Args 	| Options 	| Description 	|
|---------|--------|----------------------------------------------------|
| work-dir |  [str] 	| Working folder. 	|
| exp 	| [str] 	| ./test/	|
| arch 	|  [str] 	| model architecture. |
| source-dataset 	|  [str] 	| train-dataset. help='stie1','stie2','stie3','stie4'|
| batch_size 	| [int] 	| number of samples per batch. default : 8|
| arch 	|  [str] 	| model architecture. |
| arch-ae-detach 	|  [str] 	| autoencoder detach setting. default : True |
| embedding-alpha 	|  [float] 	| embedding loss weight. default : 1 |
| optim 	|  [str] 	| optimizer. choices=['adam','adamp','sgd']. default : sgd |
| loss-function 	|  [str] 	| loss-function. |
| lr-schedule | [int] 	| number of epochs for training. default : 100 120 |
| lr 	| [float] 	| learning rate. defalut : 0.1	|
| aug-mode | [str] | augmentation mode :  defalut : False |
| aug-range | [float] | augmentation range. default : aug6 |
| train-size| [float] | train dataset size. default : 0.7 |



### Reference
[1] U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberrger et al.)






