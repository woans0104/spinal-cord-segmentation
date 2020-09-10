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


### Segmentation - Train Examples
* python3 main.py  --exp test --train-site site2 --test-site1 site1 --test-site2 site3 --test-site3 site4 --arch unet --loss-function bce --input-size 128 --batch-size 8 --lr-schedule 100 150 --arg-mode True --arg-thres 0.5 --initial-lr 0.1 --train-size 0.7



```
python3 main.py  \
--exp test \
--train-site site2 \
--test-site1 site1 \
--test-site2 site3 \
--test-site3 site4 \
--arch unet \
--loss-function bce \
--input-size 128 \
--batch-size 8 \
--lr-schedule 100 150 \
--arg-mode True \
--arg-thres 0.5 \
--initial-lr 0.1 \
--train-size 0.7

```
| Args 	| Options 	| Description 	|
|---------|--------|----------------------------------------------------|
| work-dir |  [str] 	| Working folder. 	|
| exp 	| [str] 	| ./test/	|
| arch 	|  [str] 	| model architecture. |
| train-site 	|  [str] 	| train-dataset. help='stie1','stie2','stie3','stie4'|
| test-site1 	|  [str] 	| test-dataset. help='stie1','stie2','stie3','stie4'|
| test-site2  |  [str] 	| test-dataset. help='stie1','stie2','stie3','stie4'|
| test-site3	  |  [str] 	| test-dataset. help='stie1','stie2','stie3','stie4'|
| loss-function	  |  [str] 	| segmentation loss function. |
| input_size 	| [int] 	| Size of data. default : 128|
| batch_size 	| [int] 	| number of samples per batch. default : 8|
| lr-schedule | [int] 	| number of epochs for training. default : 100 120 |
| initial-lr 	| [float] 	| learning rate. defalut : 0.1	|
| arg-mode | [str] | augmentation mode :  defalut : False|
| arg-thres | [float] | augmentation threshold. default : 0.5|
| train-size| [float] | train dataset size. default : 0.8 |



### Reference
[1] U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberrger et al.)






