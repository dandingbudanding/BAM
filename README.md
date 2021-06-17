# BAM  (The project is a bit messy, I will clean up the code later）
This project is built from IDN, and thanks for the contributions of all the other researchers those who made their codes accessible.

## Requirements

- PyTorch>=1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

## result

Some of the indicators in the MSRN paper are quite different from the indicators given in the original author's open source code, and the indicators in the open source code are used：https://github.com/MIVRC/MSRN-PyTorch

## structure

![图片](https://user-images.githubusercontent.com/34860373/122319260-f505d480-cf52-11eb-84d4-d601a42295d9.png)

## control exp
![图片](https://user-images.githubusercontent.com/34860373/122320334-a822fd80-cf54-11eb-89f2-43f3cd02c500.png)

![图片](https://user-images.githubusercontent.com/34860373/122319127-c38d0900-cf52-11eb-85c5-b24672dce672.png)

![图片](https://user-images.githubusercontent.com/34860373/122319195-de5f7d80-cf52-11eb-9928-55d33061743d.png)

## visual and metric comparison
![图片](https://user-images.githubusercontent.com/34860373/122319315-09e26800-cf53-11eb-9939-47a37b611e98.png)

## visual comparison
![图片](https://user-images.githubusercontent.com/34860373/122319376-241c4600-cf53-11eb-9474-5948a442f7ac.png)

## metric comparison

![图片](https://user-images.githubusercontent.com/34860373/122320504-ea4c3f00-cf54-11eb-8759-4059a24cff8b.png)


## ablation exp

![图片](https://user-images.githubusercontent.com/34860373/122320164-672ae900-cf54-11eb-9158-6a7a37b3d631.png)


## speed exp

## FPS comparison
![图片](https://user-images.githubusercontent.com/34860373/122320206-7ad64f80-cf54-11eb-8434-6c622ddd2753.png)


## Train

The DIV2K, Set5 dataset converted to HDF5 can be downloaded from the links below.Otherwise, you can use `prepare.py` to create custom dataset.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| DIV2K | 2 | Train | [Download](https://www.dropbox.com/s/41sn4eie37hp6rh/DIV2K_x2.h5?dl=0) |
| DIV2K | 3 | Train | [Download](https://www.dropbox.com/s/4piy2lvhrjb2e54/DIV2K_x3.h5?dl=0) |
| DIV2K | 4 | Train | [Download](https://www.dropbox.com/s/ie4a6t7f9n5lgco/DIV2K_x4.h5?dl=0) |
| Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/b7v5vis8duh9vwd/Set5_x2.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/768b07ncpdfmgs6/Set5_x3.h5?dl=0) |
| Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/rtu89xyatbb71qv/Set5_x4.h5?dl=0) |

The Flickr2K dataset can be downloaded from the links below,and then you can use `prepare.py` to create custom dataset.
https://link.csdn.net/?target=http%3A%2F%2Fcv.snu.ac.kr%2Fresearch%2FEDSR%2FFlickr2K.tar

for preparedata:
python3 prepare.py --images-dir ../../DIV2K/DIV2K_train_HR --output-path ./h5file_DIV2K_train_HR_x4_train --scale 4 --eval False

for train:
python3 train.py --choose_net DRLN_BlancedAttention --train_file ./h5file_mirflickr_train_HR_x3_train --eval_file ./h5file_Set5_x4_test

for eval all SR size && all networks(you should download checkpoints first);
python3 eval_allsize_allnet.py

for eval dingle image:
python3 eval_singleimg.py --lr_image_file ./savedimg/Set5/4/EDSR_blanced_attention_2.png --hr_image_file ../classical_SR_datasets/Set5/Set5/butterfly.png

for infer all size && all networks SR images(the SR images will be saved in the direct ./savedimg/*):
python3 infer_allsize_allnet.py

##checkpoints
We provide all network && all size checkpoints to prove that our experiments are convincing.
you can get them from:https://pan.baidu.com/s/1gy-3jcikT2h-QfRduwoibg
password: 2ubm

If the password fails or any other questions, please contact me:2463908977@qq.com

##Our Attention mechanism is very tiny and efficient, and  has also been proved to be efficient in semantic segmentation missions,especially for light-weight models.
