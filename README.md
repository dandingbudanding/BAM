# BAM  (The project is a bit messy, I will clean up the code later）
This project is built from IDN, and thanks for the contributions of all the other researchers those who made their codes accessible.

## Requirements

- PyTorch>=1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

##result

Some of the indicators in the MSRN paper are quite different from the indicators given in the original author's open source code, and the indicators in the open source code are used：https://github.com/MIVRC/MSRN-PyTorch

![图片](https://user-images.githubusercontent.com/34860373/120951748-2248ca80-c77c-11eb-8b6e-ea51b7e28705.png)



![manga109comparison_](https://user-images.githubusercontent.com/34860373/120927948-49b87c80-c715-11eb-9968-9cd61ac24f30.jpg)


![Fig8-1](https://user-images.githubusercontent.com/34860373/115811481-5cefe100-a422-11eb-9ced-0e2f02d47915.jpg)


FPS comparison
![IMDN_fps_all](https://user-images.githubusercontent.com/34860373/120927300-107f0d00-c713-11eb-9eb0-44ccc4f3ed57.png)
![DRLN_fps_all](https://user-images.githubusercontent.com/34860373/120927305-1379fd80-c713-11eb-9db1-aa97cf6e3df7.png)


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
