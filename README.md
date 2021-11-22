Identity Recognition
This facial recognition software is capable of determining whether a human is wearing a facemask or is not wearing a facemask

Download Dataset from Kaggle.com and import into proper folders
https://www.kaggle.com/deepakat002/face-mask-detection-yolov5

[ ]
## importing required libraries
import os
import shutil
import random
import cv2
[1]
23s
## connecting to Google Drive

from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
[ ]
# ## unzipping the data
# !unzip /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_mask_detection_yolov5.zip
[ ]
train_path = "/content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/train/images"
val_path = "/content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/val/images"
test_path = "/content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test"
Open dataset.yaml file and change the paths to train and val images
cont.JPG

Clone yolov5 repo "https://github.com/ultralytics/yolov5.git"
[ ]
!git clone https://github.com/ultralytics/yolov5.git
Cloning into 'yolov5'...
remote: Enumerating objects: 9884, done.
remote: Counting objects: 100% (83/83), done.
remote: Compressing objects: 100% (72/72), done.
remote: Total 9884 (delta 43), reused 30 (delta 11), pack-reused 9801
Receiving objects: 100% (9884/9884), 10.32 MiB | 21.84 MiB/s, done.
Resolving deltas: 100% (6835/6835), done.
[ ]
### change the dir to yolov5
%cd yolov5/
/content/yolov5
[ ]
### install all requirements 

!pip install -r requirements.txt
Requirement already satisfied: matplotlib>=3.2.2 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 4)) (3.2.2)
Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 5)) (1.19.5)
Requirement already satisfied: opencv-python>=4.1.2 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 6)) (4.1.2.30)
Requirement already satisfied: Pillow>=7.1.2 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 7)) (7.1.2)
Collecting PyYAML>=5.3.1
  Downloading PyYAML-6.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (596 kB)
     |████████████████████████████████| 596 kB 4.1 MB/s 
Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 9)) (2.23.0)
Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 10)) (1.4.1)
Requirement already satisfied: torch>=1.7.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 11)) (1.10.0+cu111)
Requirement already satisfied: torchvision>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 12)) (0.11.1+cu111)
Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 13)) (4.62.3)
Requirement already satisfied: tensorboard>=2.4.1 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 16)) (2.7.0)
Requirement already satisfied: pandas>=1.1.4 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 20)) (1.1.5)
Requirement already satisfied: seaborn>=0.11.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 21)) (0.11.2)
Collecting thop
  Downloading thop-0.0.31.post2005241907-py3-none-any.whl (8.7 kB)
Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.2->-r requirements.txt (line 4)) (2.8.2)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.2->-r requirements.txt (line 4)) (2.4.7)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.2->-r requirements.txt (line 4)) (0.11.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.2->-r requirements.txt (line 4)) (1.3.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.23.0->-r requirements.txt (line 9)) (1.24.3)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.23.0->-r requirements.txt (line 9)) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.23.0->-r requirements.txt (line 9)) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.23.0->-r requirements.txt (line 9)) (2021.10.8)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch>=1.7.0->-r requirements.txt (line 11)) (3.10.0.2)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r requirements.txt (line 16)) (1.8.0)
Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r requirements.txt (line 16)) (1.35.0)
Requirement already satisfied: protobuf>=3.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r requirements.txt (line 16)) (3.17.3)
Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r requirements.txt (line 16)) (0.12.0)
Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r requirements.txt (line 16)) (57.4.0)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r requirements.txt (line 16)) (3.3.4)
Requirement already satisfied: grpcio>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r requirements.txt (line 16)) (1.41.1)
Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r requirements.txt (line 16)) (1.0.1)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r requirements.txt (line 16)) (0.4.6)
Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r requirements.txt (line 16)) (0.6.1)
Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r requirements.txt (line 16)) (0.37.0)
Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=1.1.4->-r requirements.txt (line 20)) (2018.9)
Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from absl-py>=0.4->tensorboard>=2.4.1->-r requirements.txt (line 16)) (1.15.0)
Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.4.1->-r requirements.txt (line 16)) (4.7.2)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.4.1->-r requirements.txt (line 16)) (0.2.8)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.4.1->-r requirements.txt (line 16)) (4.2.4)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.4.1->-r requirements.txt (line 16)) (1.3.0)
Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard>=2.4.1->-r requirements.txt (line 16)) (4.8.2)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard>=2.4.1->-r requirements.txt (line 16)) (0.4.8)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.4.1->-r requirements.txt (line 16)) (3.1.1)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->markdown>=2.6.8->tensorboard>=2.4.1->-r requirements.txt (line 16)) (3.6.0)
Installing collected packages: thop, PyYAML
  Attempting uninstall: PyYAML
    Found existing installation: PyYAML 3.13
    Uninstalling PyYAML-3.13:
      Successfully uninstalled PyYAML-3.13
Successfully installed PyYAML-6.0 thop-0.0.31.post2005241907
Download pre-trained weight "yolov5s.pt"
[ ]
!wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
--2021-11-12 00:03:03--  https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
Resolving github.com (github.com)... 13.114.40.48
Connecting to github.com (github.com)|13.114.40.48|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://github-releases.githubusercontent.com/264818686/eab38592-7168-4731-bdff-ad5ede2002be?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20211112%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20211112T000303Z&X-Amz-Expires=300&X-Amz-Signature=372988e1039e0a4b44e2845a3aca48c828490502eff84ea5f8f8cb43ecf17406&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=264818686&response-content-disposition=attachment%3B%20filename%3Dyolov5s.pt&response-content-type=application%2Foctet-stream [following]
--2021-11-12 00:03:03--  https://github-releases.githubusercontent.com/264818686/eab38592-7168-4731-bdff-ad5ede2002be?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20211112%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20211112T000303Z&X-Amz-Expires=300&X-Amz-Signature=372988e1039e0a4b44e2845a3aca48c828490502eff84ea5f8f8cb43ecf17406&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=264818686&response-content-disposition=attachment%3B%20filename%3Dyolov5s.pt&response-content-type=application%2Foctet-stream
Resolving github-releases.githubusercontent.com (github-releases.githubusercontent.com)... 185.199.108.154, 185.199.109.154, 185.199.111.154, ...
Connecting to github-releases.githubusercontent.com (github-releases.githubusercontent.com)|185.199.108.154|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 14698491 (14M) [application/octet-stream]
Saving to: ‘yolov5s.pt’

yolov5s.pt          100%[===================>]  14.02M  21.5MB/s    in 0.7s    

2021-11-12 00:03:04 (21.5 MB/s) - ‘yolov5s.pt’ saved [14698491/14698491]

Training the model
[ ]
!python train.py --img 416 --batch 8 --epoch 10 --data /content/drive/MyDrive/Identity_Recognition/face_mask_detection/dataset.yaml --weights /content/yolov5/yolov5s.pt --nosave --cache
Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
train: weights=/content/yolov5/yolov5s.pt, cfg=, data=/content/drive/MyDrive/Identity_Recognition/face_mask_detection/dataset.yaml, hyp=data/hyps/hyp.scratch.yaml, epochs=10, batch_size=8, imgsz=416, rect=False, resume=False, nosave=True, noval=False, noautoanchor=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, adam=False, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, patience=100, freeze=0, save_period=-1, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v6.0-84-gdef7a0f torch 1.10.0+cu111 CUDA:0 (Tesla K80, 11441MiB)

hyperparameters: lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Weights & Biases: run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=2

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     18879  models.yolo.Detect                      [2, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 270 layers, 7025023 parameters, 7025023 gradients, 15.9 GFLOPs

Transferred 343/349 items from /content/yolov5/yolov5s.pt
Scaled weight_decay = 0.0005
optimizer: SGD with parameter groups 57 weight, 60 weight (no decay), 60 bias
albumentations: version 1.0.3 required by YOLOv5, but version 0.1.12 is currently installed
train: Scanning '/content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/train/labels.cache' images and labels... 1161 found, 0 missing, 0 empty, 0 corrupted: 100% 1161/1161 [00:00<?, ?it/s]
train: Caching images (0.2GB ram):  54% 622/1161 [03:38<03:48,  2.36it/s]libpng warning: iCCP: Not recognizing known sRGB profile that has been edited
train: Caching images (0.4GB ram): 100% 1161/1161 [08:15<00:00,  2.34it/s]
val: Scanning '/content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/val/labels.cache' images and labels... 290 found, 0 missing, 0 empty, 0 corrupted: 100% 290/290 [00:00<?, ?it/s]
val: Caching images (0.1GB ram): 100% 290/290 [01:58<00:00,  2.45it/s]
Plotting labels... 

autoanchor: Analyzing anchors... anchors/target = 4.95, Best Possible Recall (BPR) = 0.9982
Image sizes 416 train, 416 val
Using 2 dataloader workers
Logging results to runs/train/exp
Starting training for 10 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       0/9    0.744G     0.109   0.03929   0.02507         6       416: 100% 146/146 [01:01<00:00,  2.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 19/19 [00:04<00:00,  4.27it/s]
                 all        290       1079     0.0783      0.187     0.0419    0.00755

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       1/9     1.13G   0.07695   0.03869   0.01933         7       416: 100% 146/146 [00:57<00:00,  2.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 19/19 [00:03<00:00,  5.08it/s]
                 all        290       1079      0.761      0.285      0.301     0.0934

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       2/9     1.13G    0.0668   0.03302   0.01776         1       416: 100% 146/146 [00:56<00:00,  2.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 19/19 [00:03<00:00,  5.28it/s]
                 all        290       1079      0.279      0.562       0.37      0.121

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       3/9     1.13G   0.06318   0.03089   0.01435         4       416: 100% 146/146 [00:56<00:00,  2.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 19/19 [00:03<00:00,  5.50it/s]
                 all        290       1079      0.515      0.644      0.582      0.233

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       4/9     1.13G   0.05934   0.02982  0.009973         9       416: 100% 146/146 [00:56<00:00,  2.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 19/19 [00:03<00:00,  5.66it/s]
                 all        290       1079      0.525      0.553      0.536      0.211

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       5/9     1.13G   0.05621   0.02875  0.007804         1       416: 100% 146/146 [00:56<00:00,  2.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 19/19 [00:03<00:00,  5.72it/s]
                 all        290       1079      0.548      0.579      0.575      0.236

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       6/9     1.13G   0.04975   0.02794  0.006794         7       416: 100% 146/146 [00:55<00:00,  2.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 19/19 [00:03<00:00,  5.48it/s]
                 all        290       1079      0.773      0.699      0.759      0.306

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       7/9     1.13G   0.04866   0.02601  0.006282         2       416: 100% 146/146 [00:55<00:00,  2.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 19/19 [00:03<00:00,  5.56it/s]
                 all        290       1079      0.809      0.728      0.773      0.354

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       8/9     1.13G   0.04004   0.02744  0.005477         4       416: 100% 146/146 [00:55<00:00,  2.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 19/19 [00:03<00:00,  5.58it/s]
                 all        290       1079       0.86      0.699      0.772       0.36

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       9/9     1.13G   0.03728   0.02591  0.005032         8       416: 100% 146/146 [00:56<00:00,  2.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 19/19 [00:03<00:00,  5.76it/s]
                 all        290       1079       0.85      0.775      0.819      0.404

10 epochs completed in 0.168 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 14.3MB
Optimizer stripped from runs/train/exp/weights/best.pt, 14.3MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model Summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 19/19 [00:05<00:00,  3.29it/s]
                 all        290       1079      0.851      0.775      0.819      0.404
                mask        290        822      0.895      0.881       0.91      0.495
              nomask        290        257      0.806      0.669      0.729      0.314
Results saved to runs/train/exp
[ ]
# Start tensorboard
# Launch after you have started the training 
# Logs save in the folder "runs"
%load_ext tensorboard
%tensorboard --logdir runs

Inference on new images
[ ]
## inference or detection on new images
!python detect.py --source /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test --weights /content/yolov5/runs/train/exp/weights/last.pt --img 416 --save-txt --save-conf
detect: weights=['/content/yolov5/runs/train/exp/weights/last.pt'], source=/content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test, imgsz=[416, 416], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=True, save_conf=True, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
YOLOv5 🚀 v6.0-84-gdef7a0f torch 1.10.0+cu111 CUDA:0 (Tesla K80, 11441MiB)

Fusing layers... 
Model Summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
image 1/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/173.jpg: 416x416 1 nomask, Done. (0.028s)
image 2/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/27.jpg: 416x416 1 nomask, Done. (0.029s)
image 3/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/crowd_mask181.jpg: 256x416 12 masks, Done. (0.025s)
image 4/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/crowd_mask23.jpg: 224x416 1 mask, Done. (0.027s)
image 5/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/crowd_mask38.jpg: 288x416 1 mask, Done. (0.027s)
image 6/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/crowd_mask62.jpg: 320x416 7 masks, 4 nomasks, Done. (0.028s)
image 7/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/crowd_mask65.jpg: 256x416 10 masks, 2 nomasks, Done. (0.026s)
image 8/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/image_117.jpg: 288x416 1 mask, Done. (0.027s)
image 9/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/image_5.jpg: 320x416 1 mask, Done. (0.027s)
image 10/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/image_503.jpg: 288x416 1 mask, Done. (0.026s)
image 11/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/image_577.jpeg: 416x288 5 masks, Done. (0.025s)
image 12/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/image_602.jpg: 416x288 1 nomask, Done. (0.023s)
image 13/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/image_605.jpg: 416x288 1 mask, Done. (0.023s)
image 14/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/image_609.jpg: 416x288 1 nomask, Done. (0.023s)
image 15/15 /content/drive/MyDrive/Identity_Recognition/face_mask_detection/face_data/test/new_116.jpg: 416x416 1 nomask, Done. (0.030s)
Speed: 0.5ms pre-process, 26.4ms inference, 1.5ms NMS per image at shape (1, 3, 416, 416)
Results saved to runs/detect/exp
15 labels saved to runs/detect/exp/labels
Display Result Images
[ ]
#display result images 

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assume JPG file
    display(Image(filename=imageName))
    print("\n")

Download weights for future use
[ ]
#export the model's weights for future use 
from google.colab import files
files.download('./runs/train/exp/weights/last.pt')
[ ]
import io
import base64
from IPython.display import HTML

def playvideo(filename):
    video = io.open(filename, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
                 </video>'''.format(encoded.decode('ascii')))
OPTIONAL!!! Retrain the model from the last saved weight for better results
[ ]
train.py --img 416 --batch 8 --epochs 150 --data /content/drive/MyDrive/Identity_Recognition/face_mask_detection/dataset.yaml --weights /content/yolov5/runs/train/exp/weights/last.pt
Finish
