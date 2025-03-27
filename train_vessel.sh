#!/bin/bash

python train.py --datasets fives_train,chasedb1,stare,hrf,hei-med,fives_test --model posal --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,stare,hrf,hei-med,fives_test --model posal --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,hrf,hei-med,fives_test --model posal --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hei-med,fives_test --model posal --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,fives_test --model posal --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,hei-med --model posal --task vessel --device cuda:1 --batch_size 16

python train.py --datasets fives_train,chasedb1,stare,hrf,hei-med,fives_test --model denet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,stare,hrf,hei-med,fives_test --model denet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,hrf,hei-med,fives_test --model denet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hei-med,fives_test --model denet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,fives_test --model denet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,hei-med --model denet --task vessel --device cuda:1 --batch_size 16

python train.py --datasets fives_train,chasedb1,stare,hrf,hei-med,fives_test --model unet --task vessel --device cuda:1 --batch_size 32
python train.py --datasets fives_train,drive,stare,hrf,hei-med,fives_test --model unet --task vessel --device cuda:1 --batch_size 32
python train.py --datasets fives_train,drive,chasedb1,hrf,hei-med,fives_test --model unet --task vessel --device cuda:1 --batch_size 32
python train.py --datasets fives_train,drive,chasedb1,stare,hei-med,fives_test --model unet --task vessel --device cuda:1 --batch_size 32
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,fives_test --model unet --task vessel --device cuda:1 --batch_size 32
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,hei-med --model unet --task vessel --device cuda:1 --batch_size 32

python train.py --datasets fives_train,chasedb1,stare,hrf,hei-med,fives_test --model swinunetr --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,stare,hrf,hei-med,fives_test --model swinunetr --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,hrf,hei-med,fives_test --model swinunetr --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hei-med,fives_test --model swinunetr --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,fives_test --model swinunetr --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,hei-med --model swinunetr --task vessel --device cuda:1 --batch_size 16

python train.py --datasets fives_train,chasedb1,stare,hrf,hei-med,fives_test --model unetr --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,stare,hrf,hei-med,fives_test --model unetr --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,hrf,hei-med,fives_test --model unetr --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hei-med,fives_test --model unetr --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,fives_test --model unetr --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,hei-med --model unetr --task vessel --device cuda:1 --batch_size 16

python train.py --datasets fives_train,chasedb1,stare,hrf,hei-med,fives_test --model csunet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,stare,hrf,hei-med,fives_test --model csunet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,hrf,hei-med,fives_test --model csunet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hei-med,fives_test --model csunet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,fives_test --model csunet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,hei-med --model csunet --task vessel --device cuda:1 --batch_size 16

python train.py --datasets fives_train,chasedb1,stare,hrf,hei-med,fives_test --model saunet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,stare,hrf,hei-med,fives_test --model saunet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,hrf,hei-med,fives_test --model saunet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hei-med,fives_test --model saunet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,fives_test --model saunet --task vessel --device cuda:1 --batch_size 16
python train.py --datasets fives_train,drive,chasedb1,stare,hrf,hei-med --model saunet --task vessel --device cuda:1 --batch_size 16