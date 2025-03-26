#!/bin/bash

python train.py --datasets refuge,fundoct,drishti,g1020,origa --model unet --task odoc --device cuda:0 --batch_size 32
python train.py --datasets refuge,papila,drishti,g1020,origa --model unet --task odoc --device cuda:0 --batch_size 32
python train.py --datasets refuge,papila,fundoct,g1020,origa --model unet --task odoc --device cuda:0 --batch_size 32
python train.py --datasets refuge,papila,fundoct,drishti,origa --model unet --task odoc --device cuda:0 --batch_size 32
python train.py --datasets refuge,papila,fundoct,drishti,g1020 --model unet --task odoc --device cuda:0 --batch_size 32

python train.py --datasets refuge,fundoct,drishti,g1020,origa --model swinunetr --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,drishti,g1020,origa --model swinunetr --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,g1020,origa --model swinunetr --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,drishti,origa --model swinunetr --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,drishti,g1020 --model swinunetr --task odoc --device cuda:0 --batch_size 16

python train.py --datasets refuge,fundoct,drishti,g1020,origa --model unetr --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,drishti,g1020,origa --model unetr --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,g1020,origa --model unetr --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,drishti,origa --model unetr --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,drishti,g1020 --model unetr --task odoc --device cuda:0 --batch_size 16

python train.py --datasets refuge,fundoct,drishti,g1020,origa --model posal --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,drishti,g1020,origa --model posal --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,g1020,origa --model posal --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,drishti,origa --model posal --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,drishti,g1020 --model posal --task odoc --device cuda:0 --batch_size 16

python train.py --datasets refuge,fundoct,drishti,g1020,origa --model denet --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,drishti,g1020,origa --model denet --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,g1020,origa --model denet --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,drishti,origa --model denet --task odoc --device cuda:0 --batch_size 16
python train.py --datasets refuge,papila,fundoct,drishti,g1020 --model denet --task odoc --device cuda:0 --batch_size 16