python predict.py --model unet --task vessel --device cuda:1 --model_path weights/unet_fives_train,chasedb1,stare,hrf,hei-med,fives_test_latest/best_unet_model.pth --datasets drive
python predict.py --model unet --task vessel --device cuda:1 --model_path weights/unet_fives_train,drive,stare,hrf,hei-med,fives_test_latest/best_unet_model.pth --datasets chasedb1
python predict.py --model unet --task vessel --device cuda:1 --model_path weights/unet_fives_train,drive,chasedb1,hrf,hei-med,fives_test_latest/best_unet_model.pth --datasets stare
python predict.py --model unet --task vessel --device cuda:1 --model_path weights/unet_fives_train,drive,chasedb1,stare,hei-med,fives_test_latest/best_unet_model.pth --datasets hrf
python predict.py --model unet --task vessel --device cuda:1 --model_path weights/unet_fives_train,drive,chasedb1,stare,hrf,fives_test_latest/best_unet_model.pth --datasets hei-med
python predict.py --model unet --task vessel --device cuda:1 --model_path weights/unet_fives_train,drive,chasedb1,stare,hrf,hei-med_latest/best_unet_model.pth --datasets fives_test

python predict.py --model swinunetr --task vessel --device cuda:1 --model_path weights/swinunetr_fives_train,chasedb1,stare,hrf,hei-med,fives_test_latest/best_swinunetr_model.pth --datasets drive
python predict.py --model swinunetr --task vessel --device cuda:1 --model_path weights/swinunetr_fives_train,drive,stare,hrf,hei-med,fives_test_latest/best_swinunetr_model.pth --datasets chasedb1
python predict.py --model swinunetr --task vessel --device cuda:1 --model_path weights/swinunetr_fives_train,drive,chasedb1,hrf,hei-med,fives_test_latest/best_swinunetr_model.pth --datasets stare
python predict.py --model swinunetr --task vessel --device cuda:1 --model_path weights/swinunetr_fives_train,drive,chasedb1,stare,hei-med,fives_test_latest/best_swinunetr_model.pth --datasets hrf
python predict.py --model swinunetr --task vessel --device cuda:1 --model_path weights/swinunetr_fives_train,drive,chasedb1,stare,hrf,fives_test_latest/best_swinunetr_model.pth --datasets hei-med
python predict.py --model swinunetr --task vessel --device cuda:1 --model_path weights/swinunetr_fives_train,drive,chasedb1,stare,hrf,hei-med_latest/best_swinunetr_model.pth --datasets fives_test

# python predict.py --model unetr --task vessel --device cuda:1 --model_path weights/unetr_fives_train,chasedb1,stare,hrf,hei-med,fives_test_latest/best_unetr_model.pth --datasets drive
# python predict.py --model unetr --task vessel --device cuda:1 --model_path weights/unetr_fives_train,drive,stare,hrf,hei-med,fives_test_latest/best_unetr_model.pth --datasets chasedb1
# python predict.py --model unetr --task vessel --device cuda:1 --model_path weights/unetr_fives_train,drive,chasedb1,hrf,hei-med,fives_test_latest/best_unetr_model.pth --datasets stare
# python predict.py --model unetr --task vessel --device cuda:1 --model_path weights/unetr_fives_train,drive,chasedb1,stare,hei-med,fives_test_latest/best_unetr_model.pth --datasets hrf
# python predict.py --model unetr --task vessel --device cuda:1 --model_path weights/unetr_fives_train,drive,chasedb1,stare,hrf,fives_test_latest/best_unetr_model.pth --datasets hei-med
# python predict.py --model unetr --task vessel --device cuda:1 --model_path weights/unetr_fives_train,drive,chasedb1,stare,hrf,hei-med_latest/best_unetr_model.pth --datasets fives_test
