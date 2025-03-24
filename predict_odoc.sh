python predict.py --model unet --task odoc --device cuda:0 --model_path weights/unet_refuge,fundoct,drishti,g1020,origa_latest/best_unet_model.pth --datasets papila
python predict.py --model unet --task odoc --device cuda:0 --model_path weights/unet_refuge,papila,drishti,g1020,origa_latest/best_unet_model.pth --datasets fundoct
python predict.py --model unet --task odoc --device cuda:0 --model_path weights/unet_refuge,papila,fundoct,g1020,origa_latest/best_unet_model.pth --datasets drishti
python predict.py --model unet --task odoc --device cuda:0 --model_path weights/unet_refuge,papila,fundoct,drishti,origa_latest/best_unet_model.pth --datasets g1020
python predict.py --model unet --task odoc --device cuda:0 --model_path weights/unet_refuge,papila,fundoct,drishti,g1020_latest/best_unet_model.pth --datasets origa

python predict.py --model swinunetr --task odoc --device cuda:0 --model_path weights/swinunetr_refuge,fundoct,drishti,g1020,origa_latest/best_swinunetr_model.pth --datasets papila
python predict.py --model swinunetr --task odoc --device cuda:0 --model_path weights/swinunetr_refuge,papila,drishti,g1020,origa_latest/best_swinunetr_model.pth --datasets fundoct
python predict.py --model swinunetr --task odoc --device cuda:0 --model_path weights/swinunetr_refuge,papila,fundoct,g1020,origa_latest/best_swinunetr_model.pth --datasets drishti
python predict.py --model swinunetr --task odoc --device cuda:0 --model_path weights/swinunetr_refuge,papila,fundoct,drishti,origa_latest/best_swinunetr_model.pth --datasets g1020
python predict.py --model swinunetr --task odoc --device cuda:0 --model_path weights/swinunetr_refuge,papila,fundoct,drishti,g1020_latest/best_swinunetr_model.pth --datasets origa

python predict.py --model unetr --task odoc --device cuda:0 --model_path weights/unetr_refuge,fundoct,drishti,g1020,origa_latest/best_unetr_model.pth --datasets papila
python predict.py --model unetr --task odoc --device cuda:0 --model_path weights/unetr_refuge,papila,drishti,g1020,origa_latest/best_unetr_model.pth --datasets fundoct
python predict.py --model unetr --task odoc --device cuda:0 --model_path weights/unetr_refuge,papila,fundoct,g1020,origa_latest/best_unetr_model.pth --datasets drishti
python predict.py --model unetr --task odoc --device cuda:0 --model_path weights/unetr_refuge,papila,fundoct,drishti,origa_latest/best_unetr_model.pth --datasets g1020
python predict.py --model unetr --task odoc --device cuda:0 --model_path weights/unetr_refuge,papila,fundoct,drishti,g1020_latest/best_unetr_model.pth --datasets origa

python predict.py --model posal --task odoc --device cuda:0 --model_path weights/posal_refuge,fundoct,drishti,g1020,origa_latest/best_posal_model.pth --datasets papila
python predict.py --model posal --task odoc --device cuda:0 --model_path weights/posal_refuge,papila,drishti,g1020,origa_latest/best_posal_model.pth --datasets fundoct
python predict.py --model posal --task odoc --device cuda:0 --model_path weights/posal_refuge,papila,fundoct,g1020,origa_latest/best_posal_model.pth --datasets drishti
python predict.py --model posal --task odoc --device cuda:0 --model_path weights/posal_refuge,papila,fundoct,drishti,origa_latest/best_posal_model.pth --datasets g1020
python predict.py --model posal --task odoc --device cuda:0 --model_path weights/posal_refuge,papila,fundoct,drishti,g1020_latest/best_posal_model.pth --datasets origa