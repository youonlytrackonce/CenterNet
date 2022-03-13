cd src
# train
python main.py ctdet --exp_id trackeveryseason_dla_2x --batch_size 16 --lr 5e-4 --gpus 0 --num_workers 16 --num_epochs 20 --dataset trackeveryseason --load_model /home/fatih/phd/experiments/training/detector/centernet/centernet_dla34_640x384_crowdhuman/model_best.pth --input_h 640 --input_w 640 --arch dla_34
# or use the following command if your have coco_s2_dla_1x trained
# python main.py ctdet --exp_id coco_dla_2x --batch_size 128 --master_batch 9 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --load_model ../exp/ctdet/coco_dla_1x/model_90.pth --resume
# test
#python test.py ctdet --exp_id coco_dla_2x --keep_res --resume
# flip test
#python test.py ctdet --exp_id coco_dla_2x --keep_res --resume --flip_test
# multi scale test
#python test.py ctdet --exp_id coco_dla_2x --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
