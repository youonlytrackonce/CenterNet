cd src
# train
python main.py ctdet --exp_id ctdet_ch_ft_trackeveryseason_dla_34 --batch_size 30  --lr 5e-3 --gpus 0 --num_workers 12 --num_epochs 50 --lr_step 40 --dataset trackeveryseason --input_w 640 --input_h 640 --load_model /home/fatih/phd/experiments/training/detector/centernet/centernet_dla34_640x384_crowdhuman/model_last.pth --val_intervals 100
