cd src
# eval
python test.py ctdet --exp_id eval_ctdet_ch_ft_trackeveryseason_dla_34 --load_model /home/fatih/phd/CenterNet/exp/ctdet/ctdet_ch_ft_trackeveryseason_dla_34/logs_2022-02-23-13-51/model_best.pth --gpus 0 --dataset trackeveryseason --input_w 640 --input_h 640 --batch_size 64
