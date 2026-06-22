python run_infer.py \
--gpu='1' \
--nr_types=5 \
--type_info_path=type_info.json \
--batch_size=16 \
--model_mode=MyNet \
--model_path=/home/ZXIAO/MyNet/logs/Neure_7.28/net_epoch=50.tar \
--nr_inference_workers=16 \
--nr_post_proc_workers=0 \
tile \
--input_dir=/home/ZXIAO/datasets/Neure_512/test4/images \
--output_dir=/home/ZXIAO/datasets/Neure_512/test4/output2 \
--mem_usage=0.2 \
--draw_dot \
--save_qupath
