python inference.py \
    --audio_dir /home/wangchenhao/Github/MMER/data/iemocap \
    --model_path /home/wangchenhao/Github/emotion2vec/upstream \
    --checkpoint_path /home/wangchenhao/Github/emotion2vec/iic/emotion2vec_base/emotion2vec_base.pt \
    --vad_model_path /home/wangchenhao/Github/baseline/models/best_model_fold2 \
    --csv_path /home/wangchenhao/Github/MMER/data/iemocap.csv \
    --output_dir ./results