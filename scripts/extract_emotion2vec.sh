python extract_features_emo2vec.py \
    --audio_dir /home/wangchenhao/Github/Dataset/MSP_Podcast/Audios \
    --csv_path ./Dataset.csv\
    --model_path /home/wangchenhao/Github/emotion2vec/upstream \
    --checkpoint_path /home/wangchenhao/Github/emotion2vec/iic/emotion2vec_base/emotion2vec_base.pt \
    --output_dir ./emotion2vec_features_fine_tune