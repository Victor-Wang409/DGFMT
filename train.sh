python baseline.py \
    --audio_path /home/wangchenhao/Github/MMER/data/MSP_Podcast/Audios \
    --csv_path ./MSP_Podcast.csv \
    --model_path /home/wangchenhao/Github/emotion2vec/upstream \
    --checkpoint_path /home/wangchenhao/Github/emotion2vec/iic/emotion2vec_base/emotion2vec_base.pt \
    --batch_size 16 \
    --epochs 20 \
    --lr 5e-5 \
    --save_dir /home/wangchenhao/Github/baseline/models