python baseline.py \
    --feature_dir /home/wangchenhao/Github/baseline/emotion2vec_features \
    --csv_path ./Dataset.csv \
    --batch_size 128 \
    --lr 1e-4 \
    --save_dir ./models \
    --patience 10 \
    --seed 20 \
    --min_delta 0.01