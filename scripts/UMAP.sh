python UMAP.py \
    --emotion2vec_dir ./Features/emo2vec_large_features \
    --hubert_dir ./Features/hubert_large_features \
    --wav2vec_dir ./Features/wav2vec_large_features \
    --data2vec_dir ./Features/data2vec_large_features \
    --model_path /home/wangchenhao/Github/DGFMT/models/emo+hubert+wav+data/epoch1/pytorch_model.bin \
    --csv_path ./csv_files/IEMOCAP.csv \
    --output_dir ./results \
    --n_neighbors 10 \
    --min_dist 0.3