python UMAP.py \
    --emotion2vec_dir ./emo2vec_large_features \
    --hubert_dir ./hubert_large_features \
    --model_path ./models/fold1/epoch1/pytorch_model.bin\
    --csv_path ./csv_files/EMOVO.csv \
    --output_dir ./results \
    --n_neighbors 15 \
    --min_dist 0.3