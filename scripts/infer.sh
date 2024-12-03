# 评估所有fold的模型
python inference.py \
    --feature_dir /home/wangchenhao/Github/baseline/emotion2vec_features \
    --model_path /home/wangchenhao/Github/baseline/models \
    --csv_path ./MSP_Podcast.csv \
    --output_dir ./results \
    --fold 1