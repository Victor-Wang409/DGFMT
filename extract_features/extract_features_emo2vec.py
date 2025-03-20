'''
Using the emotion representation model
rec_result only contains {'feats'}
	granularity="utterance": {'feats': [*1024]}
	granularity="frame": {feats: [T*1024]}
'''

from funasr import AutoModel
from pathlib import Path

directory = Path("/home/wangchenhao/Github/Dataset/EMOVO")  # 替换成你的目录路径
wav_files = [str(f.absolute()) for f in directory.rglob("*.wav")]

# print(wav_files)

model_id = "iic/emotion2vec_plus_large"
model = AutoModel(
    model=model_id,
    hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
    disable_update=True
)


rec_result = model.generate(wav_files, output_dir="/home/wangchenhao/Github/Dynamic/emo2vec_large_features", granularity="frame")

# model = AutoModel(model="iic/emotion2vec_base")

# res = model.generate(wav_files, output_dir="./emotion2vec_features", granularity="frame", extract_embedding=True)