import os
from funasr import AutoModel
from pathlib import Path
import time

# ================= 配置区域 =================
directory = Path("/home/skywingsir/DataSet/Audios")
output_dir = "/home/skywingsir/Github/DGFMT/Features/emo2vec_large_features"
progress_log_file = "processed_files.txt"

# 关键设置：
# 为了能实时看到打印的文件名，我们需要手动把任务切分成小块。
# 这里的 32 既是每次打印的文件数量，也是送给显卡处理的批次大小。
BATCH_PROCESS_SIZE = 1 
# ===========================================

# 1. 扫描文件
print(f"正在扫描音频文件: {directory} ...")
all_wav_files = [str(f.absolute()) for f in directory.rglob("*.wav")]
print(f"找到总文件数: {len(all_wav_files)}")

# 2. 读取断点记录 (跳过已处理的)
processed_files = set()
if os.path.exists(progress_log_file):
    with open(progress_log_file, 'r', encoding='utf-8') as f:
        for line in f:
            processed_files.add(line.strip())
    print(f"已跳过 {len(processed_files)} 个历史文件")

# 3. 过滤出待处理文件
files_to_process = [f for f in all_wav_files if f not in processed_files]
if not files_to_process:
    print("所有文件已处理完毕！")
    exit()

print(f"剩余 {len(files_to_process)} 个文件待处理...")

# 4. 加载模型 (建议指定 GPU)
model = AutoModel(model="iic/emotion2vec_plus_large", device="cuda")

# 5. 循环处理并打印文件名
# range的步长设为 BATCH_PROCESS_SIZE
for i in range(0, len(files_to_process), BATCH_PROCESS_SIZE):
    # 取出当前这一批 (例如 32 个)
    batch_files = files_to_process[i : i + BATCH_PROCESS_SIZE]
    
    # --- 打印当前正在处理的文件名 ---
    print(f"\n[Batch {i//BATCH_PROCESS_SIZE + 1}] 正在处理:")
    for file_path in batch_files:
        # 只打印文件名(不含长路径)，让日志更清晰；如果需要全路径去掉 .name 即可
        print(f" -> {Path(file_path).name}") 
    # -----------------------------

    try:
        # 调用模型处理这一小批
        model.generate(
            batch_files, 
            output_dir=output_dir, 
            granularity="frame", 
            extract_embedding=True, 
            disable_pbar=True  # 关闭内部进度条，防止刷屏太乱
        )
        
        # 处理完立即记录到 txt，防止白跑
        with open(progress_log_file, 'a', encoding='utf-8') as f:
            for file_path in batch_files:
                f.write(file_path + '\n')
                
    except Exception as e:
        print(f"\n[Error] 处理出错: {e}")
        print(f"出错文件在: {batch_files[0]} 到 {batch_files[-1]} 之间")
        break

print("\n处理完成！")