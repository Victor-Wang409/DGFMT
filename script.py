import csv
import os

def filter_csv_by_whitelist(source_csv, whitelist_csv, output_csv):
    """
    source_csv: 待处理的 MSP_Podcast.csv
    whitelist_csv: 包含允许文件名的 filenames.csv
    output_csv: 处理后保存的新文件名
    """
    
    # 1. 读取 filenames.csv，获取白名单文件名（不含后缀）
    allowed_names = set()
    try:
        with open(whitelist_csv, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            # 如果你的 filenames.csv 有表头，请取消下面这一行的注释
            # next(reader) 
            for row in reader:
                if row:
                    # 获取文件名并去除可能的后缀及空格
                    name = os.path.splitext(row[0].strip())[0]
                    allowed_names.add(name)
    except FileNotFoundError:
        print(f"错误：找不到白名单文件 {whitelist_csv}")
        return

    print(f"成功加载白名单，共有 {len(allowed_names)} 个基础文件名。")

    # 2. 读取源 CSV 并过滤
    filtered_rows = []
    header = None
    
    try:
        with open(source_csv, mode='r', encoding='utf-8-sig') as f:
            # 使用 DictReader 可以根据属性名 'FileName' 准确定位
            reader = csv.DictReader(f)
            header = reader.fieldnames
            
            for row in reader:
                # 获取 FileName 属性的值
                original_filename = row.get('FileName', '')
                # 去除后缀进行匹配
                pure_name = os.path.splitext(original_filename.strip())[0]
                
                if pure_name in allowed_names:
                    filtered_rows.append(row)
    except FileNotFoundError:
        print(f"错误：找不到源文件 {source_csv}")
        return
    except KeyError:
        print("错误：在 MSP_Podcast.csv 中没找到名为 'FileName' 的列，请检查列名大小写。")
        return

    # 3. 将结果写入新文件
    with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"处理完成！")
    print(f"原始行数: {reader.line_num - 1}") # 减去表头
    print(f"保留行数: {len(filtered_rows)}")
    print(f"结果已保存至: {output_csv}")

# --- 配置参数 ---
source_file = '/home/wangchenhao/Github/DGFMT/csv_files/MSP_Podcast.csv'   # 原始 CSV
whitelist = 'filenames.csv'      # 你的白名单
output = 'MSP_Podcast_Filtered.csv' # 输出文件

if __name__ == "__main__":
    filter_csv_by_whitelist(source_file, whitelist, output)