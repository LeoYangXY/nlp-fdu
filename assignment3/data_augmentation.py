import random
import os
from pathlib import Path
from tqdm import tqdm

def apply_augmentation(line, strategy):
    """应用单条数据的增强策略"""
    if len(line) != 20:
        return line
    
    chars = list(line)
    vocab = [chr(ord('a') + i) for i in range(26)]  # 仅字母（不包含空格）
    
    if strategy == "mask":
        # 挖空策略：随机挖空1-3个字母
        indices = [i for i, c in enumerate(chars) if c != ' ']
        if indices:
            for i in random.sample(indices, k=min(3, len(indices))):
                chars[i] = '_'
    
    elif strategy == "reverse_words":
        # 单词倒序（保留空格位置）
        words = line.split(' ')
        reversed_words = [w[::-1] for w in words]
        return ' '.join(reversed_words)
    
    elif strategy == "replace":
        # 随机替换字母（保留空格）
        for i in range(len(chars)):
            if chars[i] != ' ' and random.random() < 0.3:
                chars[i] = random.choice(vocab)
    
    elif strategy == "swap_adjacent":
        # 相邻字母交换（不交换空格）
        indices = [i for i in range(19) if chars[i] != ' ' and chars[i+1] != ' ']
        if indices:
            i = random.choice(indices)
            chars[i], chars[i+1] = chars[i+1], chars[i]
    
    return ''.join(chars)

def generate_mixed_augmentation(input_path, output_path, target_size=1000000):
    """
    生成混合增强数据
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（如 'data/lettercounting-train-mix-augmented.txt'）
        target_size: 目标总行数（包括原始数据）
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 读取原始数据
    with open(input_path, 'r') as f:
        original_lines = [line.strip() for line in f if len(line.strip()) == 20]
    
    if not original_lines:
        print("错误：未读取到有效数据（需确保每行都是20字符）")
        return
    
    # 增强策略配置
    strategies = [
        ("mask", 0.3),          # 30%挖空增强
        ("reverse_words", 0.2), # 20%单词倒序
        ("replace", 0.4),      # 40%字符替换
        ("swap_adjacent", 0.1)  # 10%相邻交换
    ]
    
    # 计算需要生成的行数（保留原始数据）
    augmented_lines_needed = target_size - len(original_lines)
    lines_per_strategy = {
        s[0]: int(augmented_lines_needed * s[1]) 
        for s in strategies
    }
    
    # 生成增强数据
    with open(output_path, 'w') as f:
        # 1. 先写入原始数据
        for line in original_lines:
            f.write(line + '\n')
        
        # 2. 写入增强数据（带进度条）
        pbar = tqdm(total=augmented_lines_needed, desc="生成增强数据")
        
        for strategy, count in lines_per_strategy.items():
            for _ in range(count):
                src_line = random.choice(original_lines)
                augmented_line = apply_augmentation(src_line, strategy)
                f.write(augmented_line + '\n')
                pbar.update(1)
        
        pbar.close()
    
    # 验证文件行数
    with open(output_path, 'r') as f:
        line_count = sum(1 for _ in f)
    
    print(f"\n增强完成！总数据量: {line_count}行")
    print(f"增强策略分布: {dict(strategies)}")
    print(f"结果保存到: {output_path}")

if __name__ == "__main__":
    # 配置路径
    input_file = Path(__file__).parent / 'data' / 'lettercounting-train.txt'
    output_file = Path(__file__).parent / 'data' / 'lettercounting-train-mix-augmented.txt'
    
    # 生成百万级混合增强数据
    generate_mixed_augmentation(
        input_file, 
        output_file,
        target_size=50000  # 目标总行数（包括原始数据）
    )