import re
import os
from datetime import datetime

def merge_logs_to_ai_report(rank0_path, tee_path, output_path, sample_rate=20):
    """
    将训练日志和评估日志合并。
    优化点：确保一定会提取到最后一个 batch 的数据。
    """
    if not os.path.exists(rank0_path):
        print(f"警告：找不到训练日志 {rank0_path}")
    if not os.path.exists(tee_path):
        print(f"警告：找不到评估日志 {tee_path}")

    configs = []
    lora_config = []
    training_progress = []
    system_msgs = []
    eval_scores = []
    
    seen_score_lines = set()
    is_lora_block = False
    
    # 记录最后一条 batch 信息的变量
    last_batch_info = None 
    last_added_batch_num = -1

    batch_pattern = re.compile(r"batch: (\d+),.*total: ([\d.]+)")
    detailed_header = "[Evaluation Metrics] ['SS Gender', 'SS Profession', 'SS Race', 'SS Religion', 'SS Score', 'LM Score', 'ICAT Score']"

    # --- 1. 处理 rank0.log ---
    if os.path.exists(rank0_path):
        with open(rank0_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取配置 (同前)
                if "root INFO" in line and "batch:" not in line:
                    content = line.split("root INFO ")[-1].strip()
                    if ":" in content and "batch" not in content.lower() and "LoraConfig" not in content:
                        configs.append(f"[Config] {content}")

                # 提取 LoRA 配置块 (同前)
                if "LoraConfig(" in line:
                    is_lora_block = True
                if is_lora_block:
                    lora_config.append(line.strip())
                    if ")" in line and "LoraConfig" not in line:
                        is_lora_block = False

                # 提取训练进度 (核心修改点)
                if "batch:" in line:
                    match = batch_pattern.search(line)
                    if match:
                        batch_num = int(match.group(1))
                        content = line.split("root INFO ")[-1].strip()
                        
                        # 更新“最后一条”记录
                        last_batch_info = content 
                        
                        # 按频率采样
                        if batch_num % sample_rate == 0:
                            training_progress.append(content)
                            last_added_batch_num = batch_num

        # --- 强补最后一行 ---
        # 如果最后一条记录没有因为整除被加入，手动补上，标记为 [Final]
        if last_batch_info and last_added_batch_num != -1:
            # 检查最后一条是否已经包含在列表里了
            if not training_progress or last_batch_info != training_progress[-1]:
                training_progress.append(f"[Final Batch] {last_batch_info}")

    # --- 2. 处理 tee.log (同前) ---
    if os.path.exists(tee_path):
        with open(tee_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if "Warning:" in line or "UserWarning:" in line:
                    clean_msg = re.sub(r'^\[rank\d+\]:', '', line).strip()
                    if clean_msg not in system_msgs:
                        system_msgs.append(f"[System] {clean_msg}")

                if i > 0 and "SS Score" in lines[i-1]:
                    score_values = line.replace('\t', ' ').strip()
                    if score_values and score_values not in seen_score_lines:
                        if re.match(r'^[\d.\s]+$', score_values):
                            eval_scores.append(f"[Scores] {line}")
                            seen_score_lines.add(score_values)

    # --- 3. 写入最终报告 ---
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"================================================\n")
        f.write(f"AI 分析报告: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"================================================\n\n")

        f.write("### 1. 配置与参数\n")
        f.write("\n".join(configs) + "\n\n")
        if lora_config:
            f.write("### 2. LoRA 配置\n")
            f.write("\n".join(lora_config) + "\n\n")

        if system_msgs:
            f.write("### 3. 系统日志/警告\n")
            f.write("\n".join(system_msgs) + "\n\n")

        f.write(f"### 4. 训练进度 (每 {sample_rate} 个 batch 抽样一次 + 强制保留末尾)\n")
        f.write("\n".join(training_progress) + "\n\n")

        f.write("### 5. 评估分数趋势\n")
        f.write(detailed_header + "\n")
        f.write("\n".join(eval_scores) if eval_scores else "(暂无评估数据)")

    print(f"处理完成！已确保包含最后一条 batch 记录。保存至: {output_path}")

if __name__ == "__main__":
    merge_logs_to_ai_report(
        rank0_path="../logs/gpt-m-new.rank0.log", 
        tee_path="../logs/gpt-m-new.tee.log", 
        output_path="merged_report_for_ai.txt", 
        sample_rate=20
    )