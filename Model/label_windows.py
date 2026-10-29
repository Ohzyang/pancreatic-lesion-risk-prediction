import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
CONFIG = {
    'input_path': r"D:\wzy\zyc\point_name_data\final_data\base_indicator_loose_complete_nocut_20_final_len4_nobroadcast.csv", 
    'output_path': r"D:\wzy\zyc\point_name_data\final_data\base_indicator_loose_complete_nocut_20_final_len4_nobroadcast_121824.csv",
    'id_col': 'patient_no',
    'time_col': 'ExamD',
    'target_col': '腹部彩超结果',
    'windows': [12, 18, 24],
    
    'min_input_seq_len': 3,        # 正样本的最小输入长度 (即最少需要 2 次输入 + 1 次结局 = 总长3)
    'min_neg_input_seq_len': 3     # 【新增】负样本的最小输入长度 (即最少需要 5 次输入 + 1 次结局 = 总长6)
}

def generate_labels():
    print(f"Loading data from {CONFIG['input_path']}...")
    try:
        df = pd.read_csv(CONFIG['input_path'], encoding='gb18030')
    except UnicodeDecodeError:
        df = pd.read_csv(CONFIG['input_path'], encoding='utf-8')
    
    # 1. 基础清洗与排序 (至关重要)
    df[CONFIG['time_col']] = pd.to_datetime(df[CONFIG['time_col']])
    df = df.sort_values([CONFIG['id_col'], CONFIG['time_col']])
    
    # 2. 筛选病人 (只保留最后结局为 0 或 2 的人)
    last_visit = df.groupby(CONFIG['id_col']).last().reset_index()
    valid_patients = last_visit[last_visit[CONFIG['target_col']].isin([0, 0.0, 2, 2.0])][CONFIG['id_col']].unique()
    df_valid = df[df[CONFIG['id_col']].isin(valid_patients)].copy()
    
    processed_rows = []
    
    # === 新增统计计数器 ===
    drop_due_to_pos_len = 0   # 因为正样本长度不足丢掉的人数
    drop_due_to_neg_len = 0   # 因为负样本长度不足丢掉的人数
    kept_patients = 0         # 成功保留的人数
    
    # 3. 核心处理循环
    for pid, group in tqdm(df_valid.groupby(CONFIG['id_col']), desc="Generating Labels"):
        events = group.to_dict('records')
        total_len = len(events)
        
        # === 提前提取最终结局，判断当前病人是正样本还是负样本 ===
        outcome_event = events[-1]
        outcome_label = outcome_event[CONFIG['target_col']]
        is_positive = (outcome_label in [2, 2.0])
        
        # === 分通道进行长度初筛 ===
        if is_positive:
            # 正样本：使用原来的宽松门槛
            if total_len < (CONFIG['min_input_seq_len'] + 1):
                drop_due_to_pos_len += 1
                continue
        else:
            # 负样本：使用更严格的长度门槛
            if total_len < (CONFIG['min_neg_input_seq_len'] + 1):
                drop_due_to_neg_len += 1
                continue
                
        kept_patients += 1
                
        outcome_date = outcome_event[CONFIG['time_col']]
        input_events = events[:-1]
        
        patient_rows = []
        has_valid_anchor = False # 记录这个病人是否至少产生了一个有效样本
        
        # 遍历该病人的每一次历史检查（样本扩充核心逻辑）
        for i, event in enumerate(input_events):
            current_seq_len = i + 1
            current_date = event[CONFIG['time_col']]
            
            row_data = event.copy()
            row_data['is_outcome_visit'] = 0
            
            # 计算当前截断点距离最终结局的时间
            days_to_outcome = (outcome_date - current_date).days
            if days_to_outcome < 0: days_to_outcome = 0
            row_data['days_to_outcome'] = days_to_outcome
            row_data['seq_length'] = current_seq_len
            
            # 计算距离上一次检查的天数差 (Time-LSTM 关键特征 delta_t)
            row_data['delta_t_days'] = (current_date - input_events[i-1][CONFIG['time_col']]).days if i > 0 else 0
            
            # 默认给所有窗口赋予 0 (用来处理 V1 这种长度不够的垫脚石)
            for w in CONFIG['windows']:
                row_data[f'label_{w}m'] = 0
                row_data[f'mask_{w}m'] = 0
                
            # 只有当当前序列长度 >= min_input_seq_len 时，才真正进行打标计算。
            # 这样负样本(即使要求总长>=6)依然能贡献 V2, V3, V4... 等早期的截断样本，充分利用数据。
            current_min_seq_len = CONFIG['min_input_seq_len'] if is_positive else CONFIG['min_neg_input_seq_len']
            if current_seq_len >= current_min_seq_len:
                months_to_outcome = days_to_outcome / 30.0
                y_vec = []
                mask_vec = []
                
                for w in CONFIG['windows']:
                    if is_positive:
                        if months_to_outcome <= w:
                            y_vec.append(1); mask_vec.append(1)
                        else:
                            y_vec.append(0); mask_vec.append(1)
                    else:
                        if months_to_outcome >= w:
                            y_vec.append(0); mask_vec.append(1)
                        else:
                            y_vec.append(0); mask_vec.append(0)
                
                # 如果这个截断点有任何一个窗口有效，才把掩码和标签赋进去
                if sum(mask_vec) > 0:
                    for idx, w in enumerate(CONFIG['windows']):
                        row_data[f'label_{w}m'] = y_vec[idx]
                        row_data[f'mask_{w}m'] = mask_vec[idx]
                    has_valid_anchor = True
            
            # 把这条记录加入病人的时间线中
            patient_rows.append(row_data)
            
        # 如果这个病人的时间线里，至少存在一个有效的预测锚点
        if has_valid_anchor:
            processed_rows.extend(patient_rows)
            
    # 4. 保存结果
    df_result = pd.DataFrame(processed_rows)
    
    print("\n" + "="*65)
    print(">>> 样本流失漏斗分析 <<<")
    print(f"原始有效病人总数: {len(valid_patients)}")
    print(f"[-] 因正样本总记录不足 {CONFIG['min_input_seq_len'] + 1} 次被剔除: {drop_due_to_pos_len} 人")
    print(f"[-] 因负样本总记录不足 {CONFIG['min_neg_input_seq_len'] + 1} 次被剔除: {drop_due_to_neg_len} 人")
    print(f"[+] 最终成功进入扩充队列的病人数: {kept_patients} 人")
    print("="*65)

    if len(df_result) > 0:
        df_result.to_csv(CONFIG['output_path'], index=False, encoding='gb18030')
        print(f"\nSaved labeled data to {CONFIG['output_path']}")
        print(f"Total rows (including history): {len(df_result)}")
        
        # 统计真实产生的序列样本总数 (有多少行 mask_sum > 0)
        mask_cols = [f'mask_{w}m' for w in CONFIG['windows']]
        valid_samples = df_result[df_result[mask_cols].sum(axis=1) > 0]
        print(f"Total sequence samples (after expansion): {len(valid_samples)}")
    else:
        print("\nError: No samples generated. Data criteria might be too strict.")
        return

    # 5. 统计分布
    print("\n" + "="*65)
    print(f"{'Window':<10} | {'Pos Samples':<15} | {'Neg Samples':<15} | {'Pos Ratio':<10}")
    print("-" * 65)
    
    for w in CONFIG['windows']:
        lbl_col = f'label_{w}m'
        msk_col = f'mask_{w}m'
        
        valid_df = df_result[df_result[msk_col] == 1]
        n_pos = (valid_df[lbl_col] == 1).sum()
        n_neg = (valid_df[lbl_col] == 0).sum()
        total = n_pos + n_neg
        ratio = n_pos / total if total > 0 else 0
        
        print(f"{w}m{'':<8} | {n_pos:<15} | {n_neg:<15} | {ratio:.2%}")
    print("="*65)

if __name__ == "__main__":
    generate_labels()