import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import xgboost as xgb
import warnings
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg') 
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ================= 0. 设备配置 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ================= 1. 全局配置 =================
GLOBAL_CONFIG = {
    'seed': 42,
    'save_root': r"D:\wzy\zyc\point_name_data\Experiments\Final_Experiment_Pure_XGBoost",
    
    'windows': [12, 18, 24],
    'target_cols': [f'label_{w}m' for w in [12, 18, 24]],
    'mask_cols': [f'mask_{w}m' for w in [12, 18, 24]],
    
    'batch_size': 1024,
}

QUANTILE_COLS = ['谷草转氨酶', '谷丙转氨酶']

# 连续型特征 (30个)
NUM_COLS = [
    'Age', '白细胞WBC', '单核细胞百分比', '单核细胞计数', '红细胞RBC', 
    '红细胞分布宽度RDWCV', '红细胞分布宽度RDWSD', '淋巴细胞百分比', 
    '淋巴细胞计数', '平均红细胞血红蛋白含量MCH', '平均红细胞血红蛋白浓度MCHC', 
    '平均血小板体积', '嗜碱性粒细胞百分比', '嗜碱性粒细胞计数', 
    '嗜酸性粒细胞百分比', '嗜酸性粒细胞计数', '血红蛋白', '血小板计数', 
    '血小板体积分布宽度', '血小板压积', '中性粒细胞百分比', '中性粒细胞计数', 
    '尿比重', '尿酸碱度', '空腹血糖GLU', '总胆固醇', '甘油三酯', 
    '谷丙转氨酶', '谷草转氨酶', '体质指数'
]

STD_COLS = [c for c in NUM_COLS if c not in QUANTILE_COLS]

# 离散型/分类特征 (14个) 观测值为0/1
CAT_COLS = [
    'Gender', '尿白细胞', '尿胆红素', '尿胆原', '尿蛋白质', 
    '尿管型', '尿红细胞', '尿结晶', '尿葡萄糖', '尿酮体', 
    '尿亚硝酸盐', '尿液透明度', '尿液颜色', '高血压'
]

# ================= 2. 动态 Scaler 拟合与特征处理 =================
def fit_scalers_on_train(df, train_pids):
    """仅在训练集患者数据上拟合 Scaler 以防止数据泄露"""
    train_df = df[df['patient_no'].isin(train_pids)]
    scalers = {'std': {}, 'qt': {}}
    
    # 1. 拟合 QuantileTransformer
    for col in QUANTILE_COLS:
        qt = QuantileTransformer(output_distribution='normal', random_state=GLOBAL_CONFIG['seed'])
        arr = pd.to_numeric(train_df[col], errors="coerce").values
        valid_data = arr[np.isfinite(arr)].reshape(-1, 1)
        if len(valid_data) > 0:
            qt.fit(valid_data)
            scalers['qt'][col] = qt
            
    # 2. 拟合 StandardScaler (均值和标准差)
    for col in STD_COLS:
        arr = pd.to_numeric(train_df[col], errors="coerce").values
        valid_data = arr[np.isfinite(arr)]
        if len(valid_data) > 0:
            scalers['std'][col] = {
                'mean': valid_data.mean(), 
                'std': valid_data.std() + 1e-6 # 防止除零
            }
            
    return scalers

def process_features(df, scalers):
    mask_num = (~df[NUM_COLS].isna()).astype(np.float32).values
    mask_cat = (~df[CAT_COLS].isna()).astype(np.float32).values

    X_num = np.zeros((len(df), len(NUM_COLS)), dtype=np.float32)
    
    for col in QUANTILE_COLS:
        col_idx = NUM_COLS.index(col)
        arr = pd.to_numeric(df[col], errors="coerce").values
        valid = np.isfinite(arr)
        qt = scalers["qt"].get(col)
        if qt is not None and valid.sum() > 0:
            transformed = qt.transform(arr[valid].reshape(-1, 1)).astype(np.float32).ravel()
            out = np.zeros_like(arr, dtype=np.float32)
            out[valid] = transformed
            out = np.nan_to_num(out, nan=0.0)
            X_num[:, col_idx] = np.clip(out, -5.0, 5.0)

    for col in STD_COLS:
        col_idx = NUM_COLS.index(col)
        arr = pd.to_numeric(df[col], errors="coerce").values
        valid = np.isfinite(arr)
        if col in scalers["std"]:
            mean = scalers["std"][col]["mean"]
            std = scalers["std"][col]["std"]
            out = np.zeros_like(arr, dtype=np.float32)
            out[valid] = (arr[valid] - mean) / std
            out = np.nan_to_num(out, nan=0.0)
            X_num[:, col_idx] = np.clip(out, -5.0, 5.0)

    X_cat = df[CAT_COLS].fillna(0).astype(np.int8).clip(0, 1).values
    
    return X_num, X_cat, mask_num, mask_cat

class SequenceRiskDataset(Dataset):
    def __init__(self, data_dict, max_len=10): 
        self.data = data_dict 
        self.max_len = max_len
        
    def __getitem__(self, idx):
        item = self.data[idx]
        num = item['num'][-self.max_len:]
        cat = item['cat'][-self.max_len:]
        mask_num = item['mask_num'][-self.max_len:]
        mask_cat = item['mask_cat'][-self.max_len:]
        deltas = item['deltas'][-self.max_len:]
        
        return (torch.FloatTensor(num), torch.LongTensor(cat), 
                torch.FloatTensor(mask_num), torch.FloatTensor(mask_cat),
                torch.FloatTensor(deltas).unsqueeze(1), torch.FloatTensor(item['label']),
                torch.FloatTensor(item['mask']), len(num), item['pid'])
                
    def __len__(self): 
        return len(self.data)

def seq_collate(batch):
    batch.sort(key=lambda x: x[7], reverse=True)
    n, c, mn, mc, d, l, m, le, pids = zip(*batch)
    return (pad_sequence(n, batch_first=True, padding_value=0.0), 
            pad_sequence(c, batch_first=True, padding_value=0), 
            pad_sequence(mn, batch_first=True, padding_value=0.0), 
            pad_sequence(mc, batch_first=True, padding_value=0.0), 
            pad_sequence(d, batch_first=True, padding_value=0.0), 
            torch.stack(l), torch.stack(m), torch.tensor(le), pids)

# ================= 3. 评价指标 =================
def calculate_metrics(y_true, y_score, threshold=0.5):
    y_true, y_score = np.array(y_true), np.array(y_score)
    y_pred = (y_score > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    total = tn + fp + fn + tp
    precision = precision_score(y_true, y_pred, zero_division=0)
    incidence = (tp + fn) / total if total > 0 else 0.0
    rr = (precision / incidence) if incidence > 0 else 0.0
    
    def get_topk_ppv_rr(yt, ys, k_ratio):
        if len(yt) == 0: return 0.0, 0.0
        n_top = max(1, int(len(yt) * k_ratio))
        top_indices = np.argsort(ys)[::-1][:n_top]
        top_y_true = yt[top_indices]
        ppv_k = np.mean(top_y_true)
        overall_incidence = np.mean(yt)
        rr_k = (ppv_k / overall_incidence) if overall_incidence > 0 else 0.0
        return ppv_k, rr_k
    
    results = {
        'AUC': roc_auc_score(y_true, y_score) if len(np.unique(y_true))>1 else 0,
        'AUPRC': average_precision_score(y_true, y_score) if len(np.unique(y_true))>1 else 0,
        'F1': f1_score(y_true, y_pred, zero_division=0), 'Acc': accuracy_score(y_true, y_pred),
        'Precision': precision, 'Recall': recall_score(y_true, y_pred, zero_division=0),
        'RR': rr, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

    percents = [0.1] + list(range(1, 10)) + list(range(10, 101, 10))
    for p in percents:
        k_ratio = p / 100.0
        ppv_k, rr_k = get_topk_ppv_rr(y_true, y_score, k_ratio)
        results[f'PPV@{p:g}%'] = ppv_k
        results[f'RR@{p:g}%'] = rr_k
        
    return results

def find_best_thresholds(targets, probs, masks, num_windows):
    best_ths = [0.5] * num_windows
    grid = np.linspace(0.1, 0.9, 17) 
    for w in range(num_windows):
        valid = masks[:, w] == 1
        if valid.sum() == 0: continue
        best_f1, best_t = -1, 0.5
        for th in grid:
            pred = (probs[valid, w] > th).astype(int)
            f1 = f1_score(targets[valid, w], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, th
        best_ths[w] = best_t
    return best_ths

# ================= 4. 特征提取模块 =================
def extract_raw_flattened_features(dl, max_len=10):
    all_feats, all_targets, all_masks, all_pids = [], [], [], []
    
    with torch.no_grad():
        for n, c, mn, mc, d, l, m, le, pids in tqdm(dl, desc="Extracting Raw Features", leave=False):
            B, T = n.size(0), n.size(1)
            feat = torch.cat([n, c.float()], dim=-1)
            
            padded_seq = torch.zeros(B, max_len, feat.size(-1))
            for i in range(B):
                valid_len = min(le[i].item(), max_len)
                padded_seq[i, :valid_len, :] = feat[i, :valid_len, :]
                
            flat_feats = padded_seq.reshape(B, -1).numpy()
            
            all_feats.append(flat_feats)
            all_targets.append(l.numpy())
            all_masks.append(m.numpy())
            all_pids.extend(pids)
            
    return np.concatenate(all_feats), np.concatenate(all_targets), np.concatenate(all_masks), np.array(all_pids)

# ================= 5. XGBoost 执行主函数 =================
def run_xgboost_baseline(exp_name, train_pids, val_pids, test_pids, pid_to_samples, save_dir):
    print(f"\n{'='*20} Starting XGBoost: {exp_name} {'='*20}")
    os.makedirs(save_dir, exist_ok=True)
    
    def get_samples(pids):
        lst = []
        for p in pids: lst.extend(pid_to_samples[p])
        return lst

    tr_dl = DataLoader(SequenceRiskDataset(get_samples(train_pids), max_len=10), batch_size=GLOBAL_CONFIG['batch_size'], shuffle=False, collate_fn=seq_collate)
    va_dl = DataLoader(SequenceRiskDataset(get_samples(val_pids), max_len=10), batch_size=GLOBAL_CONFIG['batch_size'], shuffle=False, collate_fn=seq_collate)
    te_dl = DataLoader(SequenceRiskDataset(get_samples(test_pids), max_len=10), batch_size=GLOBAL_CONFIG['batch_size'], shuffle=False, collate_fn=seq_collate)
    
    # --- 提取特征 ---
    X_train, Y_train, M_train, _ = extract_raw_flattened_features(tr_dl, max_len=10)
    X_val, Y_val, M_val, _ = extract_raw_flattened_features(va_dl, max_len=10)
    X_test, Y_test, M_test, pids_test = extract_raw_flattened_features(te_dl, max_len=10)
    
    print(f"Feature shape exactly extracted: {X_train.shape}")  
    
    num_windows = len(GLOBAL_CONFIG['windows'])
    val_probs = np.zeros_like(Y_val, dtype=float)
    test_probs = np.zeros_like(Y_test, dtype=float)
    
    # --- 训练 XGBoost ---
    print("Training XGBoost Classifiers for each time window...")
    for w in range(num_windows):
        valid_tr = M_train[:, w] == 1
        if valid_tr.sum() == 0: continue
        X_tr_w, Y_tr_w = X_train[valid_tr], Y_train[valid_tr, w]
        
        clf = xgb.XGBClassifier(
            n_estimators=300, 
            max_depth=5, 
            learning_rate=0.05, 
            random_state=GLOBAL_CONFIG['seed'],
            tree_method='hist',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            eval_metric='auc'
        )
        clf.fit(X_tr_w, Y_tr_w)
        
        val_probs[:, w] = clf.predict_proba(X_val)[:, 1]
        test_probs[:, w] = clf.predict_proba(X_test)[:, 1]
        
    # --- 寻找阈值并评估 ---
    fold_ths = find_best_thresholds(Y_val, val_probs, M_val, num_windows)
    
    all_fold_metrics = []
    def evaluate_single_fold(targs, prbs, msks, pids_list, thresholds, label_suffix):
        rows = []
        for i, w in enumerate(GLOBAL_CONFIG['windows']):
            valid_idx = np.where(msks[:, i] == 1)[0]
            if len(valid_idx) > 0:
                y_t, y_p = targs[valid_idx, i], prbs[valid_idx, i]
                metrics = calculate_metrics(y_t, y_p, threshold=thresholds[i])
                metrics['Evaluation_Level'] = f'Window_{w}m_{label_suffix}_Imbalanced'
                metrics['Threshold'] = thresholds[i]
                rows.append(metrics)
        
        patient_trues, patient_probs = [], []
        unique_pids = np.unique(pids_list)
        for p in unique_pids:
            idx = np.where(pids_list == p)[0]
            p_probs, p_targets, p_masks = prbs[idx], targs[idx], msks[idx]
            valid_mask = (p_masks == 1)
            if not np.any(valid_mask): continue
            patient_trues.append(int(np.any(p_targets[valid_mask] == 1)))
            patient_probs.append(float(np.max(p_probs[valid_mask])))
            
        if len(patient_trues) > 0:
            p_t = np.array(patient_trues)
            p_p = np.array(patient_probs)
            p_th = np.mean(thresholds)
            p_metrics = calculate_metrics(p_t, p_p, threshold=p_th)
            p_metrics['Evaluation_Level'] = f'Patient_Level_Overall_{label_suffix}_Imbalanced'
            p_metrics['Threshold'] = p_th
            rows.append(p_metrics)
        return rows

    all_fold_metrics.extend(evaluate_single_fold(Y_test, test_probs, M_test, pids_test, [0.5]*num_windows, 'DefaultTh(0.5)'))
    all_fold_metrics.extend(evaluate_single_fold(Y_test, test_probs, M_test, pids_test, fold_ths, 'ValidatedOptTh'))

    # --- 结果落盘 ---
    df_folds = pd.DataFrame(all_fold_metrics)
    folds_detail_path = os.path.join(save_dir, f'Metrics_Detail_{exp_name}.csv')
    df_folds.to_csv(folds_detail_path, index=False, float_format='%.4f')
    
    result_dict = {'pid': pids_test}
    for i, w in enumerate(GLOBAL_CONFIG['windows']):
        result_dict[f'prob_{w}m'] = test_probs[:, i]
        result_dict[f'target_{w}m'] = Y_test[:, i]
        result_dict[f'mask_{w}m'] = M_test[:, i]
        
    df_preds = pd.DataFrame(result_dict)
    preds_path = os.path.join(save_dir, f'Test_Predictions_{exp_name}.csv')
    df_preds.to_csv(preds_path, index=False, float_format='%.4f')
    print(f"[{exp_name}] Test evaluation complete. Metrics saved to {save_dir}")

# ================= 6. 批量主程序入口 =================
def main():
    os.makedirs(GLOBAL_CONFIG['save_root'], exist_ok=True)
    
    file_paths = [
        r"D:\wzy\zyc\point_name_data\final_data\base_indicator_loose_complete_nocut_10_final_len3_121824.csv",
        r"D:\wzy\zyc\point_name_data\final_data\base_indicator_loose_complete_nocut_30_final_len3_121824.csv"
    ]

    for f_path in file_paths:
        if not os.path.exists(f_path):
            print(f"⚠️ 找不到文件跳过: {f_path}")
            continue
            
        base_name = os.path.basename(f_path).replace('.csv', '')
        exp_name = f"PureXGB_{base_name}"
        save_dir = os.path.join(GLOBAL_CONFIG['save_root'], exp_name)
        
        print(f"\n>>>>>> Processing Dataset: {base_name} <<<<<<")
        df = pd.read_csv(f_path, encoding='gb18030')
        df['__idx'] = range(len(df))
        
        # 1. 提前判断患者标签，用于分层抽样 (Stratified Split)
        pid_labels_dict = {}
        grouped = df.groupby('patient_no')
        for pid, grp in grouped:
            is_pos = False
            for t_col, m_col in zip(GLOBAL_CONFIG['target_cols'], GLOBAL_CONFIG['mask_cols']):
                if ((grp[t_col] == 1) & (grp[m_col] == 1)).any():
                    is_pos = True
                    break
            pid_labels_dict[pid] = 1 if is_pos else 0
            
        all_pids = np.array(list(pid_labels_dict.keys()))
        pid_labels = np.array([pid_labels_dict[p] for p in all_pids])
        
        # 2. 进行 8:1:1 划分
        train_val_pids, test_pids, train_val_labels, test_labels = train_test_split(
            all_pids, pid_labels, test_size=0.1, random_state=GLOBAL_CONFIG['seed'], stratify=pid_labels
        )
        train_pids, val_pids = train_test_split(
            train_val_pids, test_size=1/9, random_state=GLOBAL_CONFIG['seed'], stratify=train_val_labels
        )
        print(f"Split Info -> Train: {len(train_pids)} | Val: {len(val_pids)} | Test: {len(test_pids)}")
        
        # 3. 仅用 Train 集合拟合 Scalers (避免数据泄露)
        print("Fitting scalers strictly on training data...")
        scalers = fit_scalers_on_train(df, train_pids)
        
        # 4. 用拟合好的 Scalers 转换全体数据
        X_num, X_cat, Mask_num, Mask_cat = process_features(df, scalers)
        
        # 5. 构建时序样本
        pid_to_samples = {}
        grouped_sorted = df.sort_values(['patient_no', 'ExamD']).groupby('patient_no')
        for pid, grp in tqdm(grouped_sorted, desc="Constructing sequences", leave=False):
            indices = grp['__idx'].values
            times = pd.to_datetime(grp['ExamD']).values
            deltas = np.zeros(len(times))
            if len(times) > 1: deltas[1:] = np.diff(times) / np.timedelta64(1, 'D')
            
            samples = []
            for i in range(len(grp)):
                masks = [grp.iloc[i][c] for c in GLOBAL_CONFIG['mask_cols']]
                if sum(masks) == 0: continue
                labels = [grp.iloc[i][c] for c in GLOBAL_CONFIG['target_cols']]
                samples.append({
                    'num': X_num[indices[:i+1]], 'cat': X_cat[indices[:i+1]],
                    'mask_num': Mask_num[indices[:i+1]], 'mask_cat': Mask_cat[indices[:i+1]],
                    'deltas': deltas[:i+1], 'label': labels, 'mask': masks, 'pid': pid
                })
            if samples: pid_to_samples[pid] = samples
        
        # 6. 运行 XGBoost
        run_xgboost_baseline(
            exp_name=exp_name,
            train_pids=train_pids,
            val_pids=val_pids,
            test_pids=test_pids,
            pid_to_samples=pid_to_samples,
            save_dir=save_dir
        )
        
    print("\n✅ All datasets have been processed securely and modeled successfully!")

if __name__ == "__main__":
    main()