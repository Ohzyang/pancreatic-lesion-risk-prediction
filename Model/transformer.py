import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from captum.attr import IntegratedGradients, GradientShap
import shap
import warnings
import copy
import gc 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ================= 0. 设备配置 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= 1. 全局配置 =================
# 新增：定义不同缺失率对应的文件路径
DATA_FILES = [
    (10, r"D:\wzy\zyc\point_name_data\final_data\base_indicator_loose_complete_nocut_10_final_len3_121824.csv"),
    (20, r"D:\wzy\zyc\point_name_data\final_data\base_indicator_loose_complete_nocut_20_final_len3_121824.csv"),
    (30, r"D:\wzy\zyc\point_name_data\final_data\base_indicator_loose_complete_nocut_30_final_len3_121824.csv")
]

GLOBAL_CONFIG = {
    'seed': 42,
    # 'labeled_data_path': 将在循环中动态分配
    # 'save_root': 将在循环中动态分配
    
    'windows': [12, 18, 24],
    'target_cols': [f'label_{w}m' for w in [12, 18, 24]],
    'mask_cols': [f'mask_{w}m' for w in [12, 18, 24]],
    
    # Transformer hyperparameters
    'd_model': 256,
    'nhead': 8,
    'num_layers': 2,
    'dim_feedforward': 1024,
    
    'batch_size': 1024,
    'epochs': 5, 
    'lr': 1e-4 
}

QUANTILE_COLS = ['谷草转氨酶', '谷丙转氨酶']
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
CAT_COLS = [
    'Gender', '尿白细胞', '尿胆红素', '尿胆原', '尿蛋白质', 
    '尿管型', '尿红细胞', '尿结晶', '尿葡萄糖', '尿酮体', 
    '尿亚硝酸盐', '尿液透明度', '尿液颜色', '高血压'
]
ALL_FEATURE_NAMES = NUM_COLS + CAT_COLS + ['Time_Delta_Days']


# ================= 2. 数据处理与 Dataset =================
# ================= 2. 数据处理与 Dataset =================
class SequenceRiskDataset(Dataset):
    def __init__(self, data_dict): 
        self.data = data_dict 
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 强制截断：最多只取最近的 10 次记录
        max_len = 10
        
        # 对当前的键 'x' 和 'deltas' 进行切片截断
        x = item['x'][-max_len:]
        deltas = item['deltas'][-max_len:]
        
        return (torch.FloatTensor(x),
                torch.FloatTensor(deltas).unsqueeze(1), 
                torch.FloatTensor(item['label']),
                torch.FloatTensor(item['mask']), 
                len(x), item['pid']) # 返回截断后的真实长度 len(x)
                
    def __len__(self): 
        return len(self.data)

def seq_collate(batch):
    batch.sort(key=lambda x: x[4], reverse=True)
    x, d, l, m, le, pids = zip(*batch)
    return (pad_sequence(x, batch_first=True, padding_value=0.0), 
            pad_sequence(d, batch_first=True, padding_value=0.0), 
            torch.stack(l), torch.stack(m), torch.tensor(le), pids)


# ================= 3. Transformer 模型架构 =================
class CumulativeRiskHead(nn.Module):
    def __init__(self, hidden_dim, num_windows=3):
        super().__init__()
        self.base_fc = nn.Linear(hidden_dim, 1)
        self.inc_fc = nn.Linear(hidden_dim, num_windows)
        self.register_buffer('cumsum_mask', torch.tril(torch.ones(num_windows, num_windows)))
    def forward(self, x):
        return self.base_fc(x) + torch.matmul(F.relu(self.inc_fc(x)), self.cumsum_mask.t())

class TransformerRiskPredictor(nn.Module):
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=2, dim_feedforward=1024, num_windows=3, max_seq_len=1000):
        super().__init__()
        # 特征和时间delta拼接到一起，输入维度需 +1
        self.proj = nn.Linear(input_size + 1, d_model)
        
        # 位置编码以补全时间特征以外的顺序信息
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出头保持不变
        self.head = CumulativeRiskHead(d_model, num_windows)
        
    def forward(self, x, d, lengths):
        B, T, _ = x.shape
        
        # 1. 拼接常规特征和时间间隔(delta)
        feat = torch.cat([x, d], dim=-1)
        emb = self.proj(feat) # (B, T, d_model)
        
        # 2. 注入位置编码
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        emb = emb + self.pos_emb(positions)
        
        # 3. 构造单向注意力掩码 (Causal Mask)
        # 上三角全为True（代表未来的时间步，需被遮挡）
        causal_mask_bool = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        src_mask = torch.zeros(T, T, device=x.device)
        src_mask.masked_fill_(causal_mask_bool, float('-inf'))
        
        # 4. 构造Padding掩码 (True 代表是被 Padding 补零的假时间步)
        pad_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # 5. 送入 Transformer
        out = self.transformer(
            emb, 
            mask=src_mask, 
            src_key_padding_mask=pad_mask
        )
        
        # 6. 提取每个序列的最后有效步
        last_h = out[torch.arange(B), lengths - 1]
        
        return self.head(last_h)


# ================= 4. Captum IG & SHAP =================
class CaptumModelWrapper(nn.Module):
    def __init__(self, model, target_idx=None):
        super().__init__()
        self.model = model
        self.target_idx = target_idx

    def forward(self, x, d, lengths):
        logits = self.model(x, d, lengths)
        if self.target_idx is not None: return logits[:, self.target_idx].unsqueeze(1) 
        else: return logits.sum(dim=1, keepdim=True) 

def compute_attributions(model, x, d, lengths, target_idx=None, method='ig', steps=50): 
    model.eval()
    B = x.shape[0]
    
    baseline_x = torch.zeros_like(x)
    baseline_d = torch.zeros_like(d)
    
    wrapper = CaptumModelWrapper(model, target_idx=target_idx)
    
    if method == 'ig':
        explainer = IntegratedGradients(wrapper)
        attributions = explainer.attribute(
            inputs=(x, d), baselines=(baseline_x, baseline_d),
            additional_forward_args=(lengths,), target=0, n_steps=steps, internal_batch_size=4
        )
    elif method == 'shap':
        explainer = GradientShap(wrapper)
        attributions = explainer.attribute(
            inputs=(x, d), baselines=(baseline_x, baseline_d),
            additional_forward_args=(lengths,), target=0, n_samples=5, stdevs=0.05
        )
    
    attr_x, attr_d = attributions
    attr_combined = torch.cat([attr_x, attr_d], dim=-1)
    
    attr_summary = torch.zeros(B, attr_combined.size(-1)).to(x.device)
    for b in range(B):
        if lengths[b] > 0:
            attr_summary[b] = attr_combined[b, :lengths[b], :].mean(dim=0)
            
    return attr_summary


# ================= 5. 评价指标与阈值 =================
def masked_loss(logits, targets, masks, pos_weights=None):
    if pos_weights is not None: loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weights, reduction='none')
    else: loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (loss * masks).sum() / (masks.sum() + 1e-6)

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
    
    # 基础指标
    results = {
        'AUC': roc_auc_score(y_true, y_score) if len(np.unique(y_true))>1 else 0,
        'AUPRC': average_precision_score(y_true, y_score) if len(np.unique(y_true))>1 else 0,
        'F1': f1_score(y_true, y_pred, zero_division=0), 'Acc': accuracy_score(y_true, y_pred),
        'Precision': precision, 'Recall': recall_score(y_true, y_pred, zero_division=0),
        'RR': rr, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

    # 动态生成所需的所有百分位: 0.1, 1~9, 10~100
    percents = [0.1] + list(range(1, 10)) + list(range(10, 101, 10))
    for p in percents:
        k_ratio = p / 100.0
        ppv_k, rr_k = get_topk_ppv_rr(y_true, y_score, k_ratio)
        results[f'PPV@{p:g}%'] = ppv_k
        results[f'RR@{p:g}%'] = rr_k
        
    return results


def find_best_thresholds(targets, probs, masks):
    best_ths = [0.5, 0.5, 0.5]
    grid = np.linspace(0.1, 0.9, 17) 
    for w in range(3):
        valid = masks[:, w] == 1
        if valid.sum() == 0: continue
        best_f1, best_t = -1, 0.5
        for th in grid:
            pred = (probs[valid, w] > th).astype(int)
            f1 = f1_score(targets[valid, w], pred, zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, th
        best_ths[w] = best_t
    return best_ths


# ================= 6. 单折训练流程 =================
def train_single_fold(train_dl, val_dl, test_dl, pos_weights=None, compute_attr_flag=False):
    input_size = len(NUM_COLS) + len(CAT_COLS)
    
    # 替换为 Transformer 预测模型
    model = TransformerRiskPredictor(
        input_size=input_size, 
        d_model=GLOBAL_CONFIG['d_model'], 
        nhead=GLOBAL_CONFIG['nhead'],
        num_layers=GLOBAL_CONFIG['num_layers'],
        dim_feedforward=GLOBAL_CONFIG['dim_feedforward'],
        num_windows=len(GLOBAL_CONFIG['windows'])
    ).to(device)
    
    opt = optim.AdamW(model.parameters(), lr=GLOBAL_CONFIG['lr'])
    
    best_val_loss = float('inf'); best_model_wts = copy.deepcopy(model.state_dict())
            
    for ep in range(GLOBAL_CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        # 1. 训练集进度条 (leave=False 防止刷屏)
        train_pbar = tqdm(train_dl, desc=f"Epoch [{ep+1}/{GLOBAL_CONFIG['epochs']}] Train", leave=False)
        for x, d, l, m, le, _ in train_pbar:
            x, d, l, m, le = x.to(device), d.to(device), l.to(device), m.to(device), le.to(device)
            opt.zero_grad()
            loss = masked_loss(model(x, d, le), l, m, pos_weights)
            loss.backward(); 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            train_loss += loss.item()
            # 动态更新进度条右侧的 Loss 信息
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        model.eval()
        val_loss = 0.0
        
        # 2. 验证集进度条
        val_pbar = tqdm(val_dl, desc=f"Epoch [{ep+1}/{GLOBAL_CONFIG['epochs']}] Val  ", leave=False)
        with torch.no_grad():
            for x, d, l, m, le, _ in val_pbar:
                x, d, l, m, le = x.to(device), d.to(device), l.to(device), m.to(device), le.to(device)
                batch_val_loss = masked_loss(model(x, d, le), l, m, pos_weights).item()
                val_loss += batch_val_loss
                val_pbar.set_postfix({'loss': f"{batch_val_loss:.4f}"})
        
        # 每个 Epoch 结束后，打印一次整轮的平均 Loss
        avg_train_loss = train_loss / len(train_dl)
        avg_val_loss = val_loss / len(val_dl)
        print(f"Epoch {ep+1:02d}/{GLOBAL_CONFIG['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    model.eval()
    
    val_probs, val_targets, val_masks = [], [], []
    with torch.no_grad():
        for x, d, l, m, le, _ in val_dl:
            x, d, l, m, le = x.to(device), d.to(device), l.to(device), m.to(device), le.to(device)
            val_probs.append(torch.sigmoid(model(x, d, le)).cpu().numpy())
            val_targets.append(l.cpu().numpy())
            val_masks.append(m.cpu().numpy())
    fold_opt_ths = find_best_thresholds(np.concatenate(val_targets), np.concatenate(val_probs), np.concatenate(val_masks))
    
    test_probs, test_targets, test_masks, test_pids = [], [], [], []
    
    print("\nRunning Inference on Test Set...")
    # 只做纯测试，去掉了冗余的 Captum 变量初始化
    test_pbar = tqdm(test_dl, desc="Testing", leave=False)
    
    for x, d, l, m, le, pids in test_pbar:
        # 1. 对应 DataLoader 输出的 6 个变量 (与 train/val 保持一致)
        x, d, l, m, le = x.to(device), d.to(device), l.to(device), m.to(device), le.to(device)
        
        with torch.no_grad():
            # 2. 修正模型的入参，TransformerRiskPredictor forward 只需要 (x, d, le)
            logits = model(x, d, le) 
            
            test_probs.append(torch.sigmoid(logits).cpu().numpy())
            test_targets.append(l.cpu().numpy())
            # 3. 这里的 m 就是 mask
            test_masks.append(m.cpu().numpy()) 
            test_pids.extend(pids)
            
        # ================= 归因计算代码已彻底删除 =================
                
    attr_results = None 
    
    return np.concatenate(test_probs), np.concatenate(test_targets), np.concatenate(test_masks), np.array(test_pids), attr_results, fold_opt_ths

# ================= 7. 单次实验流程 =================
def run_experiment(missing_rate):
    os.makedirs(GLOBAL_CONFIG['save_root'], exist_ok=True)
    print(f"\nLoading labeled sequence data for Missing Rate {missing_rate}%...")
    df = pd.read_csv(GLOBAL_CONFIG['labeled_data_path'], encoding='gb18030')
    
    # ---------------- 步骤 A: 提取全局 PID 并严格划分 8:1:1 ----------------
    pid_labels_dict = {}
    for pid, grp in df.groupby('patient_no'):
        masks = grp[GLOBAL_CONFIG['mask_cols']].values
        labels = grp[GLOBAL_CONFIG['target_cols']].values
        pid_labels_dict[pid] = 1 if np.any((labels == 1) & (masks == 1)) else 0
        
    all_pids = np.array(list(pid_labels_dict.keys()))
    all_labels = np.array(list(pid_labels_dict.values()))
    
    train_val_pids, test_pids, train_val_labels, _ = train_test_split(
        all_pids, all_labels, test_size=0.1, random_state=GLOBAL_CONFIG['seed'], stratify=all_labels
    )
    train_pids, val_pids = train_test_split(
        train_val_pids, test_size=1/9, random_state=GLOBAL_CONFIG['seed'], stratify=train_val_labels
    )
    print(f"Split Info -> Train: {len(train_pids)} | Val: {len(val_pids)} | Test: {len(test_pids)}")
    
    # ---------------- 步骤 B: 仅在训练集上利用非缺失数据 Fit Scaler ----------------
    print("Fitting Scalers strictly on Training set (Valid values only)...")
    train_df = df[df['patient_no'].isin(set(train_pids))].copy()
    scalers = {'qt': {}, 'std': {}}
    
    for col in QUANTILE_COLS:
        arr = pd.to_numeric(train_df[col], errors='coerce').dropna().values
        if len(arr) > 1:
            qt = QuantileTransformer(output_distribution='normal', random_state=GLOBAL_CONFIG['seed'])
            qt.fit(arr.reshape(-1, 1))
            scalers['qt'][col] = qt
            
    for col in STD_COLS:
        arr = pd.to_numeric(train_df[col], errors='coerce').dropna().values
        if len(arr) > 1:
            scalers['std'][col] = {'mean': arr.mean(), 'std': arr.std()}

    # ---------------- 步骤 C: 全局 Transform ----------------
    print("Applying Transforms to Global Data...")
    df_transformed = df.copy()
    
    # 连续变量 Transform
    for col in QUANTILE_COLS:
        if col in scalers['qt']:
            valid = ~df_transformed[col].isna()
            df_transformed.loc[valid, col] = np.clip(
                scalers['qt'][col].transform(df_transformed.loc[valid, col].values.reshape(-1,1)).ravel(), -5.0, 5.0
            )
            
    for col in STD_COLS:
        if col in scalers['std']:
            mean, std = scalers['std'][col]['mean'], scalers['std'][col]['std']
            valid = ~df_transformed[col].isna()
            df_transformed.loc[valid, col] = np.clip((df_transformed.loc[valid, col] - mean) / std, -5.0, 5.0)

    # 离散变量映射 0 -> -1, 1 -> 1. 缺失仍保持 NaN
    for col in CAT_COLS:
        df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce').map({0.0: -1.0, 1.0: 1.0})

    # ---------------- 步骤 D: 序列前向填充 (ffill) 及补 0 ----------------
    print("Forward Filling missing data per patient and filling remainder with 0...")
    feat_cols = NUM_COLS + CAT_COLS
    
    # 按病人级别前向填充
    df_transformed[feat_cols] = df_transformed.groupby('patient_no')[feat_cols].ffill()
    
    # 填充剩余的初始缺失值
    df_transformed[feat_cols] = df_transformed[feat_cols].fillna(0.0)

    # ---------------- 步骤 E: 构建模型所需序列 ----------------
    print("Constructing sequence samples...")
    pid_to_samples = {}
    grouped = df_transformed.sort_values(['patient_no', 'ExamD']).groupby('patient_no')
    
    for pid, grp in grouped:
        times = pd.to_datetime(grp['ExamD']).values
        deltas = np.zeros(len(times), dtype=np.float32)
        if len(times) > 1:
            deltas[1:] = ((times[1:] - times[:-1]) / np.timedelta64(1, 'D')) / 365.0
            
        x_seq = grp[feat_cols].values.astype(np.float32)
        masks_seq = grp[GLOBAL_CONFIG['mask_cols']].values
        labels_seq = grp[GLOBAL_CONFIG['target_cols']].values
        
        samples = []
        for i in range(len(grp)):
            if sum(masks_seq[i]) == 0: continue
            samples.append({
                'x': x_seq[:i+1], 'deltas': deltas[:i+1],
                'label': labels_seq[i], 'mask': masks_seq[i], 'pid': pid
            })
        if samples: pid_to_samples[pid] = samples
        
    def get_samples(pids):
        lst = []
        for p in pids: lst.extend(pid_to_samples[p])
        return lst
        
    tr_dl = DataLoader(SequenceRiskDataset(get_samples(train_pids)), batch_size=GLOBAL_CONFIG['batch_size'], shuffle=True, collate_fn=seq_collate, pin_memory=True)
    va_dl = DataLoader(SequenceRiskDataset(get_samples(val_pids)), batch_size=GLOBAL_CONFIG['batch_size'], shuffle=False, collate_fn=seq_collate, pin_memory=True)
    te_dl = DataLoader(SequenceRiskDataset(get_samples(test_pids)), batch_size=32, shuffle=False, collate_fn=seq_collate, pin_memory=True)
    
    # ---------------- 步骤 F: 训练与评估 ----------------
    print("\nStarting Training Transformer Baseline...")
    probs, targets, masks, pids_arr, attr_res, fold_ths = train_single_fold(tr_dl, va_dl, te_dl, pos_weights=None, compute_attr_flag=True)

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
            patient_probs.append(float(np.mean(p_probs[valid_mask]))) 
            
        if len(patient_trues) > 0:
            p_t, p_p = np.array(patient_trues), np.array(patient_probs)
            p_th = np.mean(thresholds)
            p_metrics = calculate_metrics(p_t, p_p, threshold=p_th)
            p_metrics['Evaluation_Level'] = f'Patient_Level_Overall_{label_suffix}_Imbalanced'
            p_metrics['Threshold'] = p_th
            rows.append(p_metrics)
        return rows

    all_fold_metrics.extend(evaluate_single_fold(targets, probs, masks, pids_arr, [0.5]*3, 'DefaultTh(0.5)'))
    all_fold_metrics.extend(evaluate_single_fold(targets, probs, masks, pids_arr, fold_ths, 'ValidatedOptTh'))
        
    print(f"Test set evaluated. Val Opt Thresholds: {[round(x,3) for x in fold_ths]}")
    
    df_folds = pd.DataFrame(all_fold_metrics)
    csv_name = f'Metrics_Detail_Transformer_Baseline_Missing_{missing_rate}.csv'
    df_folds.to_csv(os.path.join(GLOBAL_CONFIG['save_root'], csv_name), index=False, float_format='%.4f')
    # ================= 【新增】测试集预测结果与真值导出 =================
    result_dict = {'pid': pids_arr}
    for i, w in enumerate(GLOBAL_CONFIG['windows']):
        result_dict[f'prob_{w}m'] = probs[:, i]     # 预测概率
        result_dict[f'target_{w}m'] = targets[:, i] # 真实标签
        result_dict[f'mask_{w}m'] = masks[:, i]     # 掩码 (标识该窗口是否有有效随访)
        
    df_preds = pd.DataFrame(result_dict)
    
    # 【修改处】：定义 exp_name 变量，将缺失率等信息加入文件名，避免 NameError
    exp_name = f"Transformer_Baseline_Missing_{missing_rate}"
    
    # 将结果保存到动态分配的 save_root 目录下
    preds_path = os.path.join(GLOBAL_CONFIG['save_root'], f'Test_Predictions_{exp_name}.csv')
    
    df_preds.to_csv(preds_path, index=False, float_format='%.4f')
    print(f"[{exp_name}] 测试集预测结果已保存至: {preds_path}")
    # ---------------- 步骤 G: 可视化绘制 ----------------
    # print(f"Generating Explainability Plots for {missing_rate}%...")
    # global_patient_feats = attr_res['feats']
    # global_ig_patient = attr_res['ig_patient']
    # global_shap_patient = attr_res['shap_patient']

    # patient_level_ig, patient_level_shap = [], []
    # unique_pids = np.unique(pids_arr)
    # for pid in unique_pids:
    #     idx = np.where(pids_arr == pid)[0]
    #     patient_level_ig.append(global_ig_patient[idx].mean(axis=0)) 
    #     patient_level_shap.append(global_shap_patient[idx].mean(axis=0))
        
    # patient_level_ig, patient_level_shap = np.array(patient_level_ig), np.array(patient_level_shap)

    # def plot_global_directional(attr_matrix, title, save_path, top_k=20):
    #     importance_score = np.mean(np.abs(attr_matrix), axis=0)
    #     top_idx = np.argsort(importance_score)[-top_k:] 
    #     labels = [ALL_FEATURE_NAMES[i] for i in top_idx]
    #     pos_impact, neg_impact = [], []
    #     for i in top_idx:
    #         vals = attr_matrix[:, i]
    #         pos_vals, neg_vals = vals[vals > 0], vals[vals < 0]
    #         pos_impact.append(np.sum(pos_vals) / len(vals) if len(pos_vals) > 0 else 0)
    #         neg_impact.append(np.sum(neg_vals) / len(vals) if len(neg_vals) > 0 else 0)
            
    #     y = np.arange(len(labels))
    #     fig, ax = plt.subplots(figsize=(12, 10))
    #     ax.barh(y, pos_impact, color='#d62728', label='Positive Impact')
    #     ax.barh(y, neg_impact, color='#1f77b4', label='Negative Impact')
    #     ax.set_yticks(y)
    #     ax.set_yticklabels(labels, fontsize=11)
    #     ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
    #     ax.set_title(title, fontsize=14)
    #     ax.legend(loc='lower right')
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=300)
    #     plt.close()

    # plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
    # plt.rcParams['axes.unicode_minus'] = False 

    # plot_global_directional(patient_level_ig, f"IG Global Importance (Missing {missing_rate}%)", os.path.join(GLOBAL_CONFIG['save_root'], "IG_Global_Top20.png"))
    # plot_global_directional(patient_level_shap, f"SHAP Global Importance (Missing {missing_rate}%)", os.path.join(GLOBAL_CONFIG['save_root'], "SHAP_Global_Top20.png"))

    print(f"\n✅ Transformer Baseline Experiment for Missing {missing_rate}% Completed Successfully!")


# ================= 8. 顶层循环驱动 =================
def main():
    for missing_rate, data_path in DATA_FILES:
        print(f"\n\n{'='*60}")
        print(f"🚀 INITIATING RUN: TRANSFORMER | MISSING RATE {missing_rate}%")
        print(f"{'='*60}")
        
        # 动态修改全局配置中的路径，以便当前循环使用
        GLOBAL_CONFIG['labeled_data_path'] = data_path
        
        # 动态生成独立的保存路径，避免相互覆盖
        GLOBAL_CONFIG['save_root'] = rf"D:\wzy\zyc\point_name_data\Experiments\Final_Experiment_Transformer_endtoend_Missing_{missing_rate}_Data_len3"
        
        # 执行单次完整的训练与评估
        run_experiment(missing_rate)
        
        # 【关键】每次训练结束后执行垃圾回收与显存清空
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()