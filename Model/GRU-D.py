import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import QuantileTransformer, StandardScaler
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

# ================= 0. 设备配置 (强制使用 GPU 0) =================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# ================= 1. 全局配置 =================
# 新增：定义不同缺失率对应的文件路径
DATA_FILES = [
    (30, r"D:\wzy\zyc\point_name_data\final_data\base_indicator_loose_complete_nocut_30_final_len3_121824.csv")
]
# 基础配置 (去掉了原本写死的 labeled_data_path 和 save_root，因为我们会在循环里动态赋值)
GLOBAL_CONFIG = {
    'seed': 42,
    # 'labeled_data_path': 将在循环中动态分配
    # 'save_root': 将在循环中动态分配
    
    'windows': [12, 18, 24],
    'target_cols': [f'label_{w}m' for w in [12, 18, 24]],
    'mask_cols': [f'mask_{w}m' for w in [12, 18, 24]],
    
    'rnn_hidden': 1024,
    'batch_size': 16384,
    'epochs': 2, 'lr': 1e-4
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
ALL_FEATURE_NAMES = NUM_COLS + CAT_COLS

# ================= 2. GRU-D 数据构建器 =================
class GRUDDataset(Dataset):
    def __init__(self, data_dict): 
        self.data = data_dict 
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 强制截断：最多只取最近的 10 次记录
        max_len = 10
        
        # 对包含时间步维度的特征进行切片截断
        x = item['x'][-max_len:]
        m = item['m'][-max_len:]
        d = item['d'][-max_len:]
        x_prev = item['x_prev'][-max_len:]
        
        # 注意：这里 len(x) 会返回截断后的实际序列长度
        return (torch.FloatTensor(x), torch.FloatTensor(m), 
                torch.FloatTensor(d), torch.FloatTensor(x_prev),
                torch.FloatTensor(item['label']), torch.FloatTensor(item['mask']),
                len(x), item['pid'])
                
    def __len__(self): 
        return len(self.data)

def grud_collate(batch):
    batch.sort(key=lambda x: x[6], reverse=True)
    x, m, d, x_prev, l, label_mask, le, pids = zip(*batch)
    return (pad_sequence(x, batch_first=True, padding_value=0.0), 
            pad_sequence(m, batch_first=True, padding_value=0.0), 
            pad_sequence(d, batch_first=True, padding_value=0.0), 
            pad_sequence(x_prev, batch_first=True, padding_value=0.0), 
            torch.stack(l), torch.stack(label_mask), torch.tensor(le), pids)

# ================= 3. 原生 GRU-D 模型架构 =================
class GRUDCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_gamma_x = nn.Linear(input_size, input_size)
        self.W_gamma_h = nn.Linear(input_size, hidden_size)
        self.W_z = nn.Linear(input_size * 2 + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size * 2 + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size * 2 + hidden_size, hidden_size)
        
    def forward(self, x, m, d, h_prev, x_prev, emp_mean):
        # 计算衰减系数
        gamma_x = torch.exp(-torch.relu(self.W_gamma_x(d)))
        gamma_h = torch.exp(-torch.relu(self.W_gamma_h(d)))
        
        # 特征层面衰减 (缺失时向经验均值和上一观测值衰减)
        x_hat = m * x + (1 - m) * (gamma_x * x_prev + (1 - gamma_x) * emp_mean)
        # 隐藏状态衰减
        h_hat = gamma_h * h_prev
        
        # 拼接进入标准 GRU 逻辑
        comb = torch.cat([x_hat, h_hat, m], dim=1)
        z = torch.sigmoid(self.W_z(comb))
        r = torch.sigmoid(self.W_r(comb))
        
        comb_r = torch.cat([x_hat, r * h_hat, m], dim=1)
        h_tilde = torch.tanh(self.W_h(comb_r))
        h_curr = (1 - z) * h_hat + z * h_tilde
        
        return h_curr, x_hat

class CumulativeRiskHead(nn.Module):
    def __init__(self, hidden_dim, num_windows=3):
        super().__init__()
        self.base_fc = nn.Linear(hidden_dim, 1)
        self.inc_fc = nn.Linear(hidden_dim, num_windows)
        self.register_buffer('cumsum_mask', torch.tril(torch.ones(num_windows, num_windows)))
    def forward(self, x):
        return self.base_fc(x) + torch.matmul(F.relu(self.inc_fc(x)), self.cumsum_mask.t())

class GRUDPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, emp_mean, num_windows=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.emp_mean = nn.Parameter(torch.tensor(emp_mean, dtype=torch.float32), requires_grad=False)
        self.cell = GRUDCell(input_size, hidden_size)
        self.head = CumulativeRiskHead(hidden_size, num_windows)
        
    def forward(self, x, m, d, x_prev, lengths):
        B, T, D = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        last_h = torch.zeros(B, self.hidden_size, device=x.device)
        
        for t in range(T):
            h, _ = self.cell(x[:, t], m[:, t], d[:, t], h, x_prev[:, t], self.emp_mean)
            for b in range(B):
                if t == lengths[b] - 1:
                    last_h[b] = h[b]
        return self.head(last_h)


# ================= 4. Captum IG & SHAP =================
class CaptumWrapper(nn.Module):
    def __init__(self, model, target_idx=None):
        super().__init__()
        self.model = model
        self.target_idx = target_idx
    def forward(self, x, m, d, x_prev, lengths):
        logits = self.model(x, m, d, x_prev, lengths)
        if self.target_idx is not None:
            return logits[:, self.target_idx].unsqueeze(1)
        return logits.sum(dim=1, keepdim=True)

def compute_attributions(model, x, m, d, x_prev, lengths, target_idx=None, method='ig', steps=50):
    model.eval()
    B = x.shape[0]
    wrapper = CaptumWrapper(model, target_idx)
    baseline_x = torch.zeros_like(x)
    
    if method == 'ig':
        explainer = IntegratedGradients(wrapper)
        attr = explainer.attribute(inputs=x, baselines=baseline_x, additional_forward_args=(m, d, x_prev, lengths), target=0, n_steps=steps, internal_batch_size=4)
    elif method == 'shap':
        explainer = GradientShap(wrapper)
        attr = explainer.attribute(inputs=x, baselines=baseline_x, additional_forward_args=(m, d, x_prev, lengths), target=0, n_samples=5, stdevs=0.05)
        
    attr_summary = torch.zeros(B, x.size(-1)).to(x.device)
    for b in range(B):
        if lengths[b] > 0:
            attr_summary[b] = attr[b, :lengths[b], :].mean(dim=0)
    return attr_summary


# ================= 5. 评价指标与训练流 =================
def masked_loss(logits, targets, masks, pos_weights=None):
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weights, reduction='none')
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
    best_ths = [0.5] * 3
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

def train_single_fold(train_dl, val_dl, test_dl, emp_mean, pos_weights=None, compute_attr_flag=False):
    model = GRUDPredictor(input_size=len(ALL_FEATURE_NAMES), hidden_size=GLOBAL_CONFIG['rnn_hidden'], emp_mean=emp_mean).to(device)
    opt = optim.AdamW(model.parameters(), lr=GLOBAL_CONFIG['lr'])
    best_val_loss = float('inf'); best_model_wts = copy.deepcopy(model.state_dict())
    
    for ep in range(GLOBAL_CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        # 1. 训练集进度条 (leave=False 防止刷屏)
        train_pbar = tqdm(train_dl, desc=f"Epoch [{ep+1}/{GLOBAL_CONFIG['epochs']}] Train", leave=False)
        for x, m, d, x_prev, l, label_mask, le, _ in train_pbar:
            x, m, d, x_prev, l, label_mask, le = x.to(device), m.to(device), d.to(device), x_prev.to(device), l.to(device), label_mask.to(device), le.to(device)
            opt.zero_grad()
            loss = masked_loss(model(x, m, d, x_prev, le), l, label_mask, pos_weights)
            loss.backward(); opt.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        model.eval()
        val_loss = 0.0
        
        # 2. 验证集进度条
        val_pbar = tqdm(val_dl, desc=f"Epoch [{ep+1}/{GLOBAL_CONFIG['epochs']}] Val  ", leave=False)
        with torch.no_grad():
            for x, m, d, x_prev, l, label_mask, le, _ in val_pbar:
                x, m, d, x_prev, l, label_mask, le = x.to(device), m.to(device), d.to(device), x_prev.to(device), l.to(device), label_mask.to(device), le.to(device)
                batch_val_loss = masked_loss(model(x, m, d, x_prev, le), l, label_mask, pos_weights).item()
                val_loss += batch_val_loss
                val_pbar.set_postfix({'loss': f"{batch_val_loss:.4f}"})
                
        # 打印整轮的平均 Loss
        avg_train_loss = train_loss / len(train_dl)
        avg_val_loss = val_loss / len(val_dl)
        print(f"Epoch {ep+1:02d}/{GLOBAL_CONFIG['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    model.eval()
    
    val_probs, val_targets, val_masks = [], [], []
    with torch.no_grad():
        for x, m, d, x_prev, l, label_mask, le, _ in val_dl:
            x, m, d, x_prev, l, label_mask, le = x.to(device), m.to(device), d.to(device), x_prev.to(device), l.to(device), label_mask.to(device), le.to(device)
            val_probs.append(torch.sigmoid(model(x, m, d, x_prev, le)).cpu().numpy())
            val_targets.append(l.cpu().numpy())
            val_masks.append(label_mask.cpu().numpy())
    fold_opt_ths = find_best_thresholds(np.concatenate(val_targets), np.concatenate(val_probs), np.concatenate(val_masks))
    
    test_probs, test_targets, test_masks, test_pids = [], [], [], []
    test_probs, test_targets, test_masks, test_pids = [], [], [], []
    
    print("\nRunning Inference on Test Set...")
    # 只做纯测试，去掉了冗余的 Captum 变量初始化
    test_pbar = tqdm(test_dl, desc="Testing", leave=False)
    
    for x, m, d, x_prev, l, label_mask, le, pids in test_pbar:
        x, m, d, x_prev, l, label_mask, le = x.to(device), m.to(device), d.to(device), x_prev.to(device), l.to(device), label_mask.to(device), le.to(device)
        with torch.no_grad():
            logits = model(x, m, d, x_prev, le)
            test_probs.append(torch.sigmoid(logits).cpu().numpy())
            test_targets.append(l.cpu().numpy())
            test_masks.append(label_mask.cpu().numpy())
            test_pids.extend(pids)
            
        # ================= 归因计算代码已彻底删除 =================
    attr_results = None 
    
    return np.concatenate(test_probs), np.concatenate(test_targets), np.concatenate(test_masks), np.array(test_pids), attr_results, fold_opt_ths

# ================= 6. 单次实验流程 =================
def run_experiment(missing_rate):
    exp_name = f"GRUD_Baseline_Missing_{missing_rate}_811"
    os.makedirs(GLOBAL_CONFIG['save_root'], exist_ok=True)
    print(f"\n{'='*20} Starting Experiment: {exp_name} {'='*20}")
    
    # 1. 读入数据与基础 PID 划分 (使用当前循环对应的路径)
    print(f"Loading data from: {GLOBAL_CONFIG['labeled_data_path']}")
    df = pd.read_csv(GLOBAL_CONFIG['labeled_data_path'], encoding='gb18030')
    
    pid_labels_dict = {}
    grouped = df.groupby('patient_no')
    for pid, grp in grouped:
        masks = grp[GLOBAL_CONFIG['mask_cols']].values
        labels = grp[GLOBAL_CONFIG['target_cols']].values
        is_pos = np.any((labels == 1) & (masks == 1))
        pid_labels_dict[pid] = 1 if is_pos else 0
        
    all_pids = np.array(list(pid_labels_dict.keys()))
    all_labels = np.array(list(pid_labels_dict.values()))
    
    train_val_pids, test_pids, train_val_labels, test_labels = train_test_split(all_pids, all_labels, test_size=0.1, random_state=GLOBAL_CONFIG['seed'], stratify=all_labels)
    train_pids, val_pids = train_test_split(train_val_pids, test_size=1/9, random_state=GLOBAL_CONFIG['seed'], stratify=train_val_labels)
    
    print(f"Split Info -> Train: {len(train_pids)} | Val: {len(val_pids)} | Test: {len(test_pids)}")
    
    # 2. 仅使用 Train 集拟合 Scaler
    print("Fitting Scalers strictly on Training set...")
    train_df = df[df['patient_no'].isin(train_pids)].copy()
    scalers = {'qt': {}, 'std': {}}
    
    for col in QUANTILE_COLS:
        arr = pd.to_numeric(train_df[col], errors='coerce').dropna().values
        if len(arr) > 1:
            qt = QuantileTransformer(output_distribution='normal', random_state=GLOBAL_CONFIG['seed'])
            qt.fit(arr.reshape(-1,1))
            scalers['qt'][col] = qt
            
    for col in STD_COLS:
        arr = pd.to_numeric(train_df[col], errors='coerce').dropna().values
        if len(arr) > 1:
            scalers['std'][col] = {'mean': arr.mean(), 'std': arr.std()}
            
    # 3. 转换全局数据
    df_transformed = df.copy()
    for col in QUANTILE_COLS:
        qt = scalers['qt'].get(col)
        if qt is not None:
            valid = ~df_transformed[col].isna()
            df_transformed.loc[valid, col] = np.clip(qt.transform(df_transformed.loc[valid, col].values.reshape(-1,1)).ravel(), -5.0, 5.0)
            
    for col in STD_COLS:
        std_dict = scalers['std'].get(col)
        if std_dict is not None:
            valid = ~df_transformed[col].isna()
            df_transformed.loc[valid, col] = np.clip((df_transformed.loc[valid, col] - std_dict['mean']) / std_dict['std'], -5.0, 5.0)

    for col in CAT_COLS:
        df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce').clip(0, 1)

    # 4. 计算训练集的 Empirical Mean 供 GRU-D 使用 (仅用Train集计算)
    train_transformed_df = df_transformed[df_transformed['patient_no'].isin(train_pids)]
    emp_mean = train_transformed_df[ALL_FEATURE_NAMES].mean().fillna(0).values

    # 5. 为 GRU-D 构造特征级时序数据 (x, m, d, x_prev)
    print("Constructing GRU-D sequences (Tracking feature-level time decay)...")
    pid_to_samples = {}
    grouped_transformed = df_transformed.sort_values(['patient_no', 'ExamD']).groupby('patient_no')
    
    for pid, grp in grouped_transformed:
        times = pd.to_datetime(grp['ExamD']).values
        x_raw = grp[ALL_FEATURE_NAMES].values 
        
        m = (~np.isnan(x_raw)).astype(np.float32)
        x = np.nan_to_num(x_raw, nan=0.0)
        
        T, D = x.shape
        d = np.zeros((T, D), dtype=np.float32)
        x_prev = np.zeros((T, D), dtype=np.float32)
        
        for t in range(T):
            if t > 0:
                delta_t = (times[t] - times[t-1]) / np.timedelta64(1, 'D')
                d[t] = delta_t + (1 - m[t-1]) * d[t-1]
                x_prev[t] = m[t-1] * x[t-1] + (1 - m[t-1]) * x_prev[t-1]
        
        masks_seq = grp[GLOBAL_CONFIG['mask_cols']].values
        labels_seq = grp[GLOBAL_CONFIG['target_cols']].values
        
        samples = []
        for i in range(T):
            if sum(masks_seq[i]) == 0: continue
            samples.append({
                'x': x[:i+1], 'm': m[:i+1], 'd': d[:i+1], 'x_prev': x_prev[:i+1],
                'label': labels_seq[i], 'mask': masks_seq[i], 'pid': pid
            })
        if samples: pid_to_samples[pid] = samples

    # 6. DataLoader 加载
    def get_samples(pids):
        lst = []
        for p in pids: 
            if p in pid_to_samples: lst.extend(pid_to_samples[p])
        return lst
        
    tr_dl = DataLoader(GRUDDataset(get_samples(train_pids)), batch_size=GLOBAL_CONFIG['batch_size'], shuffle=True, collate_fn=grud_collate, pin_memory=True)
    va_dl = DataLoader(GRUDDataset(get_samples(val_pids)), batch_size=GLOBAL_CONFIG['batch_size'], shuffle=False, collate_fn=grud_collate, pin_memory=True)
    te_dl = DataLoader(GRUDDataset(get_samples(test_pids)), batch_size=32, shuffle=False, collate_fn=grud_collate, pin_memory=True)

    # 7. 训练及验证
    probs, targets, masks, pids_arr, attr_res, fold_ths = train_single_fold(tr_dl, va_dl, te_dl, emp_mean, compute_attr_flag=False)

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
            p_metrics = calculate_metrics(np.array(patient_trues), np.array(patient_probs), threshold=np.mean(thresholds))
            p_metrics['Evaluation_Level'] = f'Patient_Level_Overall_{label_suffix}_Imbalanced'
            p_metrics['Threshold'] = np.mean(thresholds)
            rows.append(p_metrics)
        return rows

    all_fold_metrics.extend(evaluate_single_fold(targets, probs, masks, pids_arr, [0.5]*3, 'DefaultTh(0.5)'))
    all_fold_metrics.extend(evaluate_single_fold(targets, probs, masks, pids_arr, fold_ths, 'ValidatedOptTh'))
    
    # 保存结果
    df_folds = pd.DataFrame(all_fold_metrics)
    df_folds.to_csv(os.path.join(GLOBAL_CONFIG['save_root'], f'Metrics_Detail_{exp_name}.csv'), index=False, float_format='%.4f')
    print(f"[{exp_name}] Test Evaluation Done! Best Thresholds: {[round(x,3) for x in fold_ths]}")
    # ================= 【新增】测试集预测结果与真值导出 =================
    result_dict = {'pid': pids_arr}
    for i, w in enumerate(GLOBAL_CONFIG['windows']):
        result_dict[f'prob_{w}m'] = probs[:, i]     # 预测概率
        result_dict[f'target_{w}m'] = targets[:, i] # 真实标签
        result_dict[f'mask_{w}m'] = masks[:, i]     # 掩码 (标识该窗口是否有有效随访)
        
    df_preds = pd.DataFrame(result_dict)
    
    # 【修改处】：将 save_dir 替换为 GLOBAL_CONFIG['save_root']
    preds_path = os.path.join(GLOBAL_CONFIG['save_root'], f'Test_Predictions_{exp_name}.csv')
    
    df_preds.to_csv(preds_path, index=False, float_format='%.4f')
    print(f"[{exp_name}] 测试集预测结果已保存至: {preds_path}")
    # # ================= 8. 特征归因与绘图 =================
    # print(f"[{exp_name}] Generating Explainability Plots...")
    # global_patient_feats = attr_res['feats']
    # global_ig_patient = attr_res['ig_patient']
    # global_shap_patient = attr_res['shap_patient']
    # global_ig_windows = attr_res['ig_windows']
    # global_shap_windows = attr_res['shap_windows']

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
    #         pos_impact.append(np.sum(vals[vals > 0]) / len(vals) if len(vals[vals > 0]) > 0 else 0)
    #         neg_impact.append(np.sum(vals[vals < 0]) / len(vals) if len(vals[vals < 0]) > 0 else 0)
            
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

    # def plot_beeswarm(attr_matrix, feat_values, title, save_path):
    #     plt.figure(figsize=(12, 10))
    #     shap.summary_plot(attr_matrix, feat_values, feature_names=ALL_FEATURE_NAMES, show=False, max_display=20)
    #     vmin, vmax = np.percentile(attr_matrix, 1), np.percentile(attr_matrix, 99)
    #     abs_max = max(abs(vmin), abs(vmax)) * 1.5
    #     if abs_max < 1e-3: abs_max = 0.1
    #     plt.xlim(-abs_max, abs_max)
    #     plt.title(title, fontsize=14, pad=20)
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=300)
    #     plt.close()

    # plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
    # plt.rcParams['axes.unicode_minus'] = False 

    # plot_global_directional(patient_level_ig, f"[{exp_name}] IG Global Importance", os.path.join(GLOBAL_CONFIG['save_root'], "IG_Global_Top20.png"))
    # plot_global_directional(patient_level_shap, f"[{exp_name}] SHAP Global Importance", os.path.join(GLOBAL_CONFIG['save_root'], "SHAP_Global_Top20.png"))
    # plot_beeswarm(global_ig_patient, global_patient_feats, f"[{exp_name}] IG Global Beeswarm", os.path.join(GLOBAL_CONFIG['save_root'], "IG_Global_Beeswarm.png"))
    # plot_beeswarm(global_shap_patient, global_patient_feats, f"[{exp_name}] SHAP Global Beeswarm", os.path.join(GLOBAL_CONFIG['save_root'], "SHAP_Global_Beeswarm.png"))

    # for w_idx, w in enumerate(GLOBAL_CONFIG['windows']):
    #     plot_global_directional(global_ig_windows[w_idx], f"[{exp_name}] IG (Window {w}m)", os.path.join(GLOBAL_CONFIG['save_root'], f"IG_Win_{w}m.png"))
    #     plot_beeswarm(global_ig_windows[w_idx], global_patient_feats, f"[{exp_name}] IG Beeswarm (Window {w}m)", os.path.join(GLOBAL_CONFIG['save_root'], f"IG_Win_{w}m_Beeswarm.png"))
        
    print(f"\n✅ Finished Experiment for Missing Rate {missing_rate}%!")

# ================= 7. 顶层循环驱动 =================
def main():
    for missing_rate, data_path in DATA_FILES:
        print(f"\n\n{'#'*60}")
        print(f"🚀 INITIATING RUN: MISSING RATE {missing_rate}%")
        print(f"#{'#'*59}")
        
        # 动态修改全局配置中的路径，以便当前循环使用
        GLOBAL_CONFIG['labeled_data_path'] = data_path
        
        # 动态修改保存路径，防止不同缺失率的结果相互覆盖
        # 比如：.../Final_Experiment_GRUD_Baseline_missing_10_data_len4
        GLOBAL_CONFIG['save_root'] = rf"D:\wzy\zyc\point_name_data\Experiments\Final_Experiment_GRUD_Baseline_missing_{missing_rate}_data_len3"
        
        # 执行完整的单次训练
        run_experiment(missing_rate)
        
        # 【关键】每次训练完一个数据集后，强制回收内存并清空显存
        # 否则 3 次庞大的 Captum 计算叠加会导致 CUDA Out Of Memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
