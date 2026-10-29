import os

import pandas as pd

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score

from captum.attr import IntegratedGradients, GradientShap

import shap

import warnings

import copy

import gc

import joblib

import matplotlib

from tqdm import tqdm  # 引入 tqdm



matplotlib.use('Agg')

import matplotlib.pyplot as plt



warnings.filterwarnings('ignore')



# ================= 0. 设备配置 (强制使用 GPU 0) =================

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")



# ================= 1. 全局配置 =================

GLOBAL_CONFIG = {

    'seed': 42,

    'labeled_data_path': r"D:\wzy\zyc\point_name_data\final_data\base_indicator_loose_complete_nocut_10_final_len3_121824.csv",

    'backbone_path': r"best_pretrained_model_base_indicator_44_missing_10.pth",

    'preprocess_path': r"preprocess_base_indicator_44_missing_10.pkl",

    'save_root': r"D:\wzy\zyc\point_name_data\Experiments\Final_Experiment_pretrain_Missing_Data_10_len3_121824_nocut",

   

    'windows': [12, 18, 24],

    'target_cols': [f'label_{w}m' for w in [12, 18, 24]],

    'mask_cols': [f'mask_{w}m' for w in [12, 18, 24]],

   

    'd_model': 256, 'nhead': 8, 'num_layers': 6, 'dim_feedforward': 1024,

    'dropout': 0.2, 'activation': "reglu", 'ple_bins': 64,

    'seq_hidden': 256, # 统一时序模型的隐藏层维度为 1024

   

    'batch_size': 128,

    'epochs_stage1': 3, 'lr_stage1': 1e-4,

    'epochs_stage2': 3, 'lr_stage2_head': 1e-4, 'lr_stage2_backbone': 1e-5

}



QUANTILE_COLS = [

    '谷草转氨酶', '谷丙转氨酶'

]



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

ALL_FEATURE_NAMES = NUM_COLS + CAT_COLS





# ================= 2. 数据处理与 Dataset =================

def load_scalers(path):

    bundle = joblib.load(path)

    return bundle['scalers']



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

    def __init__(self, data_dict): self.data = data_dict

    def __getitem__(self, idx):

        item = self.data[idx]

       

        # 强制截断：最多只取最近的 6 次记录（或你认为合理的上限，如 10）

        max_len = 10

       

        num = item['num'][-max_len:]

        cat = item['cat'][-max_len:]

        mask_num = item['mask_num'][-max_len:]

        mask_cat = item['mask_cat'][-max_len:]

        deltas = item['deltas'][-max_len:]

       

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





# ================= 3. 模型组件 =================

class ReGLU(nn.Module):

    def forward(self, x): a, b = x.chunk(2, dim=-1); return a * F.gelu(b)



class FeatureTokenizer(nn.Module):

    def __init__(self, num_cols, cat_cols, n_bins=64, d_model=256):

        super().__init__()

        self.num_features = len(num_cols); self.cat_features = len(cat_cols)

        self.register_buffer('bin_edges', torch.zeros(self.num_features, n_bins + 1))

       

        self.num_weights = nn.Parameter(torch.zeros(1, self.num_features, n_bins, d_model))

        self.num_bias = nn.Parameter(torch.zeros(1, self.num_features, d_model))

       

        self.cat_embeddings = nn.ModuleList([nn.Embedding(2, d_model) for _ in range(self.cat_features)])

       

        self.num_missing_emb = nn.Parameter(torch.zeros(1, self.num_features, d_model))

        self.cat_missing_emb = nn.Parameter(torch.zeros(1, self.cat_features, d_model))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

       

    def forward_num(self, x_num):

        x_ex = x_num.unsqueeze(-1)

        edges = self.bin_edges.unsqueeze(0)

        v = torch.clamp((x_ex - edges[:, :, :-1]) / (edges[:, :, 1:] - edges[:, :, :-1] + 1e-6), 0.0, 1.0).unsqueeze(-1)

        return torch.sum(v * self.num_weights, dim=2) + self.num_bias



    def forward(self, x_num, x_cat, mask_num, mask_cat, x_cat_emb_pre=None):

        B = x_num.shape[0]

       

        x_num_emb = self.forward_num(x_num)

        m_n = mask_num.unsqueeze(-1)

        x_num_emb = x_num_emb * m_n + self.num_missing_emb * (1.0 - m_n)

       

        if x_cat_emb_pre is None:

            cat_embs = [layer(torch.clamp(x_cat[:, i], 0, 1).long()) for i, layer in enumerate(self.cat_embeddings)]

            x_cat_emb_pre = torch.stack(cat_embs, dim=1)

           

        m_c = mask_cat.unsqueeze(-1)

        x_cat_emb = x_cat_emb_pre * m_c + self.cat_missing_emb * (1.0 - m_c)

           

        cls_tokens = self.cls_token.expand(B, -1, -1)

        return torch.cat((cls_tokens, x_num_emb, x_cat_emb), dim=1)



class FTTransformerLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout):

        super().__init__()

        self.norm1 = nn.LayerNorm(d_model); self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model); self.linear1 = nn.Linear(d_model, 2 * dim_feedforward)

        self.reglu = ReGLU(); self.linear2 = nn.Linear(dim_feedforward, d_model); self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])

        x = x + self.dropout(self.linear2(self.reglu(self.linear1(self.norm2(x)))))

        return x



class FTTransformerBackbone(nn.Module):

    def __init__(self, tokenizer, config):

        super().__init__()

        self.tokenizer = tokenizer; self.emb_norm = nn.LayerNorm(config['d_model'])

        self.layers = nn.ModuleList([FTTransformerLayer(config['d_model'], config['nhead'], config['dim_feedforward'], config['dropout']) for _ in range(config['num_layers'])])

    def forward(self, x_num, x_cat, mask_num, mask_cat, x_cat_emb_pre=None):

        x = self.emb_norm(self.tokenizer(x_num, x_cat, mask_num, mask_cat, x_cat_emb_pre))

        for layer in self.layers: x = layer(x)

        return x



# 时序结构一：TimeLSTM

class TimeLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.hidden_size = hidden_size

        self.W_x = nn.Linear(input_size, hidden_size * 4)

        self.W_h = nn.Linear(hidden_size, hidden_size * 4)

        self.W_d = nn.Linear(1, hidden_size)

    def forward(self, x, state, dt):

        h_prev, c_prev = state

        gamma = torch.exp(-torch.relu(self.W_d(dt)))

        c_prev_discounted = c_prev * gamma

        gates = self.W_x(x) + self.W_h(h_prev)

        i_gate, f_gate, c_candidate, o_gate = gates.chunk(4, dim=-1)

        c_next = torch.sigmoid(f_gate) * c_prev_discounted + torch.sigmoid(i_gate) * torch.tanh(c_candidate)

        h_next = torch.sigmoid(o_gate) * torch.tanh(c_next)

        return h_next, c_next



# 时序结构二：Temporal Transformer (数据量不大，采用 1 层结构)

# ================= 补充组件：连续时间表征 (Time2Vec) =================

class ContinuousTimeEmbedding(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.linear = nn.Linear(1, hidden_dim)

        self.periodic = nn.Linear(1, hidden_dim)



    def forward(self, t):

        return self.linear(t) + torch.sin(self.periodic(t))





class TemporalTransformer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=1, dropout=0.3):

        super().__init__()

        self.feat_proj = nn.Linear(input_dim, hidden_dim)

        self.time_emb = ContinuousTimeEmbedding(hidden_dim)



        encoder_layer = nn.TransformerEncoderLayer(

            d_model=hidden_dim,

            nhead=num_heads,

            dim_feedforward=hidden_dim * 2,

            batch_first=True,

            dropout=dropout,

            activation="gelu",

            norm_first = True

        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)



    def forward(self, x_seq, d, lengths):

            B, T, _ = x_seq.shape



            d_norm = torch.clamp(d, 0, 3650) / 365.0

            cum_time = torch.cumsum(d_norm, dim=1)



            x = self.feat_proj(x_seq) + self.time_emb(cum_time)



            pad_mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1)

            causal_mask = torch.triu(

                torch.ones(T, T, device=x.device) * -1e9,

                diagonal=1

            )



            out = self.transformer(

                x,

                mask=causal_mask,

                src_key_padding_mask=pad_mask

            )



            # 直接根据真实序列长度，提取最后一个有效时间步的特征

            batch_indices = torch.arange(B, device=x.device)

            last_step_indices = lengths - 1

            last_h = out[batch_indices, last_step_indices, :]



            return last_h





class CumulativeRiskHead(nn.Module):

    def __init__(self, hidden_dim, num_windows=4):

        super().__init__()

        self.base_fc = nn.Linear(hidden_dim, 1)

        self.inc_fc = nn.Linear(hidden_dim, num_windows)

        self.register_buffer('cumsum_mask', torch.tril(torch.ones(num_windows, num_windows)))

    def forward(self, x):

        return self.base_fc(x) + torch.matmul(F.relu(self.inc_fc(x)), self.cumsum_mask.t())



class RiskPredictor(nn.Module):

    def __init__(self, backbone_sd, seq_hidden, num_windows=4, seq_type='lstm'):

        super().__init__()

        self.seq_hidden = seq_hidden

        self.seq_type = seq_type

        self.backbone = FTTransformerBackbone(FeatureTokenizer(NUM_COLS, CAT_COLS, GLOBAL_CONFIG['ple_bins'], GLOBAL_CONFIG['d_model']), GLOBAL_CONFIG)

        if backbone_sd is not None:

            self.backbone.load_state_dict(backbone_sd, strict=False)

            assert not torch.all(self.backbone.tokenizer.bin_edges == 0), "警告: bin_edges 未加载或全0，PLE 失效！"

           

        if self.seq_type == 'lstm':

            self.seq_model = TimeLSTMCell(GLOBAL_CONFIG['d_model'], seq_hidden)

        else: # Temporal Transformer

            self.seq_model = TemporalTransformer(GLOBAL_CONFIG['d_model'], seq_hidden, num_layers=1)

           

        self.head = CumulativeRiskHead(seq_hidden, num_windows)

       

    def forward(self, n, c, mn, mc, d, lengths, c_emb_pre=None):

        B, T = n.size(0), n.size(1)

       

        n_flat = n.reshape(B*T, -1)

        c_flat = c.reshape(B*T, -1) if c is not None else None

        mn_flat = mn.reshape(B*T, -1)

        mc_flat = mc.reshape(B*T, -1)

        c_emb_flat = c_emb_pre.reshape(B*T, c_emb_pre.shape[2], -1) if c_emb_pre is not None else None

       

        feat = self.backbone(n_flat, c_flat, mn_flat, mc_flat, c_emb_flat)

        x_seq = feat[:, 0, :].reshape(B, T, -1)

       

        if self.seq_type == 'lstm':

            h, cell_state = torch.zeros(B, self.seq_hidden).to(n.device), torch.zeros(B, self.seq_hidden).to(n.device)

            last_h = torch.zeros(B, self.seq_hidden).to(n.device)

            for t in range(T):

                h, cell_state = self.seq_model(x_seq[:, t], (h, cell_state), d[:, t])

                for i in range(B):

                    if t == lengths[i] - 1: last_h[i] = h[i]

        else: # transformer

            last_h = self.seq_model(x_seq, d, lengths)

           

        return self.head(last_h)





# ================= 4. Captum IG & SHAP (适配Missing) =================

class CaptumModelWrapper(nn.Module):

    def __init__(self, model, target_idx=None):

        super().__init__()

        self.model = model

        self.target_idx = target_idx



    def forward(self, n, c_emb_pre, d, lengths, m_n, m_c):

        logits = self.model(n, None, m_n, m_c, d, lengths, c_emb_pre=c_emb_pre)

        if self.target_idx is not None:

            return logits[:, self.target_idx].unsqueeze(1)

        else:

            return logits.sum(dim=1, keepdim=True)



def compute_attributions(model, n, c, mn, mc, d, lengths, target_idx=None, method='ig', steps=50):

    model.eval()

    B, T, C_len = c.shape

   

    baseline_n = torch.zeros_like(n)

    baseline_d = torch.zeros_like(d)

   

    c_flat = c.reshape(B*T, -1)

    with torch.no_grad():

        actual_c_embs, baseline_c_embs = [], []

        baseline_c_flat = torch.zeros_like(c_flat, dtype=torch.long)

       

        for i, layer in enumerate(model.backbone.tokenizer.cat_embeddings):

            actual_c_embs.append(layer(torch.clamp(c_flat[:, i], 0, 1)))

            baseline_c_embs.append(layer(baseline_c_flat[:, i]))

           

        actual_c_embs = torch.stack(actual_c_embs, dim=1).reshape(B, T, C_len, -1)

        baseline_c_embs = torch.stack(baseline_c_embs, dim=1).reshape(B, T, C_len, -1)



    wrapper = CaptumModelWrapper(model, target_idx=target_idx)

   

    if method == 'ig':

        explainer = IntegratedGradients(wrapper)

        attributions = explainer.attribute(

            inputs=(n, actual_c_embs, d),

            baselines=(baseline_n, baseline_c_embs, baseline_d),

            additional_forward_args=(lengths, mn, mc),

            target=0, n_steps=steps, return_convergence_delta=False,

            internal_batch_size=4

        )

    elif method == 'shap':

        explainer = GradientShap(wrapper)

        attributions = explainer.attribute(

            inputs=(n, actual_c_embs, d),

            baselines=(baseline_n, baseline_c_embs, baseline_d),

            additional_forward_args=(lengths, mn, mc),

            target=0, n_samples=5, stdevs=0.05

        )

   

    attr_n, attr_c_emb, attr_d = attributions

    attr_c = attr_c_emb.sum(dim=-1)

    attr_combined = torch.cat([attr_n, attr_c], dim=-1)

   

    attr_summary = torch.zeros(B, attr_combined.size(-1)).to(n.device)

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

        # p:g 格式化会去掉末尾多余的零 (如 1.0 变为 1, 0.1 保持 0.1)

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





# ================= 6. 单折训练流程 (含 tqdm) =================

def train_single_fold(train_dl, val_dl, test_dl, backbone_sd, seq_type='lstm', pos_weights=None, compute_attr_flag=False, save_path = None):

    model = RiskPredictor(backbone_sd, GLOBAL_CONFIG['seq_hidden'], num_windows=len(GLOBAL_CONFIG['windows']), seq_type=seq_type).to(device)

    stages = [('Stage1', GLOBAL_CONFIG['epochs_stage1'], GLOBAL_CONFIG['lr_stage1']),

              ('Stage2', GLOBAL_CONFIG['epochs_stage2'], GLOBAL_CONFIG['lr_stage2_head'])]

    best_val_loss = float('inf'); best_model_wts = copy.deepcopy(model.state_dict())

   

    for stage_name, epochs, lr in stages:

        if stage_name == 'Stage1':

            for p in model.backbone.parameters(): p.requires_grad = False

            opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        elif stage_name == 'Stage2':

            for p in model.backbone.parameters(): p.requires_grad = True

            backbone_ids = set(id(p) for p in model.backbone.parameters())

            head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

            opt = optim.AdamW([{'params': model.backbone.parameters(), 'lr': GLOBAL_CONFIG['lr_stage2_backbone']},

                               {'params': head_params, 'lr': lr}])

           

        for ep in range(epochs):

            model.train()

            train_loss_accum = 0.0

           

            # 引入 tqdm 进行进度可视化

            train_pbar = tqdm(train_dl, desc=f"[{seq_type.upper()}] {stage_name} Ep {ep+1}/{epochs} Train", leave=False)

            for n, c, mn, mc, d, l, m, le, _ in train_pbar:

                n, c, mn, mc, d, l, m, le = n.to(device), c.to(device), mn.to(device), mc.to(device), d.to(device), l.to(device), m.to(device), le.to(device)

                opt.zero_grad()

                loss = masked_loss(model(n, c, mn, mc, d, le), l, m, pos_weights)

                loss.backward();

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                opt.step()

                train_loss_accum += loss.item()

                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

               

            model.eval()

            val_loss = 0

            val_pbar = tqdm(val_dl, desc=f"[{seq_type.upper()}] {stage_name} Ep {ep+1}/{epochs} Val", leave=False)

            with torch.no_grad():

                for n, c, mn, mc, d, l, m, le, _ in val_pbar:

                    n, c, mn, mc, d, l, m, le = n.to(device), c.to(device), mn.to(device), mc.to(device), d.to(device), l.to(device), m.to(device), le.to(device)

                    loss = masked_loss(model(n, c, mn, mc, d, le), l, m, pos_weights).item()

                    val_loss += loss

                    val_pbar.set_postfix({'loss': f'{loss:.4f}'})

                   

            if val_loss < best_val_loss:

                best_val_loss = val_loss; best_model_wts = copy.deepcopy(model.state_dict())



    model.load_state_dict(best_model_wts)
    if save_path:
        torch.save(best_model_wts, save_path) # === 新增保存逻辑 ===
        print(f"[{seq_type.upper()}] Best model weights saved to {save_path}")
    model.eval()

   

    val_probs, val_targets, val_masks = [], [], []

    with torch.no_grad():

        for n, c, mn, mc, d, l, m, le, _ in tqdm(val_dl, desc=f"[{seq_type.upper()}] Validation Set Check", leave=False):

            n, c, mn, mc, d, l, m, le = n.to(device), c.to(device), mn.to(device), mc.to(device), d.to(device), l.to(device), m.to(device), le.to(device)

            val_probs.append(torch.sigmoid(model(n, c, mn, mc, d, le)).cpu().numpy())

            val_targets.append(l.cpu().numpy())

            val_masks.append(m.cpu().numpy())

    fold_opt_ths = find_best_thresholds(np.concatenate(val_targets), np.concatenate(val_probs), np.concatenate(val_masks), len(GLOBAL_CONFIG['windows']))

   

    test_probs, test_targets, test_masks, test_pids = [], [], [], []

    num_wins = len(GLOBAL_CONFIG['windows'])

    patient_ig, window_igs = [], {w: [] for w in range(num_wins)}

    patient_feats = []

   

    test_pbar = tqdm(test_dl, desc=f"[{seq_type.upper()}] Testing & Captum Attributions", leave=False)

    for n, c, mn, mc, d, l, m, le, pids in test_pbar:

        n, c, mn, mc, d, l, m, le = n.to(device), c.to(device), mn.to(device), mc.to(device), d.to(device), l.to(device), m.to(device), le.to(device)

        with torch.no_grad():

            logits = model(n, c, mn, mc, d, le)

            test_probs.append(torch.sigmoid(logits).cpu().numpy())

            test_targets.append(l.cpu().numpy())

            test_masks.append(m.cpu().numpy())

            test_pids.extend(pids)

           

    #     if compute_attr_flag:

    #         B = n.shape[0]

    #         feat_comb = torch.cat([n, c.float()], dim=-1)

    #         f_sum = torch.zeros(B, feat_comb.size(-1)).to(n.device)

    #         for b in range(B):

    #             if le[b] > 0: f_sum[b] = feat_comb[b, :le[b], :].mean(dim=0)

    #         patient_feats.append(f_sum.cpu().numpy())



    #         patient_ig.append(compute_attributions(model, n, c, mn, mc, d, le, target_idx=None, method='ig').cpu().numpy())

       

    #         for w_idx in range(num_wins):

    #             window_igs[w_idx].append(compute_attributions(model, n, c, mn, mc, d, le, target_idx=w_idx, method='ig').cpu().numpy())

    attr_results = None

    # attr_results = {

    #     'feats': np.concatenate(patient_feats, axis=0) if compute_attr_flag else None,

    #     'ig_patient': np.concatenate(patient_ig, axis=0) if compute_attr_flag else None,

    #     'ig_windows': {w: np.concatenate(window_igs[w], axis=0) for w in range(num_wins)} if compute_attr_flag else None

    # }

   

    del model

    gc.collect()

    torch.cuda.empty_cache()

   

    return np.concatenate(test_probs), np.concatenate(test_targets), np.concatenate(test_masks), np.array(test_pids), attr_results, fold_opt_ths



# ================= 7. 实验 Runner =================

def run_experiment(exp_name, seq_type, use_class_weights, all_pids, pid_labels, pid_to_samples, sd_s2, save_dir):

    print(f"\n{'='*20} Starting Experiment: {exp_name} ({seq_type.upper()}) {'='*20}")

    os.makedirs(save_dir, exist_ok=True)

   

    all_fold_metrics = []

   

    # 8:1:1 单次切分

    train_val_pids, test_pids, train_val_labels, test_labels = train_test_split(

        all_pids, pid_labels, test_size=0.1, random_state=GLOBAL_CONFIG['seed'], stratify=pid_labels

    )

    train_pids, val_pids = train_test_split(

        train_val_pids, test_size=1/9, random_state=GLOBAL_CONFIG['seed'], stratify=train_val_labels

    )

   

    print(f"Split Info -> Train: {len(train_pids)} | Val: {len(val_pids)} | Test: {len(test_pids)}")

   

    pos_weights_tensor = None

    if use_class_weights:

        pos_weights_list = []

        for i in range(len(GLOBAL_CONFIG['windows'])):

            all_t = np.array([s['label'][i] for pid in train_pids for s in pid_to_samples[pid]])

            all_m = np.array([s['mask'][i] for pid in train_pids for s in pid_to_samples[pid]])

            valid_t = all_t[all_m == 1]

            n_pos = np.sum(valid_t == 1)

            weight = np.sum(valid_t == 0) / n_pos if n_pos > 0 else 1.0

            pos_weights_list.append(weight)

        pos_weights_tensor = torch.FloatTensor(pos_weights_list).to(device)

        print(f"Applied Class Weights: {[round(w, 2) for w in pos_weights_list]}")



    def get_samples(pids):

        lst = []

        for p in pids: lst.extend(pid_to_samples[p])

        return lst



    tr_dl = DataLoader(SequenceRiskDataset(get_samples(train_pids)), batch_size=GLOBAL_CONFIG['batch_size'], shuffle=True, collate_fn=seq_collate, pin_memory=True)

    va_dl = DataLoader(SequenceRiskDataset(get_samples(val_pids)), batch_size=GLOBAL_CONFIG['batch_size'], shuffle=False, collate_fn=seq_collate, pin_memory=True)

    te_dl = DataLoader(SequenceRiskDataset(get_samples(test_pids)), batch_size=32, shuffle=False, collate_fn=seq_collate, pin_memory=True)

   

    # 传入 seq_type 控制使用哪种时序模型
    model_save_path = os.path.join(save_dir, f"best_model_{exp_name}.pth")
    probs, targets, masks, pids_arr, attr_res, fold_ths = train_single_fold(

        tr_dl, va_dl, te_dl, copy.deepcopy(sd_s2), seq_type, pos_weights_tensor, compute_attr_flag=False, save_path = model_save_path

    )



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

    num_wins = len(GLOBAL_CONFIG['windows'])

    all_fold_metrics.extend(evaluate_single_fold(targets, probs, masks, pids_arr, [0.5]*num_wins, 'DefaultTh(0.5)'))

    all_fold_metrics.extend(evaluate_single_fold(targets, probs, masks, pids_arr, fold_ths, 'ValidatedOptTh'))

       

    print(f"[{exp_name}] Test set evaluated. Val Opt Thresholds: {[round(x,3) for x in fold_ths]}")



    # ================= 评估指标导出 =================

    df_folds = pd.DataFrame(all_fold_metrics)

    folds_detail_path = os.path.join(save_dir, f'Metrics_Single_Fold_Detail_{exp_name}.csv')

    df_folds.to_csv(folds_detail_path, index=False, float_format='%.4f')

   

    # ================= 【新增】测试集预测结果与真值导出 =================

    result_dict = {'pid': pids_arr}

    for i, w in enumerate(GLOBAL_CONFIG['windows']):

        result_dict[f'prob_{w}m'] = probs[:, i]     # 预测概率

        result_dict[f'target_{w}m'] = targets[:, i] # 真实标签

        result_dict[f'mask_{w}m'] = masks[:, i]     # 掩码 (标识该窗口是否有有效随访)

       

    df_preds = pd.DataFrame(result_dict)

    preds_path = os.path.join(save_dir, f'Test_Predictions_{exp_name}.csv')

    df_preds.to_csv(preds_path, index=False, float_format='%.4f')

    print(f"[{exp_name}] 测试集预测结果已保存至: {preds_path}")



# ================= 8. 主函数入口 =================

def main():

    os.makedirs(GLOBAL_CONFIG['save_root'], exist_ok=True)

   

    print("Loading Pretrained Scalers (Hybrid Missing Strategy)...")

    scalers = load_scalers(GLOBAL_CONFIG['preprocess_path'])

   

    print("Loading labeled sequence data...")

    df = pd.read_csv(GLOBAL_CONFIG['labeled_data_path'], encoding='gb18030')

    X_num, X_cat, Mask_num, Mask_cat = process_features(df, scalers)

    df['__idx'] = range(len(df))

   

    pid_to_samples = {}

    grouped = df.sort_values(['patient_no', 'ExamD']).groupby('patient_no')

   

    print("Constructing sequences...")

    for pid, grp in tqdm(grouped, desc="Processing Patients"):

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

   

    all_pids = np.array(list(pid_to_samples.keys()))

    pid_labels = []

    for pid in all_pids:

        is_positive = False

        for s in pid_to_samples[pid]:

            if np.any((np.array(s['label']) == 1) & (np.array(s['mask']) == 1)):

                is_positive = True; break

        pid_labels.append(1 if is_positive else 0)

    pid_labels = np.array(pid_labels)

       

    print("Loading Pretrained DAE Weights...")

    sd_s2 = torch.load(GLOBAL_CONFIG['backbone_path'], map_location='cpu')

    if 'backbone_state_dict' in sd_s2:

        sd_s2 = sd_s2['backbone_state_dict']

    sd_s2 = {k.replace('backbone.', ''): v for k,v in sd_s2.items()} if any('backbone.' in k for k in sd_s2) else sd_s2

   

    # ======== 实验 1：TimeLSTM 对比实验 ========

    run_experiment(

        exp_name="Exp_TimeLSTM_1024",

        seq_type="lstm",

        use_class_weights=False,

        all_pids=all_pids,

        pid_labels=pid_labels,

        pid_to_samples=pid_to_samples,

        sd_s2=sd_s2,

        save_dir=os.path.join(GLOBAL_CONFIG['save_root'], "TimeLSTM_Model")

    )

   

    # ======== 实验 2：Temporal Transformer 对比实验 ========

    run_experiment(

        exp_name="Exp_TemporalTransformer_1024",

        seq_type="transformer",

        use_class_weights=False,

        all_pids=all_pids,

        pid_labels=pid_labels,

        pid_to_samples=pid_to_samples,

        sd_s2=sd_s2,

        save_dir=os.path.join(GLOBAL_CONFIG['save_root'], "TemporalTransformer_Model")

    )

                   

    print("\n✅ All Experiments Completed Successfully!")



if __name__ == "__main__":

    main()