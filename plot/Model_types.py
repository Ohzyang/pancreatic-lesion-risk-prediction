import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# Configure fonts for publication-quality vector figures
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'

def get_roc_data(df, prob_col, target_col, mask_col):
    valid_df = df[df[mask_col] == 1]
    fpr, tpr, _ = roc_curve(valid_df[target_col], valid_df[prob_col])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def calculate_rr_from_df(df, prob_col, target_col, mask_col, percentages):
    valid_df = df[df[mask_col] == 1].copy()
    valid_df = valid_df.sort_values(by=prob_col, ascending=False)
    
    sorted_targets = valid_df[target_col].values
    total_valid_patients = len(sorted_targets)
    incidence = np.mean(sorted_targets)
    
    n_cutoffs = [max(1, int(p * total_valid_patients)) for p in percentages]
    
    rr_scores = []
    for n in n_cutoffs:
        precision = np.sum(sorted_targets[:n]) / n
        rr = precision / incidence
        rr_scores.append(rr)
        
    return rr_scores

# Load test-set predictions
BASE_DIR = Path(__file__).resolve().parent

df_main = pd.read_csv(BASE_DIR.parent / "test_data" / "Model types" / "Pretrain + Transformer" / "Test_set_result.csv")
df_cross = pd.read_csv(BASE_DIR.parent / "test_data" / "Model types" / "GRUD" / "Test_set_result.csv")
df_trans = pd.read_csv(BASE_DIR.parent / "test_data" / "Model types" / "Transformer" / "Test_set_result.csv")
df_xgb = pd.read_csv(BASE_DIR.parent / "test_data" / "Model types" / "XGBoost" / "Test_set_result.csv")
df_timelstm = pd.read_csv(BASE_DIR.parent / "test_data" / "Model types" / "Pretrain + TimeLSTM" / "Test_set_result.csv")

# Calculate AUROC and RR for the 18-month prediction horizon
fpr_main, tpr_main, auc_main = get_roc_data(df_main, 'prob_18m', 'target_18m', 'mask_18m')
fpr_cross, tpr_cross, auc_cross = get_roc_data(df_cross, 'prob_18m', 'target_18m', 'mask_18m')
fpr_trans, tpr_trans, auc_trans = get_roc_data(df_trans, 'prob_18m', 'target_18m', 'mask_18m')
fpr_xgb, tpr_xgb, auc_xgb = get_roc_data(df_xgb, 'prob_18m', 'target_18m', 'mask_18m')
fpr_timelstm, tpr_timelstm, auc_timelstm = get_roc_data(df_timelstm, 'prob_18m', 'target_18m', 'mask_18m')

percentages_smooth = np.logspace(np.log10(0.0008), np.log10(0.12), num=30)
x_plot_values = percentages_smooth * 100 

rr_main_smooth = calculate_rr_from_df(df_main, 'prob_18m', 'target_18m', 'mask_18m', percentages_smooth)
rr_cross_smooth = calculate_rr_from_df(df_cross, 'prob_18m', 'target_18m', 'mask_18m', percentages_smooth)
rr_trans_smooth = calculate_rr_from_df(df_trans, 'prob_18m', 'target_18m', 'mask_18m', percentages_smooth)
rr_xgb_smooth = calculate_rr_from_df(df_xgb, 'prob_18m', 'target_18m', 'mask_18m', percentages_smooth)
rr_timelstm_smooth = calculate_rr_from_df(df_timelstm, 'prob_18m', 'target_18m', 'mask_18m', percentages_smooth)

exact_percentages = [0.001, 0.01, 0.1]
exact_x = [0.1, 1, 10]

rr_main_exact = calculate_rr_from_df(df_main, 'prob_18m', 'target_18m', 'mask_18m', exact_percentages)
rr_cross_exact = calculate_rr_from_df(df_cross, 'prob_18m', 'target_18m', 'mask_18m', exact_percentages)
rr_trans_exact = calculate_rr_from_df(df_trans, 'prob_18m', 'target_18m', 'mask_18m', exact_percentages)
rr_xgb_exact = calculate_rr_from_df(df_xgb, 'prob_18m', 'target_18m', 'mask_18m', exact_percentages)
rr_timelstm_exact = calculate_rr_from_df(df_timelstm, 'prob_18m', 'target_18m', 'mask_18m', exact_percentages)

# Configure plot style and colors
sns.set_theme(style="ticks")
COLOR_MAIN = "#E64B35" 
COLOR_CROSS = "#8491B4"  
COLOR_TRANS = "#3C5488"  
COLOR_XGB = "#7F7F7F"    
COLOR_TIMELSTM = "#00A087" 

# Plot AUROC curves
plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random guess')

plt.plot(fpr_main, tpr_main, color=COLOR_MAIN, lw=2.5, label=f'Pretrain + Transformer\nAUROC {auc_main:.3f}')
plt.plot(fpr_timelstm, tpr_timelstm, color=COLOR_TIMELSTM, lw=2.5, label=f'Pretrain + TimeLSTM\nAUROC {auc_timelstm:.3f}')
plt.plot(fpr_cross, tpr_cross, color=COLOR_CROSS, lw=2.5, label=f'GRU-D:AUROC {auc_cross:.3f}')
plt.plot(fpr_trans, tpr_trans, color=COLOR_TRANS, lw=2.5, label=f'Transformer:AUROC {auc_trans:.3f}')
plt.plot(fpr_xgb, tpr_xgb, color=COLOR_XGB, lw=2.5, label=f'XGBoost:AUROC {auc_xgb:.3f}')

plt.xlabel('False-positive rate', fontsize=18)
plt.ylabel('True-positive rate', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.legend(loc='lower right', frameon=True, edgecolor='lightgrey', fontsize=14)

plt.tight_layout()

plt.savefig(r".\subfigure\Model_types_AUC.svg", format='svg')
plt.savefig(r".\subfigure\Model_types_AUC.pdf", format='pdf', dpi=300, bbox_inches="tight")
plt.close()

# Plot relative risk curves
plt.figure(figsize=(8, 5))

plt.plot(x_plot_values, rr_main_smooth, color=COLOR_MAIN, lw=2.5, label='Pretrain + Transformer')
plt.plot(x_plot_values, rr_timelstm_smooth, color=COLOR_TIMELSTM, lw=2.5, label='Pretrain + TimeLSTM')
plt.plot(x_plot_values, rr_cross_smooth, color=COLOR_CROSS, lw=2.5, label='GRU-D')
plt.plot(x_plot_values, rr_trans_smooth, color=COLOR_TRANS, lw=2.5, label='Transformer')
plt.plot(x_plot_values, rr_xgb_smooth, color=COLOR_XGB, lw=2.5, label='XGBoost')

plt.scatter(exact_x, rr_main_exact, color=COLOR_MAIN, s=10, zorder=5)
plt.scatter(exact_x, rr_timelstm_exact, color=COLOR_TIMELSTM, s=10, zorder=5)
plt.scatter(exact_x, rr_cross_exact, color=COLOR_CROSS, s=10, zorder=5)
plt.scatter(exact_x, rr_trans_exact, color=COLOR_TRANS, s=10, zorder=5)
plt.scatter(exact_x, rr_xgb_exact, color=COLOR_XGB, s=10, zorder=5)

plt.xscale('log')
plt.xticks(exact_x, ['0.1%', '1%', '10%'])
plt.axvline(x=0.1, color='grey', linestyle='--', alpha=0.7)
plt.axvline(x=1, color='grey', linestyle='--', alpha=0.3)
plt.axvline(x=10, color='grey', linestyle='--', alpha=0.3)

plt.xlabel('Top % of patients at risk', fontsize=18)
plt.ylabel('Relative risk', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
yticks = plt.gca().get_yticks()
valid_yticks = [y for y in yticks if y >= 0]
plt.gca().set_yticks(valid_yticks)
plt.gca().set_yticklabels([f'{int(y)}x' for y in valid_yticks])

plt.legend(loc='upper right', frameon=True, edgecolor='lightgrey', fontsize=12)

plt.tight_layout()

plt.savefig(r".\subfigure\Model_types_RR.svg", format='svg')
plt.savefig(r".\subfigure\Model_types_RR.pdf", format='pdf', dpi=300, bbox_inches="tight")
plt.close()