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

# Load test-set predictions
BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR.parent / "test_data" / "Prediction horizons" / "Pretrain + Transformer" / "Test_set_result.csv")

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

# Calculate AUROC for each prediction horizon
fpr_12, tpr_12, auc_12 = get_roc_data(df, 'prob_12m', 'target_12m', 'mask_12m')
fpr_18, tpr_18, auc_18 = get_roc_data(df, 'prob_18m', 'target_18m', 'mask_18m')
fpr_24, tpr_24, auc_24 = get_roc_data(df, 'prob_24m', 'target_24m', 'mask_24m')

# Calculate RR across logarithmically spaced top-risk percentages
percentages_smooth = np.logspace(np.log10(0.0008), np.log10(0.12), num=30)
x_plot_values = percentages_smooth * 100 

rr_12_smooth = calculate_rr_from_df(df, 'prob_12m', 'target_12m', 'mask_12m', percentages_smooth)
rr_18_smooth = calculate_rr_from_df(df, 'prob_18m', 'target_18m', 'mask_18m', percentages_smooth)
rr_24_smooth = calculate_rr_from_df(df, 'prob_24m', 'target_24m', 'mask_24m', percentages_smooth)

exact_percentages = [0.001, 0.01, 0.1]
exact_x = [0.1, 1, 10]

# Configure plot style and colors
COLOR_MAIN = "#E64B35"   
COLOR_12M = "#8491B4"   
COLOR_24M = "#3C5488"    

sns.set_theme(style="ticks")

# Plot AUROC curves
fig_auc, ax_auc = plt.subplots(figsize=(5, 5))
ax_auc.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random guess')
ax_auc.plot(fpr_12, tpr_12, color=COLOR_12M, lw=2, label=f'Prediction horizon 12 months\nAUROC {auc_12:.3f}')
ax_auc.plot(fpr_18, tpr_18, color=COLOR_MAIN, lw=2.5, label=f'Prediction horizon 18 months\nAUROC {auc_18:.3f}')
ax_auc.plot(fpr_24, tpr_24, color=COLOR_24M, lw=2, label=f'Prediction horizon 24 months\nAUROC {auc_24:.3f}')
ax_auc.set_xlabel('False-positive rate', fontsize=18)
ax_auc.set_ylabel('True-positive rate', fontsize=18)
ax_auc.tick_params(
    axis='both',
    which='major',
    labelsize=16
)
ax_auc.set_xlim([-0.02, 1.02])
ax_auc.set_ylim([-0.02, 1.02])

ax_auc.legend(loc='lower right', frameon=True, edgecolor='lightgrey', fontsize=14)

fig_auc.tight_layout()
fig_auc.savefig(r".\subfigure\Prediction_horizons_AUC.pdf", format="pdf", bbox_inches="tight")
fig_auc.savefig(r".\subfigure\Prediction_horizons_AUC.svg", format="svg")

# Plot relative risk curves
fig_rr, ax_rr = plt.subplots(figsize=(8, 5))
ax_rr.plot(x_plot_values, rr_12_smooth, color=COLOR_12M, lw=2, label='Prediction horizon 12 months')
ax_rr.plot(x_plot_values, rr_18_smooth, color=COLOR_MAIN, lw=2.5, label='Prediction horizon 18 months')
ax_rr.plot(x_plot_values, rr_24_smooth, color=COLOR_24M, lw=2, label='Prediction horizon 24 months')

ax_rr.scatter(exact_x, calculate_rr_from_df(df, 'prob_12m', 'target_12m', 'mask_12m', exact_percentages), color=COLOR_12M, s=10, zorder=5)
ax_rr.scatter(exact_x, calculate_rr_from_df(df, 'prob_18m', 'target_18m', 'mask_18m', exact_percentages), color=COLOR_MAIN, s=10, zorder=5)
ax_rr.scatter(exact_x, calculate_rr_from_df(df, 'prob_24m', 'target_24m', 'mask_24m', exact_percentages), color=COLOR_24M, s=10, zorder=5)

ax_rr.set_xscale('log')
ax_rr.set_xticks(exact_x)
ax_rr.set_xticklabels(['0.1%', '1%', '10%'])
ax_rr.axvline(x=0.1, color='grey', linestyle='--', alpha=0.7)
ax_rr.axvline(x=1, color='grey', linestyle='--', alpha=0.3)
ax_rr.axvline(x=10, color='grey', linestyle='--', alpha=0.3)

ax_rr.set_xlabel('Top % of patients at risk', fontsize=18)
ax_rr.set_ylabel('Relative risk', fontsize=18)
ax_rr.tick_params(
    axis='both',
    which='major',
    labelsize=16
)
yticks = ax_rr.get_yticks()
valid_yticks = [y for y in yticks if y >= 0]
ax_rr.set_yticks(valid_yticks)
ax_rr.set_yticklabels([f'{int(y)}x' for y in valid_yticks])
ax_rr.legend(loc='upper right', frameon=True, edgecolor='lightgrey', fontsize=14)

fig_rr.tight_layout()
fig_rr.savefig(r".\subfigure\Prediction_horizons_RR.pdf", format="pdf", bbox_inches="tight")
fig_rr.savefig(r".\subfigure\Prediction_horizons_RR.svg", format="svg")