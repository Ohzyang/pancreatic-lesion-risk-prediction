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

def calculate_rr_from_df(df, prob_col, target_col, mask_col, top_pcts):
    valid_df = df[df[mask_col] == 1].copy()
    valid_df = valid_df.sort_values(by=prob_col, ascending=False)
    
    sorted_targets = valid_df[target_col].values
    total_valid_patients = len(sorted_targets)
    incidence = np.mean(sorted_targets)
    
    rr_scores = []
    for pct in top_pcts:
        n = int(total_valid_patients * pct)
        n = max(1, min(n, total_valid_patients)) 
        precision = np.sum(sorted_targets[:n]) / n
        rr = precision / incidence

        if rr == 0:
            rr = np.nan

        rr_scores.append(rr)
        
    return rr_scores

# Load test-set predictions
BASE_DIR = Path(__file__).resolve().parent

df_complete = pd.read_csv(BASE_DIR.parent / "test_data" / "Missingness tolerance" / "Missing rate 0%" / "Test_set_result.csv")
df_cross = pd.read_csv(BASE_DIR.parent / "test_data" / "Missingness tolerance" / "Missing rate ≤10%" / "Test_set_result.csv")
df_main = pd.read_csv(BASE_DIR.parent / "test_data" / "Missingness tolerance" / "Missing rate ≤20%" / "Test_set_result.csv")
df_doctor = pd.read_csv(BASE_DIR.parent / "test_data" / "Missingness tolerance" / "Missing rate ≤30%" / "Test_set_result.csv")

# Calculate AUROC and RR for the 18-month prediction horizon
fpr_complete, tpr_complete, auc_complete = get_roc_data(df_complete, 'prob_18m', 'target_18m', 'mask_18m')
fpr_cross, tpr_cross, auc_cross = get_roc_data(df_cross, 'prob_18m', 'target_18m', 'mask_18m')
fpr_main, tpr_main, auc_main = get_roc_data(df_main, 'prob_18m', 'target_18m', 'mask_18m')
fpr_doctor, tpr_doctor, auc_doctor = get_roc_data(df_doctor, 'prob_18m', 'target_18m', 'mask_18m')

pct_smooth = np.logspace(-3, -1, num=30)
exact_pct = [0.001, 0.01, 0.1]  
pct_smooth = np.unique(np.concatenate([pct_smooth, exact_pct]))
x_plot_values = pct_smooth

rr_complete_smooth = calculate_rr_from_df(df_complete, 'prob_18m', 'target_18m', 'mask_18m', pct_smooth)
rr_cross_smooth = calculate_rr_from_df(df_cross, 'prob_18m', 'target_18m', 'mask_18m', pct_smooth)
rr_main_smooth = calculate_rr_from_df(df_main, 'prob_18m', 'target_18m', 'mask_18m', pct_smooth)
rr_doctor_smooth = calculate_rr_from_df(df_doctor, 'prob_18m', 'target_18m', 'mask_18m', pct_smooth)

rr_complete_exact = calculate_rr_from_df(df_complete, 'prob_18m', 'target_18m', 'mask_18m', exact_pct)
rr_cross_exact = calculate_rr_from_df(df_cross, 'prob_18m', 'target_18m', 'mask_18m', exact_pct)
rr_main_exact = calculate_rr_from_df(df_main, 'prob_18m', 'target_18m', 'mask_18m', exact_pct)
rr_doctor_exact = calculate_rr_from_df(df_doctor, 'prob_18m', 'target_18m', 'mask_18m', exact_pct)

# Configure plot style and colors
sns.set_theme(style="ticks")
COLOR_COMPLETE = "#00A087" 
COLOR_CROSS = "#3C5488"    
COLOR_MAIN = "#E64B35"      
COLOR_DOCTOR = "#9ECAE1"    

# Plot AUROC curves
plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random guess')
plt.plot(fpr_complete, tpr_complete, color=COLOR_COMPLETE, lw=2.5, label=f'Missing rate 0%\nAUROC {auc_complete:.3f}')
plt.plot(fpr_cross, tpr_cross, color=COLOR_CROSS, lw=2.5, label=f'Missing rate ≤10%\nAUROC {auc_cross:.3f}')
plt.plot(fpr_main, tpr_main, color=COLOR_MAIN, lw=2.5, label=f'Missing rate ≤20%\nAUROC {auc_main:.3f}')
plt.plot(fpr_doctor, tpr_doctor, color=COLOR_DOCTOR, lw=2.5, label=f'Missing rate ≤30%\nAUROC {auc_doctor:.3f}')

plt.xlabel('False-positive rate', fontsize=18)
plt.ylabel('True-positive rate', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.legend(loc='lower right', frameon=True, edgecolor='lightgrey', fontsize=14)

plt.tight_layout()
plt.savefig(r".\subfigure\Missing_tolerance_AUC.pdf", format="pdf", bbox_inches="tight")
plt.savefig(r".\subfigure\Missing_tolerance_AUC.svg", format="svg")
plt.close()

# Plot relative risk curves
plt.figure(figsize=(8, 5))
plt.plot(x_plot_values, rr_complete_smooth, color=COLOR_COMPLETE, lw=2.5, label='Missing rate 0%')
plt.plot(x_plot_values, rr_cross_smooth, color=COLOR_CROSS, lw=2.5, label='Missing rate ≤10%')
plt.plot(x_plot_values, rr_main_smooth, color=COLOR_MAIN, lw=2.5, label='Missing rate ≤20%')
plt.plot(x_plot_values, rr_doctor_smooth, color=COLOR_DOCTOR, lw=2.5, label='Missing rate ≤30%')

plt.scatter(exact_pct, rr_complete_exact, color=COLOR_COMPLETE, s=10, zorder=5)
plt.scatter(exact_pct, rr_cross_exact, color=COLOR_CROSS, s=10, zorder=5)
plt.scatter(exact_pct, rr_main_exact, color=COLOR_MAIN, s=10, zorder=5)
plt.scatter(exact_pct, rr_doctor_exact, color=COLOR_DOCTOR, s=10, zorder=5)

plt.xscale('log')
plt.xticks(exact_pct, ['0.1%', '1%', '10%'])
plt.axvline(x=0.001, color='grey', linestyle='--', alpha=0.7)
plt.axvline(x=0.01, color='grey', linestyle='--', alpha=0.3)
plt.axvline(x=0.1, color='grey', linestyle='--', alpha=0.3)

plt.xlabel('Top % patients at risk', fontsize=18)
plt.ylabel('Relative risk', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)

yticks = plt.gca().get_yticks()
valid_yticks = [y for y in yticks if y >= 0]
plt.gca().set_yticks(valid_yticks)
plt.gca().set_yticklabels([f'{int(y)}x' for y in valid_yticks])

plt.legend(loc='upper right', frameon=True, edgecolor='lightgrey', fontsize=14)

plt.tight_layout()

plt.savefig(r".\subfigure\Missing_tolerance_RR.pdf", format="pdf", bbox_inches="tight")
plt.savefig(r".\subfigure\Missing_tolerance_RR.svg", format="svg")
plt.close()