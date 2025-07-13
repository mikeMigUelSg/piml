"""
Train and compare physics-informed RNN (PI-RNN) and baseline RNN for:

  • Scenario 3: Maximum 10-step horizon forecasting  
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from data_utils import (
    PBM_SIM_PATHS,
    PBM_FEATURES,
    PBM_TARGET,
    BATTERY_FEATURES,
    BATTERY_TARGET,
    load_battery_data,
    make_sequences
)
from models import train_pbm_surrogate_for_PI_RNN, MultiStepPIRNN, BaselineMultiStepRNN

# —————————————————————————————
# 0. Styling & Reproducibility
# —————————————————————————————
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 18

seed = 40
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# —————————————————————————————
# 1. Train PBM surrogate (capacity-drop)
# —————————————————————————————
rf_model, scaler_sim = train_pbm_surrogate_for_PI_RNN(
    PBM_SIM_PATHS,
    PBM_FEATURES,
    PBM_TARGET,
    seed=seed
)

# —————————————————————————————
# 2. Load & preprocess battery data
# —————————————————————————————
X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler, test_df, tv = \
    load_battery_data(seed=seed)
features = BATTERY_FEATURES
target   = BATTERY_TARGET

print("Training: ", tv['Unique_Cell_ID'].unique())
print("Testing: ", test_df['Unique_Cell_ID'].unique())

# —————————————————————————————
# 3. Build sequences for Scenario 3
# —————————————————————————————
h3 = 10
X_tr3, y_tr3 = make_sequences(X_train_s, y_train, h3)
X_va3, y_va3 = make_sequences(X_val_s,   y_val,   h3)

def T(x): return torch.tensor(x, dtype=torch.float32)
X_tr3_t, y_tr3_t = T(X_tr3), T(y_tr3)
X_va3_t, y_va3_t = T(X_va3), T(y_va3)

# —————————————————————————————
# 4. Scenario-training helper
# —————————————————————————————
def train_scenario(model, optimizer, Xtr, ytr, Xva, yva, horizon, name):
    best_val, no_imp = float('inf'), 0
    criterion = nn.MSELoss()
    for ep in range(1, 2501):
        model.train(); optimizer.zero_grad()
        seed_cap = ytr[:, :1]
        preds    = model(Xtr, seed_cap, forecast_steps=horizon)
        loss     = criterion(preds, ytr[:, :horizon])
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            vpred = model(Xva, yva[:, :1], forecast_steps=horizon)
            vloss = criterion(vpred, yva[:, :horizon])

        if vloss < best_val:
            best_val, no_imp = vloss, 0
        else:
            no_imp += 1
            if no_imp >= 150:
                print(f"[{name}] early stop @ epoch {ep}")
                break

        if ep % 100 == 0:
            print(f"[{name}] Epoch {ep}: train={loss:.4f} val={vloss:.4f}")

# —————————————————————————————
# 5. Scenario 3: 10-step max-horizon
# —————————————————————————————
input_size  = len(features) + 1
hidden_size = 50

torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
pi3  = MultiStepPIRNN(input_size, hidden_size, rf_model)
opt3 = optim.Adam(pi3.parameters(), lr=1e-3)
train_scenario(pi3, opt3, X_tr3_t, y_tr3_t, X_va3_t, y_va3_t, h3, "PI-RNN S3")

torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
b3   = BaselineMultiStepRNN(input_size, hidden_size)
ob3  = optim.Adam(b3.parameters(), lr=1e-3)
train_scenario(b3,  ob3,  X_tr3_t, y_tr3_t, X_va3_t, y_va3_t, h3, "Base-RNN S3")

# —————————————————————————————
# 6. Save trained S3 models
# —————————————————————————————
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)
torch.save(pi3.state_dict(), f"{model_dir}/pi3_scenario3.pth")
torch.save(b3.state_dict(),  f"{model_dir}/b3_scenario3.pth")
print(f"Saved PI-RNN S3 and Baseline S3 to {model_dir}")

# —————————————————————————————
# 7. Visualization + fine-tuning on first 5 points
# —————————————————————————————
def visualize_scenario3(
    forecast_rpt, forecast_steps, scaler, features, target,
    Group='G3', Cell='C1', return_predictions=False,
    fine_tune=False, fine_tune_epochs=20
):
    # pick original models or clones
    if fine_tune:
        m3 = deepcopy(pi3)
        b3_ = deepcopy(b3)
    else:
        m3 = pi3
        b3_ = b3

    df = test_df[(test_df['Group']==Group)&(test_df['Cell']==Cell)].copy()
    rpts = df['RPT Number']
    min_rpt, max_rpt = rpts.min(), rpts.max()

    # find missing RPT numbers in the [min, max] range if any
    full_set = set(range(min_rpt, max_rpt + 1))
    missing = sorted(full_set - set(rpts))

    to_add = []
    for m in missing:
        prev_r, next_r = m - 1, m + 1
        if prev_r in rpts.values and next_r in rpts.values:
            v_prev = df.loc[df['RPT Number'] == prev_r, target].iloc[0]
            v_next = df.loc[df['RPT Number'] == next_r, target].iloc[0]
            to_add.append({
                'RPT Number': m,
                target:       (v_prev + v_next) / 2
            })

    if to_add:
        df = pd.concat([df, pd.DataFrame(to_add)], ignore_index=True)
        df = df.sort_values('RPT Number').reset_index(drop=True)

    avail = df[df['RPT Number'] <= forecast_rpt]
    fut   = df[df['RPT Number'] >  forecast_rpt]
    fw    = fut.iloc[:forecast_steps]
    n     = len(fw)

    # fine-tune on first 5 available points
    if fine_tune and len(avail) >= 5:
        raw_feats = avail[features].values[:5]
        raw_tgts  = avail[target].values[:5]
        scaled_5  = scaler.transform(raw_feats)

        Xm, ym = make_sequences(scaled_5, raw_tgts, 5)
        Xm_t, ym_t = torch.tensor(Xm, dtype=torch.float32), torch.tensor(ym, dtype=torch.float32)

        opts = [
            optim.Adam(m3.parameters(), lr=1e-3),
            optim.Adam(b3_.parameters(), lr=1e-3),
        ]
        loss_fn = nn.MSELoss()

        m3.train(); b3_.train()
        for _ in range(fine_tune_epochs):
            # S3: 5-step
            opts[0].zero_grad()
            p3 = m3(Xm_t, ym_t[:, :1], forecast_steps=5)
            l3 = loss_fn(p3, ym_t)
            l3.backward(); opts[0].step()

            opts[1].zero_grad()
            bp3 = b3_(Xm_t, ym_t[:, :1], forecast_steps=5)
            lb3 = loss_fn(bp3, ym_t)
            lb3.backward(); opts[1].step()

        m3.eval(); b3_.eval()

    # prepare future window
    Xf_t = torch.tensor(scaler.transform(fw[features].fillna(0)),
                       dtype=torch.float32).unsqueeze(0)
    seed_t = torch.tensor([[avail[target].iloc[-1]]],
                          dtype=torch.float32)

    with torch.no_grad():
        pi3_p = m3(Xf_t, seed_t, forecast_steps=n).squeeze(0).cpu().numpy()
        b3_p  = b3_(Xf_t, seed_t, forecast_steps=n).squeeze(0).cpu().numpy()

    if return_predictions:
        return {
            'available_idx': avail.index.values,
            'forecast_idx':  fw.index.values,
            'true_values':   fw[target].values,
            'S3': {'pi': pi3_p, 'baseline': b3_p},
        }

    # ————————————————— Plotting —————————————————
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    
    last_idx, last_val = avail.index[-1], avail[target].iloc[-1]
    first_idx, first_val = fw.index[0], fw[target].iloc[0]

    ax.axvline(x=forecast_rpt-1, color='black', linestyle='--', linewidth=1)
    ax.plot(avail.index, avail[target], 'ko-', markersize=9, linewidth=1.5, label='Data Available')
    ax.plot([last_idx, first_idx], [last_val, first_val], 'k-', linewidth=1.5)
    ax.plot(fut.index, fut[target], 'ko-', markersize=9, linewidth=1.5,
            markerfacecolor='white', label='True Capacity')
    ax.plot(fw.index, pi3_p, marker='d', color='crimson',
            markersize=6, linestyle='-', linewidth=0.5, label='PI-RNN')
    ax.plot(fw.index, b3_p, marker='s', color='crimson',
            markersize=6, linestyle='--', linewidth=0.5, label='Baseline RNN')
    
    ax.set_xlabel('RPT Number (-)', fontsize=18)
    ax.set_ylabel('Capacity (Ah)', fontsize=18)
    ax.set_xticks(np.arange(0, 35, 5))
    ax.set_yticks(np.arange(0.4, 1.6, 0.2))
    ax.set_xlim(-2, 35)
    ax.set_ylim(0.4, 1.4)
    ax.set_title('S3: Maximum-Horizon Forecasting', fontsize=16)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

# —————————————————————————————
# If run as a script, launch viz
# —————————————————————————————
if __name__ == '__main__':
    visualize_scenario3(
        forecast_rpt      = 10,
        forecast_steps    = 7,
        scaler            = scaler,
        features          = features,
        target            = target,
        Group             = 'G3',
        Cell              = 'C1',
        return_predictions= False,
        fine_tune         = False
    )