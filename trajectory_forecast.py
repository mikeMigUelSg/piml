"""
Generate and visualize capacity forecast trajectories for a chosen battery cell,
using pre-trained PI-RNN, baseline RNN, and GPR models.

- Produce a 1×3 grid of larger plots showing true vs. predicted trajectories in three "life phases"
9. Supports CLI flags:
   --group, --cell        : choose which cell to forecast  
   --fine-tune            : enable short fine-tuning  
   --epochs               : number of fine-tune epochs  
   --return-predictions   : output raw arrays instead of plotting  

Usage:
    python trajectory_forecast.py --group G13 --cell C1 [--fine-tune] [--epochs 20]
"""


import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings
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
from models import train_pbm_surrogate_for_PI_RNN, MultiStepPIRNN, BaselineMultiStepRNN, GPRBaseline

warnings.filterwarnings('ignore')
# —————————————————————————————
# 0. Styling & reproducibility
# —————————————————————————————
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 20

seed = 40
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# —————————————————————————————
# 1. Train PBM surrogate
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
X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler, test_df = \
    load_battery_data(seed=seed)

# Scenario‐3 constants
h3 = 20
input_size  = len(BATTERY_FEATURES) + 1
hidden_size = 50

# —————————————————————————————
# 3. Load pretrained Scenario‐3 models
# —————————————————————————————
pi3 = MultiStepPIRNN(input_size, hidden_size, rf_model)
pi3.load_state_dict(torch.load('saved_models/pi3_scenario3.pth'))
pi3.eval()

b3 = BaselineMultiStepRNN(input_size, hidden_size)
b3.load_state_dict(torch.load('saved_models/b3_scenario3.pth'))
b3.eval()

def trajectory_forecast(
    group: str,
    cell:  str,
    fine_tune: bool = False,
    fine_tune_epochs: int = 20,
    return_predictions: bool = False
):
    # select cell data
    df = (
        test_df
        .loc[lambda d: (d['Group']==group) & (d['Cell']==cell)]
        .sort_values('RPT Number')
        .reset_index(drop=True)
    )
    df[BATTERY_FEATURES] = df[BATTERY_FEATURES].fillna(0)

    # --- fill any single‐gap RPT numbers by neighbor‐averaging ---
    rpts = df['RPT Number']
    min_rpt, max_rpt = rpts.min(), rpts.max()

    full_set = set(range(min_rpt, max_rpt + 1))
    missing_rpts = sorted(full_set - set(rpts))

    # for each missing rpt, if both neighbors exist, interpolate
    to_add = []
    for m in missing_rpts:
        prev_r, next_r = m - 1, m + 1
        if prev_r in rpts.values and next_r in rpts.values:
            v_prev = df.loc[df['RPT Number'] == prev_r, BATTERY_TARGET].iloc[0]
            v_next = df.loc[df['RPT Number'] == next_r, BATTERY_TARGET].iloc[0]
            to_add.append({
                'RPT Number': m,
                BATTERY_TARGET: (v_prev + v_next) / 2
            })

    # append & re‐sort only if there's something to add
    if to_add:
        df = pd.concat([df, pd.DataFrame(to_add)], ignore_index=True)
        df = df.sort_values('RPT Number').reset_index(drop=True)

    # available and future splits for scenario 3 forecasting
    avail = df[df['RPT Number'] <= h3]
    fut   = df[df['RPT Number'] >  h3]

    total_future = len(fut)
    if total_future == 0:
        raise ValueError(
            f"{group}{cell} só tem {df['RPT Number'].max()} RPTs; "
            f"impossível prever {h3} passos."
        )

    # usar o que houver, mas nunca mais do que h3
    fw = fut.iloc[:min(h3, total_future)]
    n  = len(fw)

    print("----: ", n, total_future)


    model = pi3
    baseline = b3 

    # —————————————————————————————
    # 4. Precompute GPR trajectory
    # —————————————————————————————
    gpr = GPRBaseline(initial_points=10)
    gpr.fit(df, (group, cell), initial_points=10)

    hybrid = []
    for origin in range(len(df)):
        sub_df = df.iloc[origin:].reset_index(drop=True)
        _, yp = gpr.predict(sub_df, (group, cell), initial_points=1)

        # Handle empty prediction case
        if yp.size == 0:           # no more data to predict
            hybrid.append(np.nan)  # or simply break
            continue
        else:
            hybrid.append(np.nan)
        hybrid.append(yp[0])
        
    y_pred_gpr = np.array(hybrid)

    # —————————————————————————————
    # prepare future-window tensor for RNNs
    # —————————————————————————————

    Xf = scaler.transform(fw[BATTERY_FEATURES].values)
    Xt = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0)
    St = torch.tensor([[avail[BATTERY_TARGET].iloc[-1]]], dtype=torch.float32)

    with torch.no_grad():
        bb = baseline(Xt, St, forecast_steps=n).cpu().numpy().squeeze(0)
        pp = model   (Xt, St, forecast_steps=n).cpu().numpy().squeeze(0)

    # return raw arrays if requested
    if return_predictions:
        return {
            'available_idx': avail.index.values,
            'forecast_idx':  fw.index.values,
            'true_values':   fw[BATTERY_TARGET].values,
            'gpr':            y_pred_gpr[h3:h3+n],
            'baseline_rnn':   bb,
            'pi_rnn':         pp
        }
    

    # —————————————————————————————
    # 5. Final Plotting 
    # —————————————————————————————
    forecast_rpts     = [1, 7, 23]
    forecast_horizons = [7, 13, 10]
    fixed_horizon     = 5
    red_contrast      = '#D62728'
    light_red         = '#F5B7B1'
    hatches           = ['///','\\\\','...']
    phases            = ["(First-Life)","(Transition Phase)","(Second-Life)"]

    # Create a single row of 3 larger plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), dpi=100)
    
    # --- CAPACITY PLOTS ---
    for i, (rpt, horizon, phase) in enumerate(zip(forecast_rpts, forecast_horizons, phases)):
        ava  = df[df['RPT Number'] <= rpt]
        fut_i= df[df['RPT Number'] >  rpt]
        fw_i = fut_i.iloc[:horizon]
        n_i  = len(fw_i)

        ax = axes[i]
        ax.axvline(x=rpt-1, color='black', linestyle='--', linewidth=1)

        # available
        ax.plot(ava.index, ava[BATTERY_TARGET],
                'ko-', markersize=12, linewidth=1.5, label='Data Available')
        # connector
        if not ava.empty and not fut_i.empty:
            idx0, idx1 = ava.index[-1], fut_i.index[0]
            v0, v1     = ava[BATTERY_TARGET].iloc[-1], fut_i[BATTERY_TARGET].iloc[0]
            ax.plot([idx0, idx1], [v0, v1], 'k-', linewidth=1)

        # true future
        ax.plot(fut_i.index, fut_i[BATTERY_TARGET],
                'ko-', markersize=12, linewidth=1.5,
                markerfacecolor='white', label='True Capacity')

        # GPR
        gpr_fc = y_pred_gpr[rpt:rpt+n_i]
        ax.plot(fw_i.index, gpr_fc,
                marker='^', color='crimson', markersize=8,
                linestyle='-.', linewidth=0.5, label='Baseline GPR')

        # RNNs
        raw = fw_i[BATTERY_FEATURES].values
        Xf_i  = scaler.transform(raw)
        Xt_i  = torch.tensor(Xf_i, dtype=torch.float32).unsqueeze(0)
        St_i  = torch.tensor([[ava[BATTERY_TARGET].iloc[-1]]], dtype=torch.float32)
        with torch.no_grad():
            bb_i = baseline(Xt_i, St_i, forecast_steps=n_i).cpu().numpy().squeeze(0)
            pp_i = model   (Xt_i, St_i, forecast_steps=n_i).cpu().numpy().squeeze(0)

        ax.plot(fw_i.index, bb_i,
                's', color='crimson', markersize=8,
                linestyle='--', linewidth=0.5, label='Baseline RNN')
        ax.plot(fw_i.index, pp_i,
                'd', color='crimson', markersize=8,
                linestyle='-', linewidth=0.5, label='PI-RNN')

        ax.set_title(f"{group}{cell} {phase}", fontsize=24)
        ax.set_xlabel('RPT Number (-)', fontsize=20)
        ax.set_xticks(np.arange(0, 35, 5))
        ax.set_ylabel('Capacity (Ah)', fontsize=20)
        ax.set_yticks(np.arange(0.4, 1.6, 0.2))
        ax.set_xlim(-2, 35)
        ax.set_ylim(0.4, 1.4)
        ax.tick_params(labelsize=20)
        ax.legend(loc='upper right', fontsize=14)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='G3')
    parser.add_argument('--cell',  type=str, default='C1')
    parser.add_argument('--fine-tune', action='store_true')
    parser.add_argument('--epochs',    type=int,   default=20)
    parser.add_argument('--return-predictions', action='store_true')
    args = parser.parse_args()

    out = trajectory_forecast(
        args.group,
        args.cell,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.epochs,
        return_predictions=args.return_predictions
    )
    if args.return_predictions:
        print(out)