import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

OUTPUT_DIR = "catan_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── State indexing ─────────────────────────────────────────────────────────────
NUM_P, NUM_E, NUM_R = 3, 3, 18
NUM_TRANSIENT = NUM_P * NUM_E * NUM_R   # 162
WIN  = NUM_TRANSIENT       # 162
LOSS = NUM_TRANSIENT + 1   # 163
TOTAL = NUM_TRANSIENT + 2  # 164

def state_index(p, e, r):
    return (p * 3 + e) * 18 + r

def index_to_state(idx):
    r = idx % 18
    pe = idx // 18
    e = pe % 3
    p = pe // 3
    return p, e, r

# ── Dice / robber constants ────────────────────────────────────────────────────
P_SEVEN     = 6 / 36
P_NOT_SEVEN = 1 - P_SEVEN

# ── Core transition probability tables ────────────────────────────────────────
# vp_advance[p][e] = probability of advancing VP phase in one turn (given no absorption)
vp_advance = {
    0: {0: 0.06, 1: 0.12, 2: 0.20},
    1: {0: 0.08, 1: 0.16, 2: 0.28},
    2: {0: 0.00, 1: 0.00, 2: 0.00},   # replaced by win/loss logic
}

# engine_up[p][e] = probability engine improves by 1 in one turn
engine_up = {
    0: {0: 0.15, 1: 0.10, 2: 0.00},
    1: {0: 0.12, 1: 0.08, 2: 0.00},
    2: {0: 0.10, 1: 0.06, 2: 0.00},
}

# engine_down[p][e] = probability engine degrades by 1 in one turn
engine_down = {
    0: {0: 0.00, 1: 0.04, 2: 0.08},
    1: {0: 0.00, 1: 0.06, 2: 0.10},
    2: {0: 0.00, 1: 0.08, 2: 0.12},
}

# win_prob[e] = base win probability per turn when in late phase (P=2)
win_prob_late = {0: 0.04, 1: 0.10, 2: 0.20}

# loss_prob[p][e] = base loss probability per turn (opponent wins first)
loss_prob = {
    0: {0: 0.02, 1: 0.01, 2: 0.005},
    1: {0: 0.04, 1: 0.02, 2: 0.01},
    2: {0: 0.08, 1: 0.04, 2: 0.02},
}

# Robber penalty: tiles 0–5 are "high-value" hex positions
ROBBER_PENALTY_TILES = set(range(6))
ROBBER_PENALTY_FACTOR = 0.85   # multiplies VP-advance and win prob when robber is active there

# ── Build full transition matrix ───────────────────────────────────────────────
T = np.zeros((TOTAL, TOTAL))

for idx in range(NUM_TRANSIENT):
    p, e, r = index_to_state(idx)

    robber_penalty = ROBBER_PENALTY_FACTOR if r in ROBBER_PENALTY_TILES else 1.0

    base_loss = loss_prob[p][e]
    base_win  = win_prob_late[e] * robber_penalty if p == 2 else 0.0

    p_absorb_win  = base_win
    p_absorb_loss = base_loss
    p_absorb_win  = min(p_absorb_win,  1.0)
    p_absorb_loss = min(p_absorb_loss, 1.0 - p_absorb_win)

    p_remain = 1.0 - p_absorb_win - p_absorb_loss

    T[idx, WIN]  = p_absorb_win
    T[idx, LOSS] = p_absorb_loss

    p_vp_up   = (vp_advance[p][e] * robber_penalty) if p < 2 else 0.0
    p_vp_stay = 1.0 - p_vp_up

    p_e_up   = engine_up[p][e]
    p_e_down = engine_down[p][e]
    p_e_stay = 1.0 - p_e_up - p_e_down

    for vp_delta, p_vp in [(0, p_vp_stay), (1, p_vp_up)]:
        new_p = min(p + vp_delta, 2)
        if new_p == 2 and p < 2:
            new_p = 2

        for e_delta, p_e in [(-1, p_e_down), (0, p_e_stay), (1, p_e_up)]:
            new_e = e + e_delta
            if new_e < 0 or new_e > 2:
                continue
            prob_pe = p_vp * p_e

            for new_r in range(NUM_R):
                if new_r == r:
                    p_robber = P_NOT_SEVEN + P_SEVEN * (1.0 / NUM_R)
                else:
                    p_robber = P_SEVEN * (1.0 / NUM_R)

                new_idx = state_index(new_p, new_e, new_r)
                T[idx, new_idx] += p_remain * prob_pe * p_robber

# Absorbing states
T[WIN,  WIN]  = 1.0
T[LOSS, LOSS] = 1.0

# ── Normalise rows to correct floating-point drift ─────────────────────────────
for i in range(NUM_TRANSIENT):
    row_sum = T[i].sum()
    if row_sum > 0:
        T[i] /= row_sum

# ── Verify row sums ────────────────────────────────────────────────────────────
row_sums = T.sum(axis=1)
print("Row sum verification:")
print(f"  Max deviation from 1.0: {np.max(np.abs(row_sums - 1.0)):.2e}")
print(f"  All rows sum to ~1:     {np.allclose(row_sums, 1.0, atol=1e-9)}\n")

# ── Canonical form ─────────────────────────────────────────────────────────────
# Transient states: 0..161,  Absorbing: 162 (Win), 163 (Loss)
Q = T[:NUM_TRANSIENT, :NUM_TRANSIENT]
R = T[:NUM_TRANSIENT, NUM_TRANSIENT:]

# ── Fundamental matrix N = (I - Q)^{-1} ───────────────────────────────────────
I_q = np.eye(NUM_TRANSIENT)
N = np.linalg.inv(I_q - Q)

# ── Absorption probabilities B = N @ R ─────────────────────────────────────────
B = N @ R   # shape (162, 2); col 0 = P(Win), col 1 = P(Loss)

# ── Expected turns to absorption ──────────────────────────────────────────────
ones = np.ones(NUM_TRANSIENT)
expected_turns = N @ ones

# ── Results DataFrame ──────────────────────────────────────────────────────────
records = []
for idx in range(NUM_TRANSIENT):
    p, e, r = index_to_state(idx)
    records.append({
        "state_idx": idx,
        "phase": p,
        "engine": e,
        "robber_tile": r,
        "p_win": B[idx, 0],
        "p_loss": B[idx, 1],
        "exp_turns": expected_turns[idx],
    })

df = pd.DataFrame(records)

# ── Numerical outputs ──────────────────────────────────────────────────────────
print("=" * 60)
print("SELECTED STARTING STATE RESULTS")
print("=" * 60)
selected = [
    (0, 0, 0, "Early / Weak engine   / robber on good tile"),
    (0, 2, 0, "Early / Strong engine / robber on good tile"),
    (0, 2, 9, "Early / Strong engine / robber elsewhere   "),
    (1, 1, 4, "Mid   / Moderate      / robber on good tile"),
    (1, 1, 9, "Mid   / Moderate      / robber elsewhere   "),
    (2, 0, 0, "Late  / Weak engine   / robber on good tile"),
    (2, 1, 9, "Late  / Moderate      / robber elsewhere   "),
    (2, 2, 9, "Late  / Strong engine / robber elsewhere   "),
]
print(f"{'Description':<48} {'P(Win)':>8} {'P(Loss)':>8} {'E[Turns]':>10}")
print("-" * 76)
for p, e, r, desc in selected:
    idx = state_index(p, e, r)
    print(f"{desc:<48} {B[idx,0]:>8.4f} {B[idx,1]:>8.4f} {expected_turns[idx]:>10.2f}")

# ── Summary table by phase and engine ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY: AVERAGE BY PHASE AND ENGINE LEVEL")
print("=" * 60)
summary = df.groupby(["phase", "engine"])[["p_win", "p_loss", "exp_turns"]].mean().round(4)
print(summary.to_string())
summary.to_csv(os.path.join(OUTPUT_DIR, "summary_phase_engine.csv"))

# ── Summary by engine level ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY: AVERAGE BY ENGINE LEVEL")
print("=" * 60)
by_engine = df.groupby("engine")[["p_win", "p_loss", "exp_turns"]].mean().round(4)
by_engine.index = ["Weak (E=0)", "Moderate (E=1)", "Strong (E=2)"]
print(by_engine.to_string())
by_engine.to_csv(os.path.join(OUTPUT_DIR, "summary_engine.csv"))

# ── Robber effect ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY: AVERAGE BY ROBBER TILE (all phases/engines)")
print("=" * 60)
by_robber = df.groupby("robber_tile")[["p_win", "exp_turns"]].mean().round(4)
print(by_robber.to_string())
by_robber.to_csv(os.path.join(OUTPUT_DIR, "summary_robber.csv"))

# ── GRAPHS ─────────────────────────────────────────────────────────────────────
engine_labels = ["Weak\n(E=0)", "Moderate\n(E=1)", "Strong\n(E=2)"]
phase_labels  = ["Early (P=0)", "Mid (P=1)", "Late (P=2)"]

# 1. Bar chart: average win probability by engine level
fig, ax = plt.subplots(figsize=(6, 4))
vals = df.groupby("engine")["p_win"].mean().values
colors = ["#d95f02", "#7570b3", "#1b9e77"]
bars = ax.bar(engine_labels, vals, color=colors, edgecolor="black", linewidth=0.7)
ax.set_ylabel("Average P(Win)")
ax.set_title("Average Win Probability by Engine Level")
ax.set_ylim(0, 1)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_win_prob_by_engine.png"), dpi=150)
plt.close()

# 2. Bar chart: average loss probability by engine level
fig, ax = plt.subplots(figsize=(6, 4))
vals_l = df.groupby("engine")["p_loss"].mean().values
bars = ax.bar(engine_labels, vals_l, color=colors, edgecolor="black", linewidth=0.7)
ax.set_ylabel("Average P(Loss)")
ax.set_title("Average Loss Probability by Engine Level")
ax.set_ylim(0, 1)
for bar, v in zip(bars, vals_l):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_loss_prob_by_engine.png"), dpi=150)
plt.close()

# 3. Heatmap: win probability by phase and engine
pivot_win = df.groupby(["phase", "engine"])["p_win"].mean().unstack()
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(pivot_win.values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(NUM_E)); ax.set_xticklabels(["Weak", "Moderate", "Strong"])
ax.set_yticks(range(NUM_P)); ax.set_yticklabels(phase_labels)
ax.set_xlabel("Engine Level"); ax.set_ylabel("VP Phase")
ax.set_title("Avg Win Probability (Phase × Engine)")
plt.colorbar(im, ax=ax, label="P(Win)")
for i in range(NUM_P):
    for j in range(NUM_E):
        ax.text(j, i, f"{pivot_win.values[i,j]:.3f}", ha="center", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_heatmap_win_phase_engine.png"), dpi=150)
plt.close()

# 4. Heatmap: expected turns by phase and engine
pivot_exp = df.groupby(["phase", "engine"])["exp_turns"].mean().unstack()
fig, ax = plt.subplots(figsize=(5, 4))
im2 = ax.imshow(pivot_exp.values, cmap="coolwarm_r", aspect="auto")
ax.set_xticks(range(NUM_E)); ax.set_xticklabels(["Weak", "Moderate", "Strong"])
ax.set_yticks(range(NUM_P)); ax.set_yticklabels(phase_labels)
ax.set_xlabel("Engine Level"); ax.set_ylabel("VP Phase")
ax.set_title("Avg Expected Turns to Absorption (Phase × Engine)")
plt.colorbar(im2, ax=ax, label="E[Turns]")
for i in range(NUM_P):
    for j in range(NUM_E):
        ax.text(j, i, f"{pivot_exp.values[i,j]:.1f}", ha="center", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_heatmap_turns_phase_engine.png"), dpi=150)
plt.close()

# 5. Line plot: average win probability by robber tile
by_robber_win = df.groupby("robber_tile")["p_win"].mean()
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(by_robber_win.index, by_robber_win.values, marker="o", color="#1f78b4", linewidth=1.5)
ax.axvspan(-0.5, 5.5, alpha=0.12, color="red", label="High-value tiles (penalty)")
ax.set_xlabel("Robber Tile Index")
ax.set_ylabel("Average P(Win)")
ax.set_title("Average Win Probability by Robber Tile Position")
ax.set_xticks(range(NUM_R))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_win_prob_by_robber.png"), dpi=150)
plt.close()

# 6. Line plot: expected turns by robber tile
by_robber_turns = df.groupby("robber_tile")["exp_turns"].mean()
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(by_robber_turns.index, by_robber_turns.values, marker="s", color="#e31a1c", linewidth=1.5)
ax.axvspan(-0.5, 5.5, alpha=0.12, color="red", label="High-value tiles (penalty)")
ax.set_xlabel("Robber Tile Index")
ax.set_ylabel("Average E[Turns]")
ax.set_title("Average Expected Turns by Robber Tile Position")
ax.set_xticks(range(NUM_R))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_exp_turns_by_robber.png"), dpi=150)
plt.close()

# 7. Grouped bar: P(Win) for each engine, split by phase
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(NUM_P)
width = 0.25
for e_idx, (color, label) in enumerate(zip(colors, ["Weak", "Moderate", "Strong"])):
    vals_e = [df[(df.phase == p) & (df.engine == e_idx)]["p_win"].mean() for p in range(NUM_P)]
    ax.bar(x + (e_idx - 1) * width, vals_e, width, label=label, color=color, edgecolor="black", linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(phase_labels)
ax.set_ylabel("Average P(Win)")
ax.set_title("Win Probability by Phase and Engine Level")
ax.legend(title="Engine")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_grouped_bar_win_phase_engine.png"), dpi=150)
plt.close()

print("\nAll figures saved to:", OUTPUT_DIR)
print("All CSV tables saved to:", OUTPUT_DIR)

# ── REMOVABLE NOTES ────────────────────────────────────────────────────────────
# vp_advance, engine_up, engine_down, win_prob_late, loss_prob are all adjustable.
# The robber penalty applies only to tiles 0-5 (ROBBER_PENALTY_TILES).
# To change the penalty strength, adjust ROBBER_PENALTY_FACTOR (default 0.85).
# All rows are renormalised after construction, so small rounding errors are corrected.
# N = (I-Q)^{-1} may be slow on first run due to 162x162 matrix inversion; this is normal.
