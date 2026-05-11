"""
Weighted Ensemble – Hallucination Detection
===========================================
Simple, clean ensemble:
  - Three sub-models (Softmax, Attention, FFN)
  - Weighted average of calibrated probabilities
    Weights: FFN=0.50, Attention=0.30, Softmax=0.20
  - CrossEntropyLoss + balanced class weights
  - Isotonic calibration per sub-model (fit on val set)
  - Optimal threshold search (F1) on val probs
  - Best single model vs ensemble comparison
"""

import random
import sys
from datetime import datetime

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Logger ────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, filename: str):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w", buffering=1, encoding="utf-8")
        header = f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "=" * 70 + "\n\n"
        self.terminal.write(header)
        self.log.write(header)

    def write(self, message: str):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = sys.__stdout__


LOG_FILE = "results_weighted_ensemble.txt"
logger = Logger(LOG_FILE)
sys.stdout = logger

# ── Load Data ─────────────────────────────────────────────────────────────────
SAVE_FILE = "project/combined.npz"
saved = np.load(SAVE_FILE, allow_pickle=True)
X_softmax     = np.array(list(saved["X_softmax"]))
X_attn_padded = np.array(list(saved["X_attn_padded"]))
X_ffn         = np.array(list(saved["X_ffn"]))
y             = np.array(list(saved["y"])).astype(int)

print(f"Loaded {len(y)} samples")
print(f"  Label 0 (non-hallucinated): {(y == 0).sum()}")
print(f"  Label 1 (hallucinated):     {(y == 1).sum()}")
print(f"  Positive rate:              {y.mean():.3f}")
print(f"  Random F1 baseline:         {2*y.mean()/(1+y.mean()):.3f}")
print(f"  X_softmax shape:  {X_softmax.shape}")
print(f"  X_attn shape:     {X_attn_padded.shape}")
print(f"  X_ffn shape:      {X_ffn.shape}")

# ── Stratified Split: 70 / 15 / 15 ───────────────────────────────────────────
indices = np.arange(len(y))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.30, random_state=SEED, stratify=y)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=SEED, stratify=y[temp_idx])

def prepare_data(X: np.ndarray, scaler_type: str = "standard"):
    X_tr  = X[train_idx]
    X_v   = X[val_idx]
    X_te  = X[test_idx]
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_v), scaler.transform(X_te), scaler

X_sm_tr,  X_sm_v,  X_sm_te,  scaler_sm  = prepare_data(X_softmax,     "standard")
X_att_tr, X_att_v, X_att_te, scaler_att = prepare_data(X_attn_padded, "minmax")
X_ffn_tr, X_ffn_v, X_ffn_te, scaler_ffn = prepare_data(X_ffn,         "standard")

y_train = y[train_idx]
y_val   = y[val_idx]
y_test  = y[test_idx]

print(f"\nSplit sizes — Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
print(f"  Train pos: {y_train.mean():.3f} | Val pos: {y_val.mean():.3f} | Test pos: {y_test.mean():.3f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ── Tensor helpers ────────────────────────────────────────────────────────────
def to_t(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype).to(device)

X_sm_tr_t,  X_sm_v_t,  X_sm_te_t  = to_t(X_sm_tr),  to_t(X_sm_v),  to_t(X_sm_te)
X_att_tr_t, X_att_v_t, X_att_te_t = to_t(X_att_tr), to_t(X_att_v), to_t(X_att_te)
X_ffn_tr_t, X_ffn_v_t, X_ffn_te_t = to_t(X_ffn_tr), to_t(X_ffn_v), to_t(X_ffn_te)
y_tr_t = to_t(y_train, torch.long)
y_v_t  = to_t(y_val,   torch.long)
y_te_t = to_t(y_test,  torch.long)

# ── Class weights ─────────────────────────────────────────────────────────────
cw_np = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(cw_np, dtype=torch.float32).to(device)
print(f"Class weights: {cw_np}")

# ── Sub-model architectures ───────────────────────────────────────────────────
class SoftmaxClassifier(nn.Module):
    def __init__(self, p: float = 0.50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 32),  nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(p * 0.7),
            nn.Linear(32, 16),  nn.BatchNorm1d(16),  nn.ReLU(), nn.Dropout(p * 0.5),
            nn.Linear(16, 2),
        )
    def forward(self, x): return self.net(x)

class AttentionClassifier(nn.Module):
    def __init__(self, p: float = 0.65):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128,  64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)

class FFNClassifier(nn.Module):
    def __init__(self, p: float = 0.70):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14336, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128,   64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)

# ── Utilities ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_probs(model: nn.Module, X_t: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.softmax(model(X_t), dim=1)[:, 1].cpu().numpy()

@torch.no_grad()
def f1_val(model: nn.Module, X_t: torch.Tensor, y_np: np.ndarray) -> float:
    model.eval()
    probs = get_probs(model, X_t)
    return f1_score(y_np, (probs > 0.5).astype(int), zero_division=0)

# ── Generic training loop ─────────────────────────────────────────────────────
def train_model(
    name: str, model: nn.Module,
    X_tr_t: torch.Tensor, X_v_t: torch.Tensor,
    lr: float = 3e-4, weight_decay: float = 1e-3,
    epochs: int = 120, batch_size: int = 128,
    patience: int = 10,
) -> nn.Module:

    print(f"\n{'='*60}\n  TRAINING {name}\n{'='*60}")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=False)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1, best_state, wait = 0.0, None, 0
    n = X_tr_t.size(0)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss, nb = 0.0, 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1

        vf1 = f1_val(model, X_v_t, y_val)
        scheduler.step(vf1)

        if epoch % 10 == 0:
            tf1 = f1_val(model, X_tr_t, y_train)
            print(f"  Ep {epoch:03d} | loss {epoch_loss/nb:.4f} | "
                  f"train_F1 {tf1:.4f} | val_F1 {vf1:.4f}")

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [Early stop @ ep {epoch}]")
                break

    model.load_state_dict(best_state)
    print(f"  Best val F1: {best_f1:.4f}")
    return model

# ── Train sub-models ──────────────────────────────────────────────────────────
clf_sm  = train_model("SOFTMAX",   SoftmaxClassifier().to(device),
                      X_sm_tr_t,  X_sm_v_t,  lr=5e-4, patience=10)

clf_att = train_model("ATTENTION", AttentionClassifier().to(device),
                      X_att_tr_t, X_att_v_t, lr=3e-4, patience=8)

clf_ffn = train_model("FFN",       FFNClassifier().to(device),
                      X_ffn_tr_t, X_ffn_v_t, lr=3e-4, patience=8)

# ── Isotonic calibration (fit on val set) ─────────────────────────────────────
print("\n" + "="*60)
print("  PROBABILITY CALIBRATION (Isotonic on Val Set)")
print("="*60)

def calibrate(model: nn.Module, X_v_t: torch.Tensor, y_v: np.ndarray) -> IsotonicRegression:
    raw = get_probs(model, X_v_t)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, y_v)
    return iso

cal_sm  = calibrate(clf_sm,  X_sm_v_t,  y_val)
cal_att = calibrate(clf_att, X_att_v_t, y_val)
cal_ffn = calibrate(clf_ffn, X_ffn_v_t, y_val)

def calibrated_probs(model: nn.Module, cal: IsotonicRegression,
                     X_t: torch.Tensor) -> np.ndarray:
    return cal.predict(get_probs(model, X_t))

# ── Weighted ensemble ─────────────────────────────────────────────────────────
# Weights: FFN > Attention > Softmax
ENSEMBLE_WEIGHTS = {"ffn": 0.50, "attention": 0.30, "softmax": 0.20}

print("\n" + "="*60)
print("  ENSEMBLE WEIGHTS")
print("="*60)
for k, v in ENSEMBLE_WEIGHTS.items():
    print(f"  {k:10s}: {v:.2f}")

def ensemble_probs(p_sm: np.ndarray, p_att: np.ndarray,
                   p_ffn: np.ndarray) -> np.ndarray:
    return (ENSEMBLE_WEIGHTS["softmax"]   * p_sm
          + ENSEMBLE_WEIGHTS["attention"] * p_att
          + ENSEMBLE_WEIGHTS["ffn"]       * p_ffn)

# Get calibrated probs on val and test
p_sm_v   = calibrated_probs(clf_sm,  cal_sm,  X_sm_v_t)
p_att_v  = calibrated_probs(clf_att, cal_att, X_att_v_t)
p_ffn_v  = calibrated_probs(clf_ffn, cal_ffn, X_ffn_v_t)
ens_v    = ensemble_probs(p_sm_v, p_att_v, p_ffn_v)

p_sm_te  = calibrated_probs(clf_sm,  cal_sm,  X_sm_te_t)
p_att_te = calibrated_probs(clf_att, cal_att, X_att_te_t)
p_ffn_te = calibrated_probs(clf_ffn, cal_ffn, X_ffn_te_t)
ens_te   = ensemble_probs(p_sm_te, p_att_te, p_ffn_te)

# ── Threshold search (on val set) ─────────────────────────────────────────────
def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(f1s.argmax())
    best_thresh = float(thresholds[best_idx] if best_idx < len(thresholds) else 0.5)
    return best_thresh, float(f1s[best_idx])

best_thresh, best_val_f1 = find_best_threshold(y_val, ens_v)
print(f"\nThreshold search on val set:")
print(f"  Best threshold: {best_thresh:.4f}  (val F1: {best_val_f1:.4f})")

# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(y_true: np.ndarray, probs: np.ndarray,
             threshold: float, label: str = "") -> dict:
    preds = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    rec  = recall_score(y_true,    preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    f1   = f1_score(y_true,        preds, zero_division=0)
    acc  = accuracy_score(y_true,  preds)
    roc  = roc_auc_score(y_true,   probs)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    if label:
        print(f"\n  [{label}]")
    print(f"    ROC-AUC:           {roc:.4f}")
    print(f"    Accuracy:          {acc:.4f}")
    print(f"    Precision (hallu): {prec:.4f}")
    print(f"    Recall    (hallu): {rec:.4f}")
    print(f"    F1:                {f1:.4f}")
    print(f"    Specificity:       {spec:.4f}")
    print(f"    NPV:               {npv:.4f}")
    print(f"    Threshold:         {threshold:.4f}")
    print(f"    Confusion matrix:")
    print(f"                  Pred 0   Pred 1")
    print(f"    Actual 0    [{tn:6d}] [{fp:6d}]")
    print(f"    Actual 1    [{fn:6d}] [{tp:6d}]")
    return {"roc_auc": roc, "acc": acc, "prec": prec, "rec": rec, "f1": f1}

# ── Individual sub-model results (test set) ───────────────────────────────────
print("\n\n" + "="*70)
print("  INDIVIDUAL SUB-MODEL RESULTS (TEST SET)")
print("="*70)

best_single_f1, best_single_name, best_single_results = 0.0, "", {}

for clf_name, clf, X_v_t, X_te_t, cal in [
    ("Softmax",   clf_sm,  X_sm_v_t,  X_sm_te_t,  cal_sm),
    ("Attention", clf_att, X_att_v_t, X_att_te_t, cal_att),
    ("FFN",       clf_ffn, X_ffn_v_t, X_ffn_te_t, cal_ffn),
]:
    pv  = calibrated_probs(clf, cal, X_v_t)
    pte = calibrated_probs(clf, cal, X_te_t)
    thresh, _ = find_best_threshold(y_val, pv)
    print(f"\n  {clf_name}:  best-val-thresh = {thresh:.4f}")
    res = evaluate(y_test, pte, thresh, label=f"{clf_name} @ best-val threshold")
    if res["f1"] > best_single_f1:
        best_single_f1, best_single_name, best_single_results = res["f1"], clf_name, res

# ── Ensemble results (test set) ───────────────────────────────────────────────
print("\n\n" + "="*70)
print("  WEIGHTED ENSEMBLE RESULTS (TEST SET)")
print("="*70)

res_050  = evaluate(y_test, ens_te, 0.50,
                    label="Weighted Ensemble @ threshold=0.50")
res_best = evaluate(y_test, ens_te, best_thresh,
                    label=f"Weighted Ensemble @ best-val threshold={best_thresh:.4f}")

# ── Best single vs ensemble ───────────────────────────────────────────────────
print("\n\n" + "="*70)
print("  BEST SINGLE MODEL vs WEIGHTED ENSEMBLE")
print("="*70)
print(f"  Best single: {best_single_name}")
print(f"    F1       : {best_single_results['f1']:.4f}")
print(f"    ROC-AUC  : {best_single_results['roc_auc']:.4f}")
print(f"  Weighted Ensemble (best thresh):")
print(f"    F1       : {res_best['f1']:.4f}")
print(f"    ROC-AUC  : {res_best['roc_auc']:.4f}")

delta_f1  = res_best["f1"]      - best_single_results["f1"]
delta_roc = res_best["roc_auc"] - best_single_results["roc_auc"]
print(f"  ΔF1:        {delta_f1:+.4f}  "
      f"({'✓ ensemble wins' if delta_f1 > 0 else '— single model competitive'})")
print(f"  ΔROC-AUC:   {delta_roc:+.4f}")

random_f1 = 2 * y_test.mean() / (1 + y_test.mean())
print(f"\n  Random F1 baseline:   {random_f1:.4f}")
print(f"  Ensemble vs random:   {res_best['f1'] - random_f1:+.4f}")
print(f"  Relative improvement: {((res_best['f1'] - random_f1) / random_f1 * 100):.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save({
    "softmax_state":   clf_sm.state_dict(),
    "attention_state": clf_att.state_dict(),
    "ffn_state":       clf_ffn.state_dict(),
    "ensemble_weights": ENSEMBLE_WEIGHTS,
    "threshold":       float(best_thresh),
    "seed":            SEED,
}, "weighted_ensemble.pth")

joblib.dump({
    "softmax":  scaler_sm,
    "attn":     scaler_att,
    "ffn":      scaler_ffn,
    "cal_sm":   cal_sm,
    "cal_att":  cal_att,
    "cal_ffn":  cal_ffn,
}, "weighted_scalers.pkl")

print("\n✓ Models saved  → weighted_ensemble.pth")
print("✓ Scalers saved → weighted_scalers.pkl")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SUMMARY")
print("="*70)
print("""
Sub-model architectures:
  Softmax:   20    → 32  → 16 → 2  (BN, dropout 0.50)
  Attention: 2048  → 128 → 64 → 2  (BN, dropout 0.65)
  FFN:       14336 → 128 → 64 → 2  (BN, dropout 0.70)

Ensemble: weighted average of calibrated probabilities
  FFN       weight: 0.50
  Attention weight: 0.30
  Softmax   weight: 0.20

Training:
  Loss:       CrossEntropyLoss + balanced class weights
  Optimizer:  Adam, weight decay 1e-3, grad clip 1.0
  Scheduler:  ReduceLROnPlateau (factor=0.5, patience=5)
  Early stop: patience 8–10 per sub-model
  Calibration: Isotonic regression per sub-model (fit on val)
  random_state=42 in all splits (reproducible)

Artifacts:
  weighted_ensemble.pth — model weights + metadata
  weighted_scalers.pkl  — scalers + calibrators
  results_weighted_ensemble.txt — training log
""")

logger.close()
print(f"✓ Log saved → {LOG_FILE}")
print("Done.")