import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from config import DEVICE


def _predict_batched(model, X, batch_size=512):
    parts = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i + batch_size]).to(DEVICE)
            parts.append(torch.sigmoid(model(xb)).squeeze(1).cpu().numpy())
    return np.concatenate(parts)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=1024, dropout=0.4):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden)
        self.bn1  = nn.BatchNorm1d(hidden)
        self.fc2  = nn.Linear(hidden, hidden)
        self.bn2  = nn.BatchNorm1d(hidden)
        self.fc3  = nn.Linear(hidden, hidden)
        self.bn3  = nn.BatchNorm1d(hidden)
        self.fc4  = nn.Linear(hidden, hidden)
        self.bn4  = nn.BatchNorm1d(hidden)
        self.fc5  = nn.Linear(hidden, 512)
        self.head = nn.Linear(512, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.bn1(self.fc1(x)))
        h = self.drop(h)
        h = F.gelu(self.bn2(self.fc2(h))) + h
        h = self.drop(h)
        h = F.gelu(self.bn3(self.fc3(h)))
        h = self.drop(h)
        h = F.gelu(self.bn4(self.fc4(h))) + h
        h = self.drop(h)
        return self.head(F.gelu(self.fc5(h)))


def focal_bce(logits, targets, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    return (alpha * (1 - p_t) ** gamma * bce).mean()


def _run_training(model, tr_ld, X_vl, y_vl, epochs, patience):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_f1, best_state, no_imp = 0.0, None, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            focal_bce(model(xb), yb).backward()
            opt.step()
        sch.step()

        vl_probs = _predict_batched(model, X_vl)
        f1 = f1_score(y_vl, (vl_probs > 0.5).astype(int))
        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    model.load_state_dict(best_state)
    return model


def train_mlp(X_tr, y_tr, X_vl, y_vl, X_te, in_dim, seed=42, epochs=200, patience=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)
    )
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=256, shuffle=True)
    model = MLP(in_dim).to(DEVICE)
    model = _run_training(model, tr_ld, X_vl, y_vl, epochs, patience)
    return _predict_batched(model, X_vl), _predict_batched(model, X_te)


def train_final_mlp(X_all, y_all, in_dim, seed=42, epochs=150, patience=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_val  = max(1, int(0.1 * len(X_all)))
    idx    = np.random.permutation(len(X_all))
    tr_idx, vl_idx = idx[n_val:], idx[:n_val]
    tr_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_all[tr_idx]), torch.FloatTensor(y_all[tr_idx])
    )
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=256, shuffle=True)
    model = MLP(in_dim).to(DEVICE)
    model = _run_training(model, tr_ld, X_all[vl_idx], y_all[vl_idx], epochs, patience)
    return model
