import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# config (edit here, no argparse)
# -----------------------------
N_PATIENTS   = 128      # toy
M_MODALITIES = 5        # e.g. [wsi, text, tab, pathLN, pathPrimary]
D_IN         = 256      # raw feature dim per modality
D_LATENT     = 128      # shared dim
K_HUBS       = 3        # polyhubs
K_SUPP       = 16       # support points per hub
B_MODAL      = 32       # batch per modality
B_PATIENT    = 32       # batch for patient-level losses
SINK_ITERS   = 60
EPS          = 0.5
TAU          = 0.3
TAU_SEP      = 1.0
LR           = 3e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# toy dataset (random, but with missing modalities)
# X[m][i] -> feature for patient i, modality m (or None if missing)
# -----------------------------
torch.manual_seed(0)
X = []
for m in range(M_MODALITIES):
    feats = torch.randn(N_PATIENTS, D_IN)
    mask  = torch.rand(N_PATIENTS) > 0.15  # 15% missing
    feats[~mask] = 0.
    X.append((feats, mask))  # (N,D_IN), (N,)
# survival targets (toy)
times  = torch.rand(N_PATIENTS) * 10.0
events = (torch.rand(N_PATIENTS) > 0.4).float()

# -----------------------------
# model parts
# -----------------------------
class WPA(nn.Module):
    def __init__(self):
        super().__init__()
        # per-modality linear encoders to shared dim
        self.encoders = nn.ModuleList([nn.Linear(D_IN, D_LATENT) for _ in range(M_MODALITIES)])
        # hubs: supports and weights
        self.hub_supports = nn.Parameter(torch.randn(K_HUBS, K_SUPP, D_LATENT) * 0.1)
        self.hub_logits   = nn.Parameter(torch.zeros(K_HUBS, K_SUPP))
        # fusion + survival head
        self.fuse = nn.Linear(D_LATENT, D_LATENT)
        self.surv = nn.Linear(D_LATENT, 1)

    def encode_modality(self, m, x):
        return self.encoders[m](x)

    def hub_weights(self):
        return F.softmax(self.hub_logits, dim=1)  # (K, K_SUPP)

model = WPA().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)

# -----------------------------
# sinkhorn (log-domain, balanced, small)
# P: (n,d), w: (n,)   Q: (m,d), v: (m,)
# returns OT_eps(P,Q) scalar
# -----------------------------
def sinkhorn_cost(P, w, Q, v, eps=EPS, iters=SINK_ITERS):
    # cost matrix
    C = torch.cdist(P, Q, p=2).pow(2)  # (n,m)
    K = torch.exp(-C / eps)
    u = torch.ones_like(w) / w.size(0)
    vhat = torch.ones_like(v) / v.size(0)
    for _ in range(iters):
        u = w / (K @ vhat + 1e-9)
        vhat = v / (K.t() @ u + 1e-9)
    pi = u.unsqueeze(1) * K * vhat.unsqueeze(0)  # (n,m)
    return (pi * C).sum()

def sinkhorn_div(P, w, Q, v):
    pq = sinkhorn_cost(P, w, Q, v)
    pp = sinkhorn_cost(P, w, P, w)
    qq = sinkhorn_cost(Q, v, Q, v)
    return pq - 0.5 * pp - 0.5 * qq

# -----------------------------
# Cox partial log-likelihood (toy, no ties)
# -----------------------------
def cox_loss(risk, time, event):
    # risk: (B,1); time/event: (B,)
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cum = risk.logsumexp(dim=0)
    # actually need running logsumexp
    log_cum = []
    cur = None
    for i in range(risk.size(0)):
        part = risk[i:].logsumexp(dim=0)
        log_cum.append(part)
    log_cum = torch.stack(log_cum)
    ll = (risk.squeeze() - log_cum) * event
    return -ll.mean()

# -----------------------------
# training loop (toy, single file)
# -----------------------------
for step in range(200):  # 200 toy steps
    model.train()
    opt.zero_grad()

    # -------- modality-level sampler (OT-APA + hub-sep) --------
    hub_supp = model.hub_supports  # (K,Ks,D)
    hub_w    = model.hub_weights() # (K,Ks)
    L_apa = 0.
    # store modality-level distances for logging
    for m in range(M_MODALITIES):
        feats, mask = X[m]
        idx = torch.where(mask)[0]
        idx = idx[torch.randint(0, idx.numel(), (min(B_MODAL, idx.numel()),))]
        z = model.encode_modality(m, feats[idx].to(DEVICE))  # (Bm,D)
        P = z
        w = torch.full((P.size(0),), 1.0 / P.size(0), device=DEVICE)
        # compute distance to each hub
        dists = []
        for k in range(K_HUBS):
            Q = hub_supp[k]        # (Ks,D)
            v = hub_w[k]           # (Ks,)
            d = sinkhorn_div(P, w, Q, v)
            dists.append(d)
        dists = torch.stack(dists)  # (K,)
        # soft assign
        a = F.softmax(-dists / TAU, dim=0)   # (K,)
        L_apa = L_apa + (a * dists).sum()
    L_apa = L_apa / M_MODALITIES

    # hub separation
    L_sep = 0.
    cnt = 0
    for k in range(K_HUBS):
        for k2 in range(K_HUBS):
            if k == k2:
                continue
            d = sinkhorn_div(hub_supp[k], hub_w[k], hub_supp[k2], hub_w[k2])
            pen = F.relu(TAU_SEP - d)
            L_sep = L_sep + pen
            cnt += 1
    if cnt > 0:
        L_sep = L_sep / cnt

    # -------- patient-level sampler (IUGA-lite + surv) --------
    # sample patients with at least 2 modalities
    has2 = []
    for i in range(N_PATIENTS):
        c = 0
        for m in range(M_MODALITIES):
            if X[m][1][i]:
                c += 1
        if c >= 2:
            has2.append(i)
    has2 = torch.tensor(has2)
    if has2.numel() > 0:
        pick = has2[torch.randint(0, has2.numel(), (min(B_PATIENT, has2.numel()),))]
    else:
        pick = torch.arange(min(B_PATIENT, N_PATIENTS))
    pick = pick.to(DEVICE)

    L_iuga = 0.
    surv_feats = []
    surv_times = []
    surv_events = []
    for i in pick:
        # per-patient hub posteriors
        posteriors = []
        patient_embeds = []
        for m in range(M_MODALITIES):
            feats, mask = X[m]
            if not mask[i]:
                continue
            z = model.encode_modality(m, feats[i].unsqueeze(0).to(DEVICE))  # (1,D)
            # delta vs hub
            dists = []
            for k in range(K_HUBS):
                Q = hub_supp[k]
                v = hub_w[k]
                d = sinkhorn_div(z, torch.ones(1, device=DEVICE), Q, v)
                dists.append(d)
            dists = torch.stack(dists)  # (K,)
            a = F.softmax(-dists / TAU, dim=0)  # (K,)
            posteriors.append(a)
            patient_embeds.append(z)
        # IUGA-lite: symmetric KL across posteriors
        if len(posteriors) > 1:
            pl = 0.
            cntp = 0
            for p1 in range(len(posteriors)):
                for p2 in range(p1 + 1, len(posteriors)):
                    p = posteriors[p1]
                    q = posteriors[p2]
                    kl1 = F.kl_div(p.log(), q, reduction='batchmean')
                    kl2 = F.kl_div(q.log(), p, reduction='batchmean')
                    pl = pl + kl1 + kl2
                    cntp += 1
            L_iuga = L_iuga + pl / cntp
        # fusion for survival
        if len(patient_embeds) > 0:
            emb = torch.stack(patient_embeds, dim=0).mean(0)  # (1,D)
            emb = model.fuse(emb)
            surv_feats.append(emb)
            surv_times.append(times[i].to(DEVICE))
            surv_events.append(events[i].to(DEVICE))

    if len(surv_feats) > 0:
        surv_feats = torch.cat(surv_feats, dim=0)  # (B,D)
        risk = model.surv(surv_feats)              # (B,1)
        surv_times = torch.stack(surv_times)
        surv_events = torch.stack(surv_events)
        L_surv = cox_loss(risk, surv_times, surv_events)
    else:
        L_surv = torch.tensor(0., device=DEVICE)

    if pick.numel() > 0:
        L_iuga = L_iuga / max(1, pick.numel())
    else:
        L_iuga = torch.tensor(0., device=DEVICE)

    # -------- total loss --------
    L = 1.0 * L_apa + 0.5 * L_sep + 0.5 * L_iuga + 1.0 * L_surv
    L.backward()
    opt.step()

    if step % 20 == 0:
        print(f"step {step:03d} | L={L.item():.4f} | APA={L_apa.item():.4f} | SEP={L_sep.item():.4f} | IUGA={L_iuga.item():.4f} | SURV={L_surv.item():.4f}")
