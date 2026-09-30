"""Share of the 10 aligned seeds' spread captured by the top PCA directions."""
import numpy as np
import torch
for m in ["M1", "M2", "M4"]:
    th = torch.stack([torch.nn.utils.parameters_to_vector([v for k, v in sorted(
        torch.load(f"figures/landscape/fig1/aligned/{m}_s{s}.pt", map_location="cpu", weights_only=False)["model_state_dict"].items(),
        key=lambda kv: kv[0])]) for s in range(10)]).double()
    C = th - th.mean(0)
    ev = torch.linalg.svdvals(C) ** 2
    frac = (ev / ev.sum()).numpy()
    print(m, "variance share of PCs 1-9:", np.round(frac[:9], 3), " top-2 total:", round(float(frac[:2].sum()), 3))
