import os
import torch
import torch.nn as nn
from cfdsolver import generategrid

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(2, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


IPINN = PINN  # the inverse models use the exact same architecture


def load_checkpoint(path, model, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run the corresponding training script first."
        )
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


X_RANGE = (-1.0, 1.0)
T_RANGE = (0.0, 3.0 / torch.pi)


def sample_random_points(n_points, device, x_range=X_RANGE, t_range=T_RANGE):
    x_lo, x_hi = x_range
    t_lo, t_hi = t_range
    rand_x = x_lo + (x_hi - x_lo) * torch.rand(n_points, 1, device=device)
    rand_t = t_lo + (t_hi - t_lo) * torch.rand(n_points, 1, device=device)
    return rand_x, rand_t


def evaluate_model(model, x, t, label):
    with torch.no_grad():
        u_pred = model(x, t)
    print(f"  {label}: shape={tuple(u_pred.shape)}, "
          f"min={u_pred.min().item():.4f}, max={u_pred.max().item():.4f}, mean={u_pred.mean().item():.4f}")
    return u_pred


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(0)  # remove/change this for different random points each run
    N_RANDOM_POINTS = 1000

    X, T, vu, points = generategrid()
    grid_x = torch.tensor(X.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
    grid_t = torch.tensor(T.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
    rand_x, rand_t = sample_random_points(N_RANDOM_POINTS, device)

    print("--- Testing Inverse PINN (modelbase) ---")
    inverse_model = IPINN().to(device)
    ckpt = load_checkpoint("checkpoints/modelbase_checkpoint.pt", inverse_model, device)
    nu = ckpt["nu"]
    inverse_model.eval()
    print(f"  Discovered nu: {nu.item():.6f}")
    evaluate_model(inverse_model, grid_x, grid_t, "grid points")
    evaluate_model(inverse_model, rand_x, rand_t, "random points")



    print("\n--- Testing Pruned Inverse PINN (model1) ---")
    pruned1_model = IPINN().to(device)
    ckpt1 = load_checkpoint("checkpoints/model1_checkpoint.pt", pruned1_model, device)
    nu1 = ckpt1["nu"]
    pruned1_model.eval()
    print(f"  Discovered nu (model1): {nu1.item():.6f}")
    evaluate_model(pruned1_model, grid_x, grid_t, "grid points")
    evaluate_model(pruned1_model, rand_x, rand_t, "random points")

    print("\n--- Testing Pruned Inverse PINN (model2) ---")
    pruned2_model = IPINN().to(device)
    ckpt2 = load_checkpoint("checkpoints/model2_checkpoint.pt", pruned2_model, device)
    nu2 = ckpt2["nu"]
    pruned2_model.eval()
    print(f"  Discovered nu (model2): {nu2.item():.6f}")
    evaluate_model(pruned2_model, grid_x, grid_t, "grid points")
    evaluate_model(pruned2_model, rand_x, rand_t, "random points")
    print("\n--- Testing LBFGS-refined Inverse PINN (modelbase) ---")
    lbfgs_base_model = IPINN().to(device)
    ckpt_lbfgs_base = load_checkpoint("checkpoints/modelbase_lbfgs.pt", lbfgs_base_model, device)
    nu_lbfgs_base = ckpt_lbfgs_base["nu"]
    lbfgs_base_model.eval()
    print(f"  Refined nu (modelbase): {nu_lbfgs_base.item():.6f}")
    evaluate_model(lbfgs_base_model, grid_x, grid_t, "grid points")
    evaluate_model(lbfgs_base_model, rand_x, rand_t, "random points")

    print("\n--- Testing LBFGS-refined Pruned Inverse PINN (model1) ---")
    lbfgs1_model = IPINN().to(device)
    ckpt_lbfgs1 = load_checkpoint("checkpoints/model1_lbfgs.pt", lbfgs1_model, device)
    nu_lbfgs1 = ckpt_lbfgs1["nu"]
    lbfgs1_model.eval()
    print(f"  Refined nu (model1): {nu_lbfgs1.item():.6f}")
    evaluate_model(lbfgs1_model, grid_x, grid_t, "grid points")
    evaluate_model(lbfgs1_model, rand_x, rand_t, "random points")

    print("\n--- Testing LBFGS-refined Pruned Inverse PINN (model2) ---")
    lbfgs2_model = IPINN().to(device)
    ckpt_lbfgs2 = load_checkpoint("checkpoints/model2_lbfgs.pt", lbfgs2_model, device)
    nu_lbfgs2 = ckpt_lbfgs2["nu"]
    lbfgs2_model.eval()
    print(f"  Refined nu (model2): {nu_lbfgs2.item():.6f}")
    evaluate_model(lbfgs2_model, grid_x, grid_t, "grid points")
    evaluate_model(lbfgs2_model, rand_x, rand_t, "random points")
