import torch
import torch.nn as nn
from cfdsolver import get_static_dataset, sample_lhs_xt
class IPINN(nn.Module):
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
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IPINN().to(device)
    ckpt = torch.load("checkpoints/modelbase_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    masks = [(p != 0).clone() for p in model.parameters() if p.dim() == 2]
    raw_nu = torch.nn.Parameter(ckpt["raw_nu"].to(device))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _data = get_static_dataset(device=device)
    initIn, initOut = _data["initIn"], _data["initOut"]
    boundIn, boundOut = _data["boundIn"], _data["boundOut"]
    obsIn, obsOut = _data["dataIn"], _data["dataOut"]  # .clone() no longer needed — fresh load, nothing to defend against


    #x_static, t_static = sample_lhs_xt(8000, device)
    #dataIn = torch.cat([x_static, t_static], dim=1)


    #new
    torch.manual_seed(42) # Keep it static so L-BFGS doesn't get confused by changing loss landscapes
    idx_lbfgs = torch.randperm(obsIn.shape[0])[:5000]
    obsIn = obsIn[idx_lbfgs]
    obsOut = obsOut[idx_lbfgs]
    #new
    lbfgs = torch.optim.LBFGS(
    [*model.parameters(), raw_nu],
    lr=1.0,
    max_iter=5000,
    history_size=50,
    line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()

        initPred = model(initIn[:, 0:1], initIn[:, 1:2])
        loss_ic = torch.mean((initPred - initOut) ** 2)

        boundPred = model(boundIn[:, 0:1], boundIn[:, 1:2])
        loss_bc = torch.mean((boundPred - boundOut) ** 2)

        x = obsIn[:, 0:1].clone().detach().requires_grad_(True)
        t = obsIn[:, 1:2].clone().detach().requires_grad_(True)
        u = model(x, t)

        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        nu = torch.exp(raw_nu)
        loss_pde = torch.mean((u_t + u * u_x - u_xx * nu) ** 2)

# Reuse 'u' for Data Loss:
        loss_data = torch.mean((obsOut.reshape(-1, 1) - u) ** 2)
        total_loss = loss_ic + loss_bc + loss_pde + loss_data
        print(f"L-BFGS Loss: {total_loss.item():.6f} | Nu: {nu.item():.6f}")
        total_loss.backward()
        return total_loss


    loss = lbfgs.step(closure)
    torch.save({
    "model_state_dict": model.state_dict(),
    "nu": torch.exp(raw_nu).detach(),
    "raw_nu": raw_nu.detach(),
    }, "checkpoints/modelbase_lbfgs.pt")
