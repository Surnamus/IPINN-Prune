import torch
import torch.nn as nn
from model2 import initIn, initOut, boundIn, boundOut, dataIn, dataOut
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
    ckpt = torch.load("checkpoints/model2_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    masks = [(p != 0).clone() for p in model.parameters() if p.dim() == 2]
    log_nu = torch.nn.Parameter(torch.log(torch.expm1(ckpt["nu"])).to(device))
    obsIn, obsOut = dataIn.clone(), dataOut.clone()

    x_c = torch.empty(10000, 1, device=device).uniform_(-1, 1).requires_grad_(True)
    t_c = torch.empty(10000, 1, device=device).uniform_(0, 3.0 / torch.pi).requires_grad_(True)

    u_c = model(x_c, t_c)
    ux_c = torch.autograd.grad(u_c, x_c, torch.ones_like(u_c), create_graph=True, retain_graph=True)[0]
    ut_c = torch.autograd.grad(u_c, t_c, torch.ones_like(u_c), create_graph=True, retain_graph=True)[0]
    uxx_c = torch.autograd.grad(ux_c, x_c, torch.ones_like(ux_c), create_graph=True, retain_graph=True)[0]
    with torch.no_grad():
        nu_c = torch.nn.functional.softplus(log_nu)
        res = torch.abs(ut_c + ux_c * u_c - uxx_c * nu_c).squeeze()
        w = res ** 2                       # k=2: sharpen toward high-residual region
        w = w / (w.mean() + 1e-12) + 1.0   # uniform floor
        idx = torch.multinomial(w / w.sum(), 1000, replacement=False)
        dataIn = torch.cat([x_c[idx].detach(), t_c[idx].detach()], dim=1)

    lbfgs = torch.optim.LBFGS(
    [*model.parameters(), log_nu],
    lr=1.0,
    max_iter=500,
    history_size=50,
    line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()

        initPred = model(initIn[:, 0:1], initIn[:, 1:2])
        loss_ic = torch.mean((initPred - initOut) ** 2)

        boundPred = model(boundIn[:, 0:1], boundIn[:, 1:2])
        loss_bc = torch.mean((boundPred - boundOut) ** 2)

        x = dataIn[:, 0:1].clone().detach().requires_grad_(True)
        t = dataIn[:, 1:2].clone().detach().requires_grad_(True)
        u = model(x, t)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), create_graph=True)[0]

        nu = torch.nn.functional.softplus(log_nu)
        loss_pde = torch.mean((u_t + u_x * u - u_xx * nu) ** 2)

        dataPred = model(obsIn[:, 0:1], obsIn[:, 1:2])
        loss_data = torch.mean((obsOut.reshape(-1, 1) - dataPred) ** 2)

        total_loss = loss_ic + loss_bc + loss_pde + loss_data
        total_loss.backward()
        with torch.no_grad():
            i = 0
            for p in model.parameters():
                if p.dim() == 2:
                    p.grad *= masks[i]
                    i += 1
        return total_loss

    prev_loss = float('inf')
    for epoch in range(1000):
        loss = lbfgs.step(closure)
        current_loss = loss.item()
        print(f"L-BFGS Epoch {epoch}, Loss: {current_loss}")

    # Early stopping if loss stops changing significantly
        if abs(prev_loss - current_loss) < 1e-8:
            print("L-BFGS converged.")
            break
        prev_loss = current_loss
    torch.save({
    "model_state_dict": model.state_dict(),
    "nu": torch.nn.functional.softplus(log_nu).detach()
    }, "checkpoints/model2_lbfgs.pt")
