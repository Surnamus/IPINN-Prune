import torch
import torch.nn as nn
import numpy as np
from cfdsolver import get_static_dataset, sample_lhs_xt
from scipy.stats import qmc
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
#    The form of the Burgers equation considered here is
#
#      du       du        d^2 u
#      -- + u * -- = nu * -----
#      dt       dx        dx^2
#
#    for -1.0 < x < +1.0, and 0 < t.
#
#    Initial conditions are u(x,0) = - sin(pi*x).  Boundary conditions
#    are u(-1,t) = u(+1,t) = 0.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_data = get_static_dataset(device=device)
X, T, vu = _data["X"], _data["T"], _data["vu"]
initIn, initOut = _data["initIn"], _data["initOut"]
boundIn, boundOut = _data["boundIn"], _data["boundOut"]
dataIn, dataOut = _data["dataIn"], _data["dataOut"]

if __name__ == "__main__":
    model = IPINN().to(device)
    log_nu = torch.nn.parameter.Parameter(torch.tensor(np.log(0.008), device=device), requires_grad=True)
    #optimizer = torch.optim.Adam(list(model.parameters())+[log_nu], lr=0.005)
    optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 0.001},   # Model weights stay stable
    {'params': [log_nu], 'lr': 5e-3}               # was 0.0085
    ], lr=0.005)
    cosinesch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000, eta_min=1e-4) #maybe 1e-4 or 1e-3 or 2e-4
    obsIn, obsOut = dataIn.clone(), dataOut.clone()
    for step in range(150000):
        if step % 1000 == 0:
            x_c, t_c = sample_lhs_xt(10000, device)
            u_c = model(x_c, t_c)
            ux_c = torch.autograd.grad(u_c, x_c, torch.ones_like(u_c), create_graph=True, retain_graph=True)[0]   # needs create_graph (differentiated again below)
            ut_c = torch.autograd.grad(u_c, t_c, torch.ones_like(u_c),create_graph=True, retain_graph=True)[0]                       # not differentiated again
            uxx_c = torch.autograd.grad(ux_c, x_c, torch.ones_like(ux_c), create_graph=True, retain_graph=True)[0]                  # not differentiated again

            with torch.no_grad():
                nu_c = torch.nn.functional.softplus(log_nu)
                res = torch.abs(ut_c + ux_c * u_c - uxx_c * nu_c).squeeze()
                w = res ** 2                                    # k=2: sharpen toward high-residual region
                w = w / (w.mean() + 1e-12) + 1.0                # uniform floor: low-residual regions still get picked sometimes
                idx = torch.multinomial(w / w.sum(), 1000, replacement=False)
                dataIn = torch.cat([x_c[idx].detach(), t_c[idx].detach()], dim=1)
        optimizer.zero_grad()

        xInit, tInit = initIn[:, 0:1], initIn[:, 1:2]
        initPred = model(xInit, tInit)
        initTarget = initOut
        loss_ic = torch.mean((initPred - initTarget) ** 2)

        xBound, tBound = boundIn[:, 0:1], boundIn[:, 1:2]
        boundPred = model(xBound, tBound)
        boundTarget = boundOut
        loss_bc = torch.mean((boundPred - boundTarget) ** 2)

        x = dataIn[:, 0:1].clone().detach().requires_grad_(True)
        t = dataIn[:, 1:2].clone().detach().requires_grad_(True)
        x_grad = x.clone().detach().requires_grad_(True)
        t_grad = t.clone().detach().requires_grad_(True)

        u = model(x_grad, t_grad)

        u_x = torch.autograd.grad(u, x_grad, torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t_grad, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_grad, torch.ones_like(u_x), create_graph=True)[0] #ones like u was here before in all of the scripts, it can be u but whatever
        u_f=u
        nu = torch.nn.functional.softplus(log_nu)

        pde_residual = u_t + u_x * u_f - u_xx * nu
        loss_pde = torch.mean(pde_residual ** 2)

        dataPred = model(obsIn[:, 0:1], obsIn[:, 1:2])
        obsOut_reshaped = obsOut.reshape(-1, 1)

        loss_data = torch.mean((obsOut_reshaped - dataPred) ** 2)
        total_loss = loss_ic + loss_bc + loss_pde + loss_data #data loss scaling
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [log_nu], max_norm=1.0) #added clipping because apparently the high gradients are too much for lobotomised models, not for this one though
        optimizer.step()
        cosinesch.step()
        if step % 1000 == 0:
            print(f"step {step}: loss_ic={loss_ic.item():.6f}, loss_bc={loss_bc.item():.6f}, loss_pde={loss_pde.item():.6f}, loss_data={loss_data.item():.6f}, nu={nu.item():.6f}")
        if step % 15000 == 0:
            torch.save({"model_state_dict": model.state_dict(), "log_nu": log_nu.detach(), "nu": nu.detach(), "step": step}, f"checkpoints/modelbase_checkpoint_step{step}.pt")
    torch.save({"model_state_dict": model.state_dict(), "nu": nu.detach()}, "checkpoints/modelbase_checkpoint.pt")
#maknuti cosine annealing ako treba
