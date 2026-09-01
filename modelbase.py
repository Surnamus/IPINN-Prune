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
    raw_nu = torch.nn.Parameter(torch.tensor(np.log(0.008), dtype=torch.float32, device=device), requires_grad=True)
    torch.manual_seed(42)
  #optimizer = torch.optim.Adam(list(model.parameters())+[log_nu], lr=0.005)
    optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 0.001},   # Model weights stay stable
    {'params': [raw_nu], 'lr': 1e-3}               # was 0.0085
    ], lr=0.005)
    cosinesch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000, eta_min=1e-4) #maybe 1e-4 or 1e-3 or 2e-4
    obsIn, obsOut = dataIn.clone(), dataOut.clone()
    #instead of RAD

    #x_static, t_static = sample_lhs_xt(8000, device)
    #dataIn = torch.cat([x_static, t_static], dim=1)
    for step in range(150000):
        #no RAD
        optimizer.zero_grad()

        xInit, tInit = initIn[:, 0:1], initIn[:, 1:2]
        initPred = model(xInit, tInit)
        initTarget = initOut
        loss_ic = torch.mean((initPred - initTarget) ** 2)

        xBound, tBound = boundIn[:, 0:1], boundIn[:, 1:2]
        boundPred = model(xBound, tBound)
        boundTarget = boundOut
        loss_bc = torch.mean((boundPred - boundTarget) ** 2)


        #dataPred = model(obsIn[:, 0:1], obsIn[:, 1:2])
        #obsOut_reshaped = obsOut.reshape(-1, 1)

        #loss_data = torch.mean((obsOut_reshaped - dataPred) ** 2)

        batch_size = 6000
        idx_data = torch.randperm(obsIn.shape[0], device=device)[:batch_size]
        obsIn_batch = obsIn[idx_data]
        obsOut_batch = obsOut[idx_data]

        x = obsIn_batch[:, 0:1].clone().detach().requires_grad_(True)
        t = obsIn_batch[:, 1:2].clone().detach().requires_grad_(True)
        u = model(x, t)

# PDE uses u, x, t:
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        nu = torch.exp(raw_nu)
        loss_pde = torch.mean((u_t + u * u_x - u_xx * nu) ** 2)

# Data loss reuses the exact same 'u':
        obsOut_reshaped = obsOut_batch.reshape(-1, 1)
        loss_data = torch.mean((obsOut_reshaped - u) ** 2)

        #dataPred = model(obsIn_static[:, 0:1], obsIn_static[:, 1:2])
        #obsOut_reshaped = obsOut_static.reshape(-1, 1)

        #loss_data = torch.mean((obsOut_reshaped - dataPred) ** 2)
        total_loss = loss_ic + loss_bc + loss_pde + loss_data #data loss scaling
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #torch.nn.utils.clip_grad_norm_([raw_nu], max_norm=1.0) #added clipping because apparently the high gradients are too much for lobotomised models, not for this one though
        optimizer.step()
        cosinesch.step()
        #optimizer.param_groups[1]['lr'] = 1e-3

        if step % 1000 == 0:
            print(f"step {step}: loss_ic={loss_ic.item():.6f}, loss_bc={loss_bc.item():.6f}, loss_pde={loss_pde.item():.6f}, loss_data={loss_data.item():.6f}, nu={nu.item():.6f}")
        if step % 15000 == 0:
            torch.save({"model_state_dict": model.state_dict(),  "raw_nu": raw_nu.detach(), "nu": nu.detach(), "step": step}, f"checkpoints/modelbase_checkpoint_step{step}.pt")
    torch.save({"model_state_dict": model.state_dict(),"nu": nu.detach(), "raw_nu": raw_nu.detach()}, "checkpoints/modelbase_checkpoint.pt")
#maknuti cosine annealing ako treba
