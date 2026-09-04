import argparse
import torch
import torch.nn as nn
import numpy as np
from cfdsolver import get_static_dataset, sample_lhs_xt
from pruningalg import RigLScheduler
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
  parser = argparse.ArgumentParser()
  parser.add_argument("--sparsity", type=float, required=True,
                       help="Target sparsity in (0,1); dense_allocation = 1 - sparsity")
  args = parser.parse_args()
  sparsity = args.sparsity
  dense_allocation = 1.0 - sparsity

  model = IPINN().to(device)
  raw_nu = torch.nn.Parameter(torch.tensor(np.log(0.008), dtype=torch.float32, device=device), requires_grad=True)
  torch.manual_seed(42)
  #optimizer = torch.optim.Adam(list(model.parameters())+[raw_nu], lr=0.005)
  optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 0.001},   # Model weights stay stable
    {'params': [raw_nu], 'lr': 0.001}               # Parameter gets 10x higher learning rate
  ], lr=0.005)
  cosinesch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000, eta_min=1e-4) #maybe 1e-4 or 1e-3
  T_end = 112500 # Define T_end as the total number of training steps
  pruner = RigLScheduler(model,                           # model you created
                        optimizer,                       # optimizer (recommended = SGD w/ momentum)
                        dense_allocation=dense_allocation, # a float between 0 and 1 that designates how sparse you want the network to be
                                                            # (1 - sparsity, passed via --sparsity)
                        sparsity_distribution='uniform', # distribution hyperparam within the paper, currently only supports `uniform`
                        T_end=T_end,                     # T_end hyperparam within the paper (recommended = 75% * total_iterations)
                        delta=100,                       # delta hyperparam within the paper (recommended = 100)
                        alpha=0.3,                       # alpha hyperparam within the paper (recommended = 0.3)
                        grad_accumulation_n=1,           # new hyperparam contribution (not in the paper)
                                                            # for more information, see the `Contributions Beyond the Paper` section
                        static_topo=False,               # if True, the topology will be frozen, in other words RigL will not do it's job
                                                            # (for debugging)
                        ignore_linear_layers=False,      # if True, linear layers in the network will be kept fully dense
                        state_dict=None)                 # if you have checkpointing enabled for your training script, you should save
                                                            # `pruner.state_dict()` and when resuming pass the loaded `state_dict` into
                                                            # the pruner constructor
  obsIn, obsOut = dataIn.clone(), dataOut.clone()

  for step in range(150000):
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
        torch.nn.utils.clip_grad_norm_([raw_nu], max_norm=1.0) #added clipping because apparently the high gradients are too much for lobotomised models, not for this one though

        if pruner():
            optimizer.step()
            cosinesch.step()
        if step % 1000 == 0:
            print(f"step {step}: loss_ic={loss_ic.item():.6f}, loss_bc={loss_bc.item():.6f}, loss_pde={loss_pde.item():.6f}, loss_data={loss_data.item():.6f}, nu={nu.item():.6f}")
        if step % 1000 == 0:
            print(pruner)
        if step % 75000 == 0:
            torch.save({"model_state_dict": model.state_dict(), "raw_nu": raw_nu.detach(), "nu": nu.detach(), "step": step, "sparsity": sparsity, "pruner_state_dict": pruner.state_dict()}, f"checkpoints/model2_checkpoint_sparsity{sparsity}_step{step}.pt")

  torch.save({"model_state_dict": model.state_dict(), "raw_nu": raw_nu.detach(), "nu": nu.detach(), "sparsity": sparsity, "pruner_state_dict": pruner.state_dict()}, f"checkpoints/model2_checkpoint_sparsity{sparsity}.pt")
