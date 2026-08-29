import torch
import torch.nn as nn
import numpy as np
from cfdsolver import generategrid, to_tensor
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
X, T, vu, points = generategrid()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
(initIn, initOut), (boundIn, boundOut), (dataIn,dataOut) = to_tensor(X, T, vu,device=device)
if __name__ == "__main__":
  model = IPINN().to(device)
  log_nu = torch.nn.parameter.Parameter(torch.tensor(np.log(0.08), device=device), requires_grad=True)
  #optimizer = torch.optim.Adam(list(model.parameters())+[log_nu], lr=0.005)
  optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 0.001},   # Model weights stay stable
    {'params': [log_nu], 'lr': 0.0085}               # Parameter gets 10x higher learning rate
  ], lr=0.005)
  cosinesch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000, eta_min=1e-5)
  T_end = 112500 # Define T_end as the total number of training steps
  pruner = RigLScheduler(model,                           # model you created
                        optimizer,                       # optimizer (recommended = SGD w/ momentum)
                        dense_allocation=0.1,            # a float between 0 and 1 that designates how sparse you want the network to be
                                                            # (0.1 dense_allocation = 90% sparse)
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
      if step % 500 == 0:
          x_c = torch.empty(10000, 1, device=device).uniform_(-1, 1).requires_grad_(True)
          t_c = torch.empty(10000, 1, device=device).uniform_(0, 3.0 / torch.pi).requires_grad_(True)  # was (0,1)

          u_c = model(x_c, t_c)
          ux_c = torch.autograd.grad(u_c, x_c, torch.ones_like(u_c), create_graph=True, retain_graph=True)[0]   # needs create_graph (differentiated again below)
          ut_c = torch.autograd.grad(u_c, t_c, torch.ones_like(u_c),create_graph=True, retain_graph=True)[0]                       # not differentiated again
          uxx_c = torch.autograd.grad(ux_c, x_c, torch.ones_like(ux_c), create_graph=True, retain_graph=True)[0]                    # not differentiated again

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
      u_xx = torch.autograd.grad(u_x, x_grad, torch.ones_like(u), create_graph=True)[0]
      u_f=u
      nu = torch.nn.functional.softplus(log_nu)

      pde_residual = u_t + u_x * u_f - u_xx * nu
      loss_pde = torch.mean(pde_residual ** 2)

      dataPred = model(obsIn[:, 0:1], obsIn[:, 1:2])
      obsOut_reshaped = obsOut.reshape(-1, 1)

      loss_data = torch.mean((obsOut_reshaped - dataPred) ** 2)
      params = [p for p in model.parameters() if p.requires_grad]

      total_loss = loss_ic + loss_bc + loss_data + loss_pde
      #g1 = torch.autograd.grad(loss_ic, params, retain_graph=True)
      #g2 = torch.autograd.grad(loss_bc, params, retain_graph=True)
      #g3 = torch.autograd.grad(loss_data, params, retain_graph=True)
      #g4 = torch.autograd.grad(loss_pde, params)  # Last loss doesn't need retain_graph=True

      total_loss.backward(retain_graph=True)
      #No params means standard total loss
      if pruner():
        optimizer.step()
        cosinesch.step()
      if step % 1000 == 0:
        print(f"step {step}: loss_ic={loss_ic.item():.6f}, loss_bc={loss_bc.item():.6f}, loss_pde={loss_pde.item():.6f}, loss_data={loss_data.item():.6f}, nu={nu.item():.6f}")
      if step % 1000 == 0:
        print(pruner)
      if step % 15000 == 0:
        torch.save({"model_state_dict": model.state_dict(), "log_nu": log_nu.detach(), "nu": nu.detach(), "step": step, "pruner_state_dict": pruner.state_dict()}, f"checkpoints/model2_checkpoint_step{step}.pt")

  torch.save({"model_state_dict": model.state_dict(), "nu": nu.detach(), "pruner_state_dict": pruner.state_dict()}, "checkpoints/model2_checkpoint.pt")
