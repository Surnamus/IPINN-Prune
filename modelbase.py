import torch
import torch.nn as nn
import numpy as np
from cfdsolver import generategrid, to_tensor
from pruningalg import RigLScheduler
class IPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
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
model = IPINN().to(device)
nu = torch.nn.parameter.Parameter(torch.tensor(0.8), requires_grad=True)
optimizer = torch.optim.Adam(list(model.parameters())+[nu], lr=0.005)

for step in range(2000000):
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

    pde_residual = u_t + u_x * u_f - u_xx * nu
    loss_pde = torch.mean(pde_residual ** 2)

    dataPred = model(dataIn[:, 0:1], dataIn[:, 1:2])
    dataOut = dataOut.reshape(-1, 1)

    loss_data=torch.mean((dataOut-dataPred)**2)
    total_loss = loss_ic + loss_bc + loss_pde + loss_data
    total_loss.backward()
    #the one with post-order goes here
    optimizer.step()
torch.save({"model_state_dict": model.state_dict(), "nu": nu.detach()}, "modelbase_checkpoint.pt")
