import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
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
#    are u(-1,t) = u(+1,t) = 0.  The viscosity parameter nu is taken
#    to be 0.01 / pi, although this is not essential.
X, T, vu, points = generategrid()
(initIn, initOut), (boundIn, boundOut), (dataIn,dataOut) = to_tensor(X, T, vu)
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for step in range(200000):
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
    nu=0.01 / np.pi
    pde_residual = u_t + u_x * u_f - u_xx * nu
    loss_pde = torch.mean(pde_residual ** 2)

    total_loss = loss_ic + loss_bc + loss_pde
    total_loss.backward()
    optimizer.step()
