"""# Pruning Algorithm(s)"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
class PruneMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    def compute_mask(self, t, default_mask):

        #this is the method where the code goes
        # https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html
        #see the tutorial for more info
        # but this is how it begins i suppose
        #after that, i think that PINN code wont be modified at all, so it will be called as an existing code, probably by making an another object for the same PINN class in order to reduce boilerplate

        mask = default_mask.clone()

        #code, do some fun stuff with mask
        mask[torch.abs(t) < self.threshold] = 0
        return mask
EXCLUDED_TYPES = (torch.nn.BatchNorm2d, )

def get_weighted_layers(model, i=0, layers=None, linear_layers_mask=None):
    if layers is None:
        layers = []
    if linear_layers_mask is None:
        linear_layers_mask = []

    items = model._modules.items()
    if i == 0:
        items = [(None, model)]

    for layer_name, p in items:
        if isinstance(p, torch.nn.Linear):
            layers.append([p])
            linear_layers_mask.append(1)
        elif hasattr(p, 'weight') and type(p) not in EXCLUDED_TYPES:
            layers.append([p])
            linear_layers_mask.append(0)
        elif isinstance(p, torchvision.models.resnet.Bottleneck) or isinstance(p, torchvision.models.resnet.BasicBlock):
            _, linear_layers_mask, i = get_weighted_layers(p, i=i + 1, layers=layers, linear_layers_mask=linear_layers_mask)
        else:
            _, linear_layers_mask, i = get_weighted_layers(p, i=i + 1, layers=layers, linear_layers_mask=linear_layers_mask)

    return layers, linear_layers_mask, i

def get_W(model, return_linear_layers_mask=False):
    layers, linear_layers_mask, _ = get_weighted_layers(model)

    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1
        W.append(layer[idx].weight)

    assert len(W) == len(linear_layers_mask)

    if return_linear_layers_mask:
        return W, linear_layers_mask
    return W

""" implementation of https://arxiv.org/abs/1911.11134 """
#directly from https://github.com/verbiiz/rigl-torch/blob/master/rigl_torch/RigL.py#L245
import math
import numpy as np
import torch
import torchvision

class IndexMaskHook:
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return 'IndexMaskHook'

    @torch.no_grad()
    def __call__(self, grad):
        mask = self.scheduler.backward_masks[self.layer]

        # only calculate dense_grads when necessary
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                # initialize as all 0s so we can do a rolling average
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad / self.scheduler.grad_accumulation_n
        else:
            self.dense_grad = None

        return grad * mask


def _create_step_wrapper(scheduler, optimizer):
    _unwrapped_step = optimizer.step
    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
    optimizer.step = _wrapped_step


class RigLScheduler:

    def __init__(self, model, optimizer, dense_allocation=1, T_end=None, sparsity_distribution='uniform', ignore_linear_layers=False, delta=100, alpha=0.3, static_topo=False, grad_accumulation_n=1, state_dict=None):
        if dense_allocation <= 0 or dense_allocation > 1:
            raise Exception('Dense allocation must be on the interval (0, 1]. Got: %f' % dense_allocation)

        self.model = model
        self.optimizer = optimizer

        self.W, self._linear_layers_mask = get_W(model, return_linear_layers_mask=True)

        # Explicit mask application in the training loop keeps score gradients dense.

        self.dense_allocation = dense_allocation
        self.N = [torch.numel(w) for w in self.W]

        if state_dict is not None:
            # Current flat checkpoints omit these original constructor settings.
            self.sparsity_distribution = sparsity_distribution
            self.static_topo = static_topo
            self.grad_accumulation_n = grad_accumulation_n
            self.ignore_linear_layers = ignore_linear_layers
            self.delta_T, self.alpha, self.T_end = delta, alpha, T_end
            self.load_state_dict(state_dict)
            self.apply_mask_to_weights()

        else:
            self.sparsity_distribution = sparsity_distribution
            self.static_topo = static_topo
            self.grad_accumulation_n = grad_accumulation_n
            self.ignore_linear_layers = ignore_linear_layers
            self.backward_masks = None

            # define sparsity allocation
            self.S = []
            for i, (W, is_linear) in enumerate(zip(self.W, self._linear_layers_mask)):
                # when using uniform sparsity, the first layer is always 100% dense
                # UNLESS there is only 1 layer
                is_first_layer = i == 0
                if is_first_layer and self.sparsity_distribution == 'uniform' and len(self.W) > 1:
                    self.S.append(0)

                elif is_linear and self.ignore_linear_layers:
                    # if choosing to ignore linear layers, keep them 100% dense
                    self.S.append(0)

                else:
                    self.S.append(1-dense_allocation)

            # randomly sparsify model according to S
            self.random_sparsify()

            # scheduler keeps a log of how many times it's called. this is how it does its scheduling
            self.step = 0
            self.rigl_steps = 0

            # define the actual schedule
            self.delta_T = delta
            self.alpha = alpha
            self.T_end = T_end

        self.changed_connections = getattr(self, "changed_connections", 0)

        # Keep original score holders, without gradient-masking hooks.
        self.backward_hook_objects = []
        for i, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[i] <= 0:
                self.backward_hook_objects.append(None)
                continue

            if getattr(w, '_has_rigl_backward_hook', False):
                raise Exception('This model already has been registered to a RigLScheduler.')

            self.backward_hook_objects.append(IndexMaskHook(i, self))
            # Mask gradients only after computing dense drop/growth scores.

        assert self.grad_accumulation_n > 0 and self.grad_accumulation_n < delta
        assert self.sparsity_distribution in ('uniform', )




    def state_dict(self):
        obj = {
            'dense_allocation': self.dense_allocation,
            'S': self.S,
            'N': self.N,
            'hyperparams': {
                'delta_T': self.delta_T,
                'alpha': self.alpha,
                'T_end': self.T_end,
                'ignore_linear_layers': self.ignore_linear_layers,
                'static_topo': self.static_topo,
                'sparsity_distribution': self.sparsity_distribution,
                'grad_accumulation_n': self.grad_accumulation_n,
            },
            'step': self.step,
            'rigl_steps': self.rigl_steps,
            'backward_masks': self.backward_masks,
            '_linear_layers_mask': self._linear_layers_mask,
            'changed_connections': self.changed_connections,
        }

        return obj

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if type(v) == dict:
                self.load_state_dict(v)
            setattr(self, k, v)


    @torch.no_grad()
    def random_sparsify(self):
        self.backward_masks = []
        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:
                self.backward_masks.append(None)
                continue

            n = self.N[l]
            s = int(self.S[l] * n)
            perm = torch.randperm(n,device=w.device)
            perm = perm[:s]
            flat_mask = torch.ones(n, device=w.device)
            flat_mask[perm] = 0
            mask = torch.reshape(flat_mask, w.shape)


            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)


    def __str__(self):
        s = 'RigLScheduler(\n'
        s += 'layers=%i,\n' % len(self.N)

        # calculate the number of non-zero elements out of the total number of elements
        N_str = '['
        S_str = '['
        sparsity_percentages = []
        total_params = 0
        total_conv_params = 0
        total_nonzero = 0
        total_conv_nonzero = 0

        for N, S, mask, W, is_linear in zip(self.N, self.S, self.backward_masks, self.W, self._linear_layers_mask):
            actual_S = torch.sum(W[mask == 0] == 0).item()
            N_str += ('%i/%i, ' % (N-actual_S, N))
            sp_p = float(N-actual_S) / float(N) * 100
            S_str += '%.2f%%, ' % sp_p
            sparsity_percentages.append(sp_p)
            total_params += N
            total_nonzero += N-actual_S
            if not is_linear:
                total_conv_nonzero += N-actual_S
                total_conv_params += N

        N_str = N_str[:-2] + ']'
        S_str = S_str[:-2] + ']'

        s += 'nonzero_params=' + N_str + ',\n'
        s += 'nonzero_percentages=' + S_str + ',\n'
        s += 'total_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_nonzero, total_params, float(total_nonzero)/float(total_params)*100)) + ',\n'
        if total_conv_params != 0:
            s += 'total_CONV_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_conv_nonzero, total_conv_params, float(total_conv_nonzero)/float(total_conv_params)*100)) + ',\n'
        else:
            s += 'total_CONV_nonzero_params=0/0 (0.00%),\n'
        s += 'step=' + str(self.step) + ',\n'
        s += 'num_rigl_steps=' + str(self.rigl_steps) + ',\n'
        s += 'ignoring_linear_layers=' + str(self.ignore_linear_layers) + ',\n'
        s += 'sparsity_distribution=' + str(self.sparsity_distribution) + ',\n'

        return s + ')'


    @torch.no_grad()
    def reset_momentum(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            for value in self.optimizer.state.get(w, {}).values():
                if torch.is_tensor(value) and value.shape == w.shape:
                    value.mul_(mask)



    @torch.no_grad()
    def apply_mask_to_weights(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            w *= mask
    #accidentally deleted this method
    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            if w.grad is not None:
                w.grad *= mask
    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next rigl step is,
        if it's within `self.grad_accumulation_n` steps, return True.
        """

        if self.step >= self.T_end:
            return False

        steps_til_next_rigl_step = self.delta_T - (self.step % self.delta_T)
        return steps_til_next_rigl_step <= self.grad_accumulation_n


    def cosine_annealing(self):
        return self.alpha / 2 * (1 + math.cos((self.step * math.pi) / self.T_end))


    #def __call__(self):
        #self.step += 1
        #if self.static_topo:
            #return True
        #if (self.step % self.delta_T) == 0 and self.step < self.T_end: # check schedule
            #self._rigl_step()
            #self.rigl_steps += 1
            #return False
        #return True

    def __call__(self, loss_1=None, loss_2=None, loss_3=None, loss_4=None,alpha=1,beta=1):
      self.step += 1
      if self.static_topo:
        return True

      if (self.step % self.delta_T) == 0 and self.step < self.T_end:
        losspassed = all(l is not None for l in (loss_1, loss_2, loss_3, loss_4))

        if losspassed:
            g1 = torch.autograd.grad(loss_1, self.W, retain_graph=True)
            g2 = torch.autograd.grad(loss_2, self.W, retain_graph=True)
            g3 = torch.autograd.grad(loss_3, self.W, retain_graph=True)
            g4 = torch.autograd.grad(loss_4, self.W, retain_graph=True)

            with torch.no_grad():
                for l in range(len(self.W)):
                    if self.backward_hook_objects[l] is not None:
                      self.backward_hook_objects[l].dense_grad_drop = alpha*(g1[l] + g2[l] + g3[l]) + beta*g4[l]
                      self.backward_hook_objects[l].dense_grad_grow = g1[l] + g2[l] + g3[l]
        else:
          #fallback to standard DST
            with torch.no_grad():
                for l, w in enumerate(self.W):
                    if self.backward_hook_objects[l] is not None:
                    # Use same fallback gradient for both drop and grow
                      self.backward_hook_objects[l].dense_grad_drop = w.grad.detach().clone()
                      self.backward_hook_objects[l].dense_grad_grow = w.grad.detach().clone()
                      #added because it is possible that this is the exact cause for 0 loss, apart from clipless

        self._rigl_step()
        self.rigl_steps += 1
        return True  # Adam and its LR schedule also advance on topology updates.

      return True

    @torch.no_grad()
    def _rigl_step(self):
        drop_fraction = self.cosine_annealing()


        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:
                continue

            current_mask = self.backward_masks[l]

            # calculate raw scores
            #MODIFIED DROP TO INCLUDE self.backward_hook_objects[l].dense_grad
            #this one is with L=L1 + L2
            #score_drop = torch.abs(w * self.backward_hook_objects[l].dense_grad)
            #score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)
            #The formula is alpha*|dL1/dW| + beta*|dL1/dW| , but there must be a custom training loop in order to get the gradients separately, commonly refered as sequential backwards
            #The growth formula will be focused on bringing the physics loss down
            score_drop = torch.abs(w * self.backward_hook_objects[l].dense_grad_drop)
            score_grow = torch.abs(self.backward_hook_objects[l].dense_grad_grow)

            # Drop active edges; grow edges inactive BEFORE this update.
            old_mask = current_mask.flatten().clone()
            n_prune = min(int(int(old_mask.sum()) * drop_fraction),
                          int((~old_mask).sum()))
            if n_prune == 0:
                continue
            drop_scores = score_drop.flatten().masked_fill(~old_mask, float('inf'))
            grow_scores = score_grow.flatten().masked_fill(old_mask, -float('inf'))
            drop_indices = torch.topk(drop_scores, n_prune, largest=False).indices
            grow_indices = torch.topk(grow_scores, n_prune).indices
            new_mask = old_mask.clone()
            new_mask[drop_indices] = False
            new_mask[grow_indices] = True
            w.flatten()[grow_indices] = 0
            current_mask.copy_(new_mask.reshape_as(current_mask))
            for value in self.optimizer.state.get(w, {}).values():
                if torch.is_tensor(value) and value.shape == w.shape:
                    value.flatten()[drop_indices] = 0
                    value.flatten()[grow_indices] = 0
            self.changed_connections += n_prune

        self.reset_momentum()
        self.apply_mask_to_weights()

    def due(self):
        return (not self.static_topo and (self.step + 1) % self.delta_T == 0
                and self.step + 1 < self.T_end)

    @torch.no_grad()
    def update(self, drop_grad, grow_grad):
        # Bridge to the current training loop; use the original score holders.
        for hook, drop, grow in zip(self.backward_hook_objects, drop_grad, grow_grad):
            if hook is not None:
                hook.dense_grad_drop = drop
                hook.dense_grad_grow = grow
        self._rigl_step()
        self.rigl_steps += 1


def angle_coefficients(angle):
    theta = float(angle) % 180.0
    a, b = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    return (0. if abs(a) < 1e-12 else a, 0. if abs(b) < 1e-12 else b)
