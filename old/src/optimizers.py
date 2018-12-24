from collections import deque, namedtuple
import random
import torch
from torch.optim import Optimizer
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch.nn.utils import clip_grad_norm


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        #if lr is not required and lr < 0.0:
        #    raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None, scaling_factor=1, grad_clip=-1):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'] / float(scaling_factor), d_p)

            # Add gradient clipping if necessary
            if grad_clip != -1:
                clip_grad_norm(group['params'], grad_clip)

        return loss


class NoisySGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr, noise_factor=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(NoisySGD, self).__init__(params, defaults)
        self.noise_factor = noise_factor

    def __setstate__(self, state):
        super(NoisySGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
		# Add spherical noise here!
                noise = torch.normal(means=torch.zeros_like(p), std=self.noise_factor)
                p.data.add_(-group['lr'], d_p + noise.data)

        return loss

class HessianVecSGD(Optimizer):
    def __init__(self, params, lr, noise_factor=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, cg_damping=0.001):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(HessianVecSGD, self).__init__(params, defaults)
        self.noise_factor = noise_factor
        self.cg_damping = cg_damping

    def __setstate__(self, state):
        super(NoisySGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # HESSIAN VEC COMPUTATION
            # vectorize all parameters
            grad_vec = parameters_to_vector(group['params'])
            # create noise vector
            noise = torch.normal(means=torch.zeros_like(grad_vec), std=self.noise_factor)
            # compute the product
            grad_product = torch.sum(grad_vec * noise)
            grad_grad = torch.autograd.grad(
                grad_product, group['params'], retain_graph=True
            )
            # h_v_p = hessian_vec_product
            fisher_vec_prod = torch.cat([g.contiguous().view(-1) for g in grad_grad])
            hessian_vec_prod = fisher_vec_prod + (self.cg_damping * noise)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                d_p = p.grad.clone().data

                # REST OF SGD STUFF
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
            flattened = parameters_to_vector(group['params'])
            flattened.data.add_(group['lr'], hessian_vec_prod.data)
            vector_to_parameters(flattened, group['params'])

        return loss


class ReservoirSGD(Optimizer):
    def __init__(self, params, lr, scale=0.1, num_gradients_to_sample=4, max_reservoir_size=8, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ReservoirSGD, self).__init__(params, defaults)
        self.scale = scale
        self.reservoir = [[[]]] # r[i][j] is reservoir for param group i, param j
        self.max_reservoir_size = max_reservoir_size
        self.num_gradients_to_sample = num_gradients_to_sample

    def __setstate__(self, state):
        super(ReservoirSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def sample_reservoir(self, i, j, batch_size=1):
        """
        Sample from the gradients reservoir!
        """
        return random.sample(self.reservoir[i][j], min(batch_size, len(self.reservoir)))

    def update_reservoir(self, i, j, gradients):
        """
        Put fresh gradients into the reservoir.
        Pop out the oldest ones if it's full.
        """
        num_to_pop = len(gradients) + len(self.reservoir[i][j]) - self.max_reservoir_size - 1
        if num_to_pop > 0:
            self.reservoir[i][j] = self.reservoir[i][j][num_to_pop:]
        self.reservoir[i][j] += gradients

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for i, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # reservoir bookkeeping:
            # if we haven't seen this parameter group before,
            # add new param reservoir.
            if len(self.reservoir) == i:
                self.reservoir.append([[]])

            for j, p in enumerate(group['params']):
                # reservoir bookkeeping:
                # if we haven't seen this param before, add
                # its reservoir.
                if len(self.reservoir[i]) == j:
                    self.reservoir[i].append([])

                if p.grad is None:
                    continue
                d_p = p.grad.data
                grad = p.grad.clone()

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # Sample older gradients here!
                if len(self.reservoir[i][j]) > 0:
                    old_gradients = torch.stack(self.sample_reservoir(i, j, self.num_gradients_to_sample))
                    avg_old_grad = self.scale * torch.mean(old_gradients, dim=0)
                    p.data.add_(-group['lr'], d_p + avg_old_grad.data)
                self.update_reservoir(i, j, [grad])

        return loss

