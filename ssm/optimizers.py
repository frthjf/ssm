"""
Single step of a variety of optimization routines.
Modified from autograd.misc.optimizers.

The function being optimized must take two arguments,
an input value and an iteration number.
"""
from functools import partial
from warnings import warn

from autograd import grad, value_and_grad
import autograd.numpy as np
from autograd.misc import flatten
from autograd.wrap_util import wraps

from scipy.optimize import minimize
from ssm.primitives import solve_symm_block_tridiag

import torch
import numpy

def convex_combination(curr, target, alpha):
    """
    Output next = (1-alpha) * target + alpha * curr
    where target, curr, and next can be trees of nested
    containers with arrays/scalars at the leaves.
    Assume curr and target have the same structure.
    """
    assert alpha >= 0 and alpha <= 1
    _curr, unflatten = flatten(curr)
    _target, _ = flatten(target)
    return unflatten(alpha * _curr + (1-alpha) * _target)

def convex_combination_torch(curr, target, alpha):
    assert alpha >= 0 and alpha <= 1
    return [(c * alpha + (1-alpha) * t) for c, t in zip(curr, target)]

def unflatten_optimizer_step(step):
    """
    Wrap an optimizer step function that operates on flat 1D arrays
    with a version that handles trees of nested containers,
    i.e. (lists/tuples/dicts), with arrays/scalars at the leaves.
    """
    @wraps(step)
    def _step(value_and_grad, x, itr, state=None, *args, **kwargs):
        _x, unflatten = flatten(x)
        def _value_and_grad(x, i):
            v, g = value_and_grad(unflatten(x), i)
            return v, flatten(g)[0]
        _next_x, _next_val, _next_g, _next_state = \
            step(_value_and_grad, _x, itr, state=state, *args, **kwargs)
        return unflatten(_next_x), _next_val, _next_g, _next_state
    return _step


@unflatten_optimizer_step
def sgd_step(value_and_grad, x, itr, state=None, step_size=0.1, mass=0.9):
    # Stochastic gradient descent with momentum.
    velocity = state if state is not None else np.zeros(len(x))
    val, g = value_and_grad(x, itr)
    velocity = mass * velocity - (1.0 - mass) * g
    x = x + step_size * velocity
    return x, val, g, velocity

@unflatten_optimizer_step
def rmsprop_step(value_and_grad, x, itr, state=None, step_size=0.1, gamma=0.9, eps=10**-8):
    # Root mean squared prop: See Adagrad paper for details.
    avg_sq_grad = np.ones(len(x)) if state is None else state
    val, g = value_and_grad(x, itr)
    avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
    x = x - (step_size * g) / (np.sqrt(avg_sq_grad) + eps)
    return x, val, g, avg_sq_grad


@unflatten_optimizer_step
def adam_step(value_and_grad, x, itr, state=None, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """
    Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms.
    """
    m, v = (np.zeros(len(x)), np.zeros(len(x))) if state is None else state
    val, g = value_and_grad(x, itr)
    m = (1 - b1) * g      + b1 * m    # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v    # Second moment estimate.
    mhat = m / (1 - b1**(itr + 1))    # Bias correction.
    vhat = v / (1 - b2**(itr + 1))
    x = x - (step_size * mhat) / (np.sqrt(vhat) + eps)
    return x, val, g, (m, v)


def _generic_sgd(method, loss, x0, callback=None, num_iters=200, state=None, full_output=False, **kwargs):
    """
    Generic stochastic gradient descent step.
    """
    step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[method]

    # Initialize outputs
    x, losses, grads = x0, [], []
    for itr in range(num_iters):
        x, val, g, state = step(value_and_grad(loss), x, itr, state, **kwargs)
        losses.append(val)
        grads.append(g)

    if full_output:
        return x, state
    else:
        return x


def _generic_minimize(method, loss, x0,
                      verbose=False,
                      num_iters=1000,
                      tol=1e-4,
                      state=None,
                      full_output=False,
                      suppress_warnings=False,
                      **kwargs):
    """
    Minimize a given loss function with scipy.optimize.minimize.
    """
    # Flatten the loss
    _x0, unflatten = flatten(x0)
    _objective = lambda x_flat, itr: loss(unflatten(x_flat), itr)

    if verbose:
        print("Fitting with {}.".format(method))

    # Specify callback for fitting
    itr = [0]
    def callback(x_flat):
        itr[0] += 1
        print("Iteration {} loss: {:.3f}".format(itr[0], loss(unflatten(x_flat), -1)))

    # Wrap the gradient to avoid NaNs
    def safe_grad(x, itr):
        # import pdb; pdb.set_trace()
        g = grad(_objective)(x, itr)
        g[~np.isfinite(g)] = 1e8
        return g

    # Call the optimizer.  Pass in -1 as the iteration since it is unused.
    result = minimize(_objective, _x0, args=(-1,),
                      jac=safe_grad,
                      method=method,
                      callback=callback if verbose else None,
                      options=dict(maxiter=num_iters, disp=verbose),
                      tol=tol,
                      **kwargs)
    if verbose:
        print("{} completed with message: \n{}".format(method, result.message))

    if not suppress_warnings and not result.success:
        warn("{} failed with message:\n{}".format(method, result.message))

    # print(result.x[:20], result.x[-20:])
    
    if full_output:
        return unflatten(result.x), result
    else:
        return unflatten(result.x)

def _generic_minimize_torch(method, loss, x0,
                      verbose=False,
                      num_iters=1000,
                      tol=1e-4,
                      state=None,
                      full_output=False,
                      suppress_warnings=False,
                      **kwargs):
    _x0 = torch.cat([torch.flatten(arr) for arr in x0]).cpu().numpy()
    splits = [torch.numel(arr) for arr in x0]
    shapes = [arr.shape for arr in x0]
    def unflatten(arr):
        if isinstance(arr, numpy.ndarray):
            arr = torch.tensor(arr)
        return [a.reshape(shape) for a, shape in zip(torch.split(arr, splits), shapes)]

    _objective = lambda x_flat, itr: loss(unflatten(x_flat), itr)

    def safe_grad(x, itr):
        # import pdb; pdb.set_trace()
        if isinstance(x, numpy.ndarray):
            x = torch.tensor(x)
        x.requires_grad_(True)
        out = _objective(x, itr)
        g = torch.autograd.grad([out], x)[0]
        g[~torch.isfinite(g)] = 1e8
        return g.cpu().numpy()

    result = minimize(_objective, _x0, args=(-1,),
                      jac=safe_grad,
                      method=method,
                      callback=None,
                      options=dict(maxiter=num_iters, disp=verbose),
                      tol=tol,
                      **kwargs)

    if not suppress_warnings and not result.success:
        warn("{} failed with message:\n{}".format(method, result.message))

    # print(result.x[:20], result.x[-20:])

    if full_output:
        return unflatten(result.x), result
    else:
        return unflatten(result.x)

# Define optimizers
sgd = partial(_generic_sgd, "sgd")
rmsprop = partial(_generic_sgd, "rmsprop")
adam = partial(_generic_sgd, "adam")
bfgs = partial(_generic_minimize, "BFGS")
lbfgs = partial(_generic_minimize_torch, "L-BFGS-B")


# Special optimizer for function with block-tridiagonal hessian
def newtons_method_block_tridiag_hessian(
    x0, obj, grad_func, hess_func,
    tolerance=1e-4, maxiter=100):
    """
    Newton's method to minimize a positive definite function with a
    block tridiagonal Hessian matrix.
    Algorithm 9.5, Boyd & Vandenberghe, 2004.
    """
    # import pdb; pdb.set_trace()

    x = x0
    is_converged = False
    count = 0
    while not is_converged:
        H_diag, H_lower_diag = hess_func(x)
        g = grad_func(x)
        dx = -1.0 * solve_symm_block_tridiag(H_diag, H_lower_diag, g)
        lambdasq = np.dot(g.ravel(), -1.0*dx.ravel())
        if lambdasq / 2.0 <= tolerance:
            is_converged = True
            break
        stepsize = backtracking_line_search(x, dx, obj, g)
        x = x + stepsize * dx
        count += 1
        if count > maxiter:
            break

    if not is_converged:
        warn("Newton's method failed to converge in {} iterations. "
             "Final mean abs(dx): {}".format(maxiter, np.mean(np.abs(dx))))

    return x


def backtracking_line_search(x0, dx, obj, g, stepsize = 1.0, min_stepsize=1e-8,
                             alpha=0.2, beta=0.7):
    """
    A backtracking line search for the step size in Newton's method.
    Algorithm 9.2, Boyd & Vandenberghe, 2004.
    - dx is the descent direction
    - g is the gradient evaluated at x0
    - alpha in (0,0.5) is fraction of decrease in objective predicted  by
        a linear extrapolation that we will accept
    - beta in (0,1) is step size reduction factor
    """
    x = x0

    # criterion: stop when f(x + stepsize * dx) < f(x) + \alpha * stepsize * f'(x)^T dx
    f_term = obj(x)
    grad_term = alpha * np.dot(g.ravel(), dx.ravel())

    # decrease stepsize until criterion is met
    # or stop at minimum step size
    while stepsize > min_stepsize:
        fx = obj(x+ stepsize*dx)
        if np.isnan(fx) or fx > f_term + grad_term*stepsize:
            stepsize *= beta
        else:
            break

    return stepsize

def newtons_method_block_tridiag_hessian_3d(
    x0s, obj, grad_func, hess_func,
    tolerance=1e-4, maxiter=100):

    xs = x0s
    is_converged = torch.zeros((xs.shape[0],), dtype=torch.bool, device=x0s.device)
    count = 0
    while not torch.all(is_converged):
        H_diags, H_lower_diags = hess_func(xs[~is_converged], ~is_converged)
        gs = grad_func(xs[~is_converged].clone().detach().requires_grad_(True), ~is_converged)
        dxs = -1.0 * torch.tensor([solve_symm_block_tridiag(H_diag, H_lower_diag, g) \
            for H_diag, H_lower_diag, g in zip(H_diags.cpu().numpy(), H_lower_diags.cpu().numpy(), gs.cpu().numpy())], device=x0s.device)
        lambdasq = torch.sum(gs * dxs * -1.0, axis=(1,2))
        is_converged[~is_converged] = (lambdasq / 2.0 <= tolerance)
        if torch.all(is_converged):
            break
        stepsize = backtracking_line_search_3d(xs[~is_converged], dxs[lambdasq / 2.0 > tolerance], obj, gs[lambdasq / 2.0 > tolerance], is_converged)
        xs[~is_converged] = xs[~is_converged] + stepsize[:, None, None] * dxs[lambdasq / 2.0 > tolerance]

        count += 1
        if count > maxiter:
            break
    
    if not torch.all(is_converged):
        warn("Newton's method failed to converge in {} iterations for {} trials."
             "Final mean abs(dx): {}".format(maxiter, torch.sum(~is_converged), torch.mean(torch.abs(dxs))))
    
    return xs

def backtracking_line_search_3d(x0s, dxs, obj, gs, is_converged, stepsize=1.0, min_stepsize=1e-8, alpha=0.2, beta=0.7):
    xs = x0s
    f_term = obj(xs, ~is_converged)
    grad_term = alpha * torch.sum(gs * dxs, axis=(1,2))
    stepsize = torch.full((gs.shape[0],), stepsize, device=x0s.device)

    # xs = xs[~is_converged]
    # f_term = f_term[~is_converged]
    # grad_term = grad_term[~is_converged]
    # stepsize = stepsize[~is_converged]
    # dxs = dxs[~is_converged]

    while not torch.all(stepsize <= min_stepsize):
        fxs = obj(xs + stepsize[:, None, None] * dxs, ~is_converged)
        mask = (stepsize > min_stepsize) & (torch.isnan(fxs) | (fxs > (f_term + grad_term * stepsize)))
        if torch.any(mask):
            stepsize = torch.where(mask, stepsize * beta, stepsize)
        else:
            break
    
    return stepsize