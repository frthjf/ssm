from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd import hessian

from ssm.util import ensure_args_are_lists, \
    logistic, logit, softplus, inv_softplus
from ssm.preprocessing import interpolate_data, interpolate_data_3d, pca_with_imputation
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
from ssm.stats import independent_studentst_logpdf, bernoulli_logpdf
from ssm.regression import fit_linear_regression

import torch

# Observation models for SLDS
class Emissions(object):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        self.N, self.K, self.D, self.M, self.single_subspace = \
            N, K, D, M, single_subspace

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        pass

    def initialize_from_arhmm(self, arhmm, pca):
        pass

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag, x):
        raise NotImplementedError

    def forward(self, x, input=None, tag=None):
        raise NotImplementedError

    def invert(self, data, input=None, mask=None, tag=None):
        raise NotImplementedError

    def sample(self, z, x, input=None, tag=None):
        raise NotImplementedError

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")
        warn("Analytical Hessian is not implemented for this Emissions class. \
              Optimization via Laplace-EM may be slow. Consider using an \
              alternative posterior and inference method.")
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        obj = lambda xt, datat, inputt, maskt: \
            self.log_likelihoods(datat[None,:], inputt[None,:], maskt[None,:], tag, xt[None,:])[0, 0]
        hess = hessian(obj)
        terms = np.array([np.squeeze(hess(xt, datat, inputt, maskt))
                          for xt, datat, inputt, maskt in zip(x, data, input, mask)])
        return -1 * terms

    def m_step(self, discrete_expectations, continuous_expectations,
               datas, inputs, masks, tags,
               optimizer="bfgs", maxiter=100, **kwargs):
        """
        If M-step in Laplace-EM cannot be done in closed form for the emissions, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        def _get_obj(discrete_expectations, continuous_expectations):
            obj = self.log_prior()
            lls = self.log_likelihoods_3d(datas, inputs, masks, tags, continuous_expectations)#, m_step=True)
            # obj += np.sum(discrete_expectations[0].cpu().numpy() * lls)
            obj += torch.sum(discrete_expectations[0] * lls)
            return obj

        # expected log likelihood
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            # obj = 0
            # obj += self.log_prior()
            # for data, input, mask, tag, x, Ez in \
            #     zip(datas, inputs, masks, tags, continuous_expectations, discrete_expectations[0]):
            #     obj += np.sum(Ez.numpy() * self.log_likelihoods(data.numpy().astype(int), input.numpy(), mask.numpy(), tag, x.numpy()))
            # lls = self.log_likelihoods_3d(datas, inputs, masks, tags, continuous_expectations, m_step=False).cpu().numpy()
            # obj += np.sum(discrete_expectations[0].cpu().numpy() * lls)
            # assert np.abs(obj - obj2) < 1e-6, "fail"
            obj = _get_obj(discrete_expectations, continuous_expectations)
            return -obj / T

        # Optimize emissions log-likelihood
        self.params = optimizer(_objective, self.params,
                                num_iters=maxiter,
                                suppress_warnings=True,
                                **kwargs)


# Many emissions models start with a linear layer
class _LinearEmissions(Emissions):
    """
    A simple linear mapping from continuous states x to data y.

        E[y | x] = Cx + d + Fu

    where C is an emission matrix, d is a bias, F an input matrix,
    and u is an input.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_LinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer.  Set _Cs to be private so that it can be
        # changed in subclasses.
        self._Cs = torch.tensor(npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D))
        self.Fs = torch.tensor(npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M))
        self.ds = torch.tensor(npr.randn(1, N) if single_subspace else npr.randn(K, N))
        # self._Cs = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        # self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
        # self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def Cs(self):
        return self._Cs

    @Cs.setter
    def Cs(self, value):
        K, N, D = self.K, self.N, self.D
        assert value.shape == (1, N, D) if self.single_subspace else (K, N, D)
        self._Cs = value

    @property
    def params(self):
        return self.Cs, self.Fs, self.ds

    @params.setter
    def params(self, value):
        self.Cs, self.Fs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self.Cs = self.Cs[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        # Invert with the average emission parameters
        assert self.M == 0, "inputs not supported"
        # C = torch.mean(torch.tensor(self.Cs, device=data.device), axis=0)
        C = torch.mean(self.Cs, axis=0)
        # F = torch.mean(torch.tensor(self.Fs, device=data.device), axis=0)
        # d = torch.mean(torch.tensor(self.ds, device=data.device), axis=0)
        d = torch.mean(self.ds, axis=0)
        # C_pseudoinv = torch.solve(C.T, C.T @ C)[0].T
        C_pseudoinv = torch.linalg.solve(C.T @ C, C.T).T

        # Account for the bias
        # bias = (input @ F.T) + d
        bias = d

        if not torch.all(mask):
            if len(data.shape) == 3:
                data = interpolate_data_3d(data, mask)
                temp_mask = mask[:, 0:1, :].repeat(1, mask.shape[1], 1)
                for itr in range(25):
                    mu = (data - bias) @ (C_pseudoinv)
                    data[~temp_mask] = (mu @ (C.T) + bias)[~temp_mask]
            elif len(data.shape) == 2:
                data = interpolate_data(data, mask)
                for itr in range(25):
                    mu = (data - bias) @ (C_pseudoinv)
                    data[:, ~mask[0]] = (mu @ (C.T) + bias)[:, ~mask[0]]
            else:
                raise NotImplementedError
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            # for itr in range(25):
            #     mu = (data - bias) @ (C_pseudoinv)
            #     data[:, :, ~mask[0]] = (mu @ (C.T) + bias)[:, :, ~mask[0]]

        # Project data to get the mean
        return (data - bias) @ (C_pseudoinv)

    def forward(self, x, input, tag):
        Cs = self.Cs.cpu().numpy() if isinstance(self.Cs, torch.Tensor) else self.Cs
        Fs = self.Fs.cpu().numpy() if isinstance(self.Fs, torch.Tensor) else self.Fs
        ds = self.ds.cpu().numpy() if isinstance(self.ds, torch.Tensor) else self.ds
        return np.matmul(Cs[None, ...], x[:, None, :, None])[:, :, :, 0] \
            + np.matmul(Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
            + ds

    def forward_3d(self, xs, inputs, tags, m_step=False):
        B = len(xs)
        # Cs (1, N, D) -> (1, 1, N, D)
        # x (T, D) -> (T, 1, D, 1)
        # Cs @ x -> (T, 1, N, 1) -> (T, 1, N)

        # Fs (1, N, M) -> (1, 1, N, M)
        # input (T, M) -> (T, 1, M, 1)
        # Fs @ input -> (T, 1, N, 1) -> (T, 1, N)
        
        # ds (1, N)
        assert self.M == 0, "inputs not supported"
        if m_step:
            xs = xs[:, :, None, :, None].detach().cpu().numpy()
            return (self.Cs @ xs)[:, :, :, :, 0] + self.ds
        elif isinstance(self.Cs, np.numpy_boxes.ArrayBox):
            Cs = torch.tensor(self.Cs._value[None, None, ...], device=xs.device)
            ds = torch.tensor(self.ds._value, device=xs.device)
            xs = xs[:, :, None, :, None]
            return (Cs @ xs)[:, :, :, :, 0] + ds
        else:
            # Cs = torch.tensor(self.Cs[None, None, ...], device=xs.device)
            Cs = self.Cs[None, None, ...]
            # ds = torch.tensor(self.ds, device=xs.device)
            ds = self.ds
            xs = xs[:, :, None, :, None]
            return (Cs @ xs)[:, :, :, :, 0] + ds

    # @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        assert self.M == 0, "inputs not supported"
        Keff = 1 if self.single_subspace else self.K

        # First solve a linear regression for data given input
        if self.M > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack(inputs), np.vstack(datas))
            self.Fs = np.tile(lr.coef_[None, :, :], (Keff, 1, 1))

        # Compute residual after accounting for input
        # resids = [data - np.dot(input, self.Fs[0].T) for data, input in zip(datas, inputs)]
        resids = datas

        # Run PCA to get a linear embedding of the data with the maximum effective dimension
        pca, xs, ll = pca_with_imputation(min(self.D * Keff, self.N),
                                          resids, masks, num_iters=num_iters)

        # Assign each state a random projection of these dimensions
        Cs, ds = [], []
        for k in range(Keff):
            weights = npr.randn(self.D, self.D * Keff)
            weights = np.linalg.svd(weights, full_matrices=False)[2]
            Cs.append((weights @ pca.components_).T)
            ds.append(pca.mean_)

        # Find the components with the largest power
        self.Cs = torch.tensor(np.array(Cs))
        self.ds = torch.tensor(np.array(ds))
        # self.Cs = np.array(Cs)
        # self.ds = np.array(ds)

        return pca


class _OrthogonalLinearEmissions(_LinearEmissions):
    """
    A linear emissions matrix constrained such that the emissions matrix
    is orthogonal. Use the rational Cayley transform to parameterize
    the set of orthogonal emission matrices. See
    https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.5b02015
    for a derivation of the rational Cayley transform.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_OrthogonalLinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer
        assert N > D
        self._Ms = npr.randn(1, D, D) if single_subspace else npr.randn(K, D, D)
        self._As = npr.randn(1, N-D, D) if single_subspace else npr.randn(K, N-D, D)
        self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)

        # Set the emission matrix to be a random orthogonal matrix
        C0 = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        for k in range(C0.shape[0]):
            C0[k] = np.linalg.svd(C0[k], full_matrices=False)[0]
        self.Cs = C0

    @property
    def Cs(self):
        D = self.D
        T = lambda X: np.swapaxes(X, -1, -2)

        Bs = 0.5 * (self._Ms - T(self._Ms))    # Bs is skew symmetric
        Fs = np.matmul(T(self._As), self._As) - Bs
        trm1 = np.concatenate((np.eye(D) - Fs, 2 * self._As), axis=1)
        trm2 = np.eye(D) + Fs
        Cs = T(np.linalg.solve(T(trm2), T(trm1)))
        assert np.allclose(
            np.matmul(T(Cs), Cs),
            np.tile(np.eye(D)[None, :, :], (Cs.shape[0], 1, 1))
            )
        return Cs

    @Cs.setter
    def Cs(self, value):
        N, D = self.N, self.D
        T = lambda X: np.swapaxes(X, -1, -2)

        # Make sure value is the right shape and orthogonal
        Keff = 1 if self.single_subspace else self.K
        assert value.shape == (Keff, N, D)
        assert np.allclose(
            np.matmul(T(value), value),
            np.tile(np.eye(D)[None, :, :], (Keff, 1, 1))
            )

        Q1s, Q2s = value[:, :D, :], value[:, D:, :]
        Fs = T(np.linalg.solve(T(np.eye(D) + Q1s), T(np.eye(D) - Q1s)))
        # Bs = 0.5 * (T(Fs) - Fs) = 0.5 * (self._Ms - T(self._Ms)) -> _Ms = T(Fs)
        self._Ms = T(Fs)
        self._As = 0.5 * np.matmul(Q2s, np.eye(D) + Fs)
        assert np.allclose(self.Cs, value)

    @property
    def params(self):
        return self._As, self._Ms, self.Fs, self.ds

    @params.setter
    def params(self, value):
        self._As, self._Ms, self.Fs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self._As = self._As[perm]
            self._Ms = self._Ms[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]


# Sometimes we just want a bit of additive noise on the observations
class _IdentityEmissions(Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_IdentityEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)
        assert N == D

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass

    def forward(self, x, input, tag):
        return x[:, None, :]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Inverse is just the data
        """
        return np.copy(data)


# Allow general nonlinear emission models with neural networks
class _NeuralNetworkEmissions(Emissions):
    def __init__(self, N, K, D, M=0, hidden_layer_sizes=(50,), single_subspace=True):
        assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
        super(_NeuralNetworkEmissions, self).__init__(N, K, D, M=M, single_subspace=True)

        # Initialize the neural network weights
        assert N > D
        layer_sizes = (D + M,) + hidden_layer_sizes + (N,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

    @property
    def params(self):
        return self.weights, self.biases

    @params.setter
    def params(self, value):
        self.weights, self.biases = value

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        inputs = np.column_stack((x, input))
        for W, b in zip(self.weights, self.biases):
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs[:, None, :]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Inverse is... who knows!
        """
        return npr.randn(data.shape[0], self.D)


# Observation models for SLDS
class _GaussianEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_GaussianEmissionsMixin, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -1 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return super(_GaussianEmissionsMixin, self).params + (self.inv_etas,)

    @params.setter
    def params(self, value):
        self.inv_etas = value[-1]
        super(_GaussianEmissionsMixin, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(_GaussianEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        return mus[np.arange(T), z, :] + np.sqrt(etas[z]) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        yhat = mus[:, 0, :] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)
        return yhat


class GaussianEmissions(_GaussianEmissionsMixin, _LinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        if self.single_subspace:
            block = -1.0 * self.Cs[0].T@np.diag( 1.0 / np.exp(self.inv_etas[0]) )@self.Cs[0]
            hess = np.tile(block[None,:,:], (T, 1, 1))
        else:
            blocks = np.array([-1.0 * C.T@np.diag(1.0/np.exp(inv_eta))@C
                               for C, inv_eta in zip(self.Cs, self.inv_etas)])
            hess = np.sum(Ez[:,:,None,None] * blocks, axis=1)
        return -1 * hess

    def m_step(self, discrete_expectations, continuous_expectations,
               datas, inputs, masks, tags,
               optimizer="bfgs", maxiter=100, **kwargs):

        Xs = [np.column_stack([x, u]) for x, u in
              zip(continuous_expectations, inputs)]
        ys = datas
        ws = [Ez for (Ez, _, _) in discrete_expectations]

        if self.single_subspace and all([np.all(mask) for mask in masks]):
            # Return exact m-step updates for C, F, d, and inv_etas
            CF, d, Sigma = fit_linear_regression(
                Xs, ys,
                prior_ExxT=1e-4 * np.eye(self.D + self.M + 1),
                prior_ExyT=np.zeros((self.D + self.M + 1, self.N)))
            self.Cs = CF[None, :, :self.D]
            self.Fs = CF[None, :, self.D:]
            self.ds = d[None, :]
            self.inv_etas = np.log(np.diag(Sigma))[None, :]
        else:
            Cs, Fs, ds, inv_etas = [], [], [], []
            for k in range(self.K):
                CF, d, Sigma = fit_linear_regression(
                    Xs, ys, weights=[w[:, k] for w in ws],
                    prior_ExxT=1e-4 * np.eye(self.D + self.M + 1),
                    prior_ExyT=np.zeros((self.D + self.M + 1, self.N)))
                Cs.append(CF[:, :self.D])
                Fs.append(CF[:, self.D:])
                ds.append(d)
                inv_etas.append(np.log(np.diag(Sigma)))
            self.Cs = np.array(Cs)
            self.Fs = np.array(Fs)
            self.ds = np.array(ds)
            self.inv_etas = np.array(inv_etas)


class GaussianOrthogonalEmissions(_GaussianEmissionsMixin, _OrthogonalLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        if self.single_subspace:
            block = -1.0 * self.Cs[0].T@np.diag( 1.0 / np.exp(self.inv_etas[0]) )@self.Cs[0]
            hess = np.tile(block[None,:,:], (T, 1, 1))
        else:
            blocks = np.array([-1.0 * C.T@np.diag(1.0/np.exp(inv_eta))@C
                               for C, inv_eta in zip(self.Cs, self.inv_etas)])
            hess = np.sum(Ez[:,:,None,None] * blocks, axis=1)
        return -1 * hess


class GaussianIdentityEmissions(_GaussianEmissionsMixin, _IdentityEmissions):

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        if self.single_subspace:
            block = -1.0 * np.diag(1.0 / np.exp(self.inv_etas[0]))
            hess = np.tile(block[None, :, :], (T, 1, 1))
        else:
            raise NotImplementedError

        return -1 * hess


class GaussianNeuralNetworkEmissions(_GaussianEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _StudentsTEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_StudentsTEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_nus = np.log(4) * np.ones((1, N)) if single_subspace else np.log(4) * np.ones(K, N)

    @property
    def params(self):
        return super(_StudentsTEmissionsMixin, self).params + (self.inv_etas, self.inv_nus)

    @params.setter
    def params(self, value):
        super(_StudentsTEmissionsMixin, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(_StudentsTEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]
            self.inv_nus = self.inv_nus[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        etas, nus = np.exp(self.inv_etas), np.exp(self.inv_nus)
        mus = self.forward(x, input, tag)
        return independent_studentst_logpdf(data[:, None, :],
                                            mus, etas, nus, mask=mask[:, None, :])

    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        nus = np.exp(self.inv_nus)
        etas = np.exp(self.inv_etas)
        taus = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        return mus[np.arange(T), z, :] + np.sqrt(etas[z] / taus) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)


class StudentsTEmissions(_StudentsTEmissionsMixin, _LinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class StudentsTOrthogonalEmissions(_StudentsTEmissionsMixin, _OrthogonalLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class StudentsTIdentityEmissions(_StudentsTEmissionsMixin, _IdentityEmissions):
    pass


class StudentsTNeuralNetworkEmissions(_StudentsTEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _BernoulliEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="logit", **kwargs):
        super(_BernoulliEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        self.link_name = link
        mean_functions = dict(
            logit=logistic
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            logit=logit
            )
        self.link = link_functions[link]

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == bool or (data.dtype == int and data.min() >= 0 and data.max() <= 1)
        assert self.link_name == "logit", "Log likelihood is only implemented for logit link."
        logit_ps = self.forward(x, input, tag)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return bernoulli_logpdf(data[:, None, :], logit_ps, mask=mask[:, None, :])

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, .9))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        ps = self.mean(self.forward(x, input, tag))
        return (npr.rand(T, self.N) < ps[np.arange(T), z,:]).astype(int)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        ps = self.mean(self.forward(variational_mean, input, tag))
        return ps[:,0,:] if self.single_subspace else np.sum(ps * expected_states[:,:,None], axis=1)


class BernoulliEmissions(_BernoulliEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx  (y - p) * C
            = -dpsi/dx (dp/d\psi)  C
            = -C p (1-p) C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")
        assert self.link_name == "logit"
        psi =  self.forward(x, input, tag)[:, 0, :]
        p = self.mean(psi)
        dp_dpsi = p * (1 - p)
        hess = np.einsum('tn, ni, nj ->tij', -dp_dpsi, self.Cs[0], self.Cs[0])
        return -1 * hess


class BernoulliOrthogonalEmissions(_BernoulliEmissionsMixin, _OrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx  (y - p) * C
            = -dpsi/dx (dp/d\psi)  C
            = -C p (1-p) C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")
        assert self.link_name == "logit"
        psi =  self.forward(x, input, tag)[:, 0, :]
        p = self.mean(psi)
        dp_dpsi = p * (1 - p)
        hess = np.einsum('tn, ni, nj ->tij', -dp_dpsi, self.Cs[0], self.Cs[0])
        return -1 * hess


class BernoulliIdentityEmissions(_BernoulliEmissionsMixin, _IdentityEmissions):
    pass


class BernoulliNeuralNetworkEmissions(_BernoulliEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _PoissonEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=1.0, **kwargs):

        super(_PoissonEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        self.link_name = link
        self.bin_size = bin_size
        mean_functions = dict(
            log=self._log_mean,
            softplus=self._softplus_mean
            )
        self.mean = mean_functions[link]
        link_functions = dict(
            log=self._log_link,
            softplus=self._softplus_link
            )
        self.link = link_functions[link]

        # Set the bias to be small if using log link
        if link == "log":
            self.ds = torch.tensor(-3 + .5 * npr.randn(1, N) if single_subspace else npr.randn(K, N))
            # self.ds = -3 + .5 * npr.randn(1, N) if single_subspace else npr.randn(K, N)

    def _log_mean(self, x):
        if isinstance(x, torch.Tensor):
            return torch.exp(x) * self.bin_size
        else:
            return np.exp(x) * self.bin_size

    def _softplus_mean(self, x):
        if isinstance(x, torch.Tensor):
            return torch.log1p(torch.exp(x)) * self.bin_size
        else:
            return softplus(x) * self.bin_size

    def _log_link(self, rate):
        if isinstance(rate, torch.Tensor):
            return torch.log(rate) - np.log(self.bin_size)
        else:
            return np.log(rate) - np.log(self.bin_size)

    def _softplus_link(self, rate):
        if isinstance(rate, torch.Tensor):
            return torch.log(torch.exp(rate / self.bin_size) - 1)
        else:
            return inv_softplus(rate / self.bin_size)

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == int
        lambdas = self.mean(self.forward(x, input, tag))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas)
        return np.sum(lls * mask[:, None, :], axis=2)

    def log_likelihoods_3d(self, datas, inputs, masks, tags, xs, m_step=False):
        assert datas.dtype == torch.int
        if m_step:
            lambdas = self.mean(self.forward_3d(xs, inputs, tags, m_step=m_step))
            datas = datas.cpu().numpy()
            masks = torch.ones(datas.shape, dtype=bool) if masks is None else masks.cpu().numpy()
            lls = -gammaln(datas[:,:,None,:] + 1) -lambdas + datas[:,:,None,:] * np.log(lambdas)
            return np.sum(lls * masks[:, :, None, :], axis=3)
        else:
            lambdas = self.mean(self.forward_3d(xs, inputs, tags, m_step=m_step))
            masks = torch.ones_like(datas, dtype=torch.bool) if masks is None else masks
            lls = -torch.tensor(gammaln(datas[:,:,None,:].cpu().numpy() + 1), device=datas.device) -lambdas + datas[:,:,None,:] * torch.log(lambdas)
            return torch.sum(lls * masks[:, :, None, :], axis=3)

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(torch.tensor(np.clip(data.cpu().numpy(), .1, np.inf), device=data.device))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        lambdas = self.mean(self.forward(x, input, tag))
        y = npr.poisson(lambdas[np.arange(T), z, :])
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        lambdas = self.mean(self.forward(variational_mean, input, tag))
        return lambdas[:,0,:] if self.single_subspace else np.sum(lambdas * expected_states[:,:,None], axis=1)
    
    def smooth_3d(self, expected_states, variational_mean, datas, inputs, masks, tags):
        lambdas = self.mean(self.forward_3d(variational_mean, inputs, tags))
        return lambdas[:,:,0,:] if self.single_subspace else torch.sum(lambdas * expected_states[:,:,:,None], dim=2)

class PoissonEmissions(_PoissonEmissionsMixin, _LinearEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(PoissonEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)
        # Scale down the measurement and control matrices so that
        # rate params don't explode when exponentiated.
        if self.link_name == "log":
            # self.Cs /= np.exp(np.linalg.norm(self.Cs, axis=2)[:,:,None])
            # self.Fs /= np.exp(np.linalg.norm(self.Fs, axis=2)[:,:,None])
            self.Cs /= torch.exp(torch.norm(self.Cs, dim=2)[:,:,None])
            self.Fs /= torch.exp(torch.norm(self.Fs, dim=2)[:,:,None])

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        # yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        # self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

        # datastack = np.stack(datas)
        # if masks is not None:
            # maskstack = np.stack(masks)
        datas = interpolate_data_3d(datas, masks)
        yhats = self.link(torch.tensor(np.clip(datas.cpu().numpy(), .1, np.inf), device=datas.device))
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)
        

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
                          = y * C - lmbda * C
                          = (y - lmbda) * C

        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
            = -C^T exp(Cx + Fu + d)^T C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")

        if self.link_name == "log":
            lambdas = self.mean(self.forward(x, input, tag))
            hess = np.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], self.Cs[0], self.Cs[0])
            return -1 * hess

        elif self.link_name == "softplus":
            # For stability, we avoid evaluating terms that look like exp(x)**2.
            # Instead, we rearrange things so that all terms with exp(x)**2 are of the form
            # (exp(x) / exp(x)**2) which evaluates to sigmoid(x)sigmoid(-x) and avoids overflow.
            lambdas = self.mean(self.forward(x, input, tag))[:, 0, :] / self.bin_size
            linear_terms = -np.dot(x,self.Cs[0].T)-np.dot(input,self.Fs[0].T)-self.ds[0]
            expterms = np.exp(linear_terms)
            outer = logistic(linear_terms) * logistic(-linear_terms)
            diags = outer * (data / lambdas - data / (lambdas**2 * expterms) - self.bin_size)
            hess = np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])
            return -hess

        else:
            raise Exception("No Hessian calculation for link: {}".format(self.link_name))
    
    def neg_hessian_log_emissions_prob_3d(self, datas, inputs, masks, tags, xs, Ezs):
        assert self.single_subspace, "single subspace required"
        # if self.link_name != 'log':
        #     raise NotImplementedError
        
        lambdas = self.mean(self.forward_3d(xs, inputs, tags))
        C = self.Cs[0]
        # C = torch.tensor(self.Cs[0], device=datas.device)
        hess = torch.einsum('btn,ni,nj->btij', -lambdas[:, :, 0, :], C, C)
        return -1 * hess

class PoissonOrthogonalEmissions(_PoissonEmissionsMixin, _OrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
                          = y * C - lmbda * C
                          = (y - lmbda) * C

        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
            = -C^T exp(Cx + Fu + d)^T C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")

        if self.link_name == "log":
            lambdas = self.mean(self.forward(x, input, tag))
            hess = np.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], self.Cs[0], self.Cs[0])
            return -1 * hess

        elif self.link_name == "softplus":
            # For stability, we avoid evaluating terms that look like exp(x)**2.
            # See comment in PoissoinEmissions.
            lambdas = self.mean(self.forward(x, input, tag))[:, 0, :] / self.bin_size
            linear_terms = -np.dot(x,self.Cs[0].T)-np.dot(input,self.Fs[0].T)-self.ds[0]
            expterms = np.exp(linear_terms)
            outer = logistic(linear_terms) * logistic(-linear_terms)
            diags = outer * (data / lambdas - data / (lambdas**2 * expterms) - self.bin_size)
            hess = np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])
            return -hess

        else:
            raise Exception("No Hessian calculation for link: {}".format(self.link_name))

class PoissonIdentityEmissions(_PoissonEmissionsMixin, _IdentityEmissions):
    pass


class PoissonNeuralNetworkEmissions(_PoissonEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _AutoRegressiveEmissionsMixin(object):
    """
    Include past observations as a covariate in the SLDS emissions.
    The AR model is restricted to be diagonal.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_AutoRegressiveEmissionsMixin, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)

        # Initialize AR component of the model
        self.As = npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

        # Shrink the eigenvalues of the A matrices to avoid instability.
        # Since the As are diagonal, this is just a clip.
        self.As = np.clip(self.As, -1.0 + 1e-8, 1 - 1e-8)

    @property
    def params(self):
        return super(_AutoRegressiveEmissionsMixin, self).params + (self.As, self.inv_etas)

    @params.setter
    def params(self, value):
        self.As, self.inv_etas = value[-2:]
        super(_AutoRegressiveEmissionsMixin, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(_AutoRegressiveEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.As = self.inv_nus[perm]
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.forward(x, input, tag)
        pad = np.zeros((1, 1, self.N)) if self.single_subspace else np.zeros((1, self.K, self.N))
        mus = mus + np.concatenate((pad, self.As[None, :, :] * data[:-1, None, :]))

        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        assert self.single_subspace, "Can only invert with a single emission model"
        pad = np.zeros((1, self.N))
        resid = data - np.concatenate((pad, self.As * data[:-1]))
        return self._invert(resid, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T, N = z.shape[0], self.N
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)

        y = np.zeros((T, N))
        y[0] = mus[0, z[0], :] + np.sqrt(etas[z[0]]) * npr.randn(N)
        for t in range(1, T):
            y[t] = mus[t, z[t], :] + self.As[z[t]] * y[t-1] + np.sqrt(etas[z[0]]) * npr.randn(N)
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        mus[1:] += self.As[None, :, :] * data[:-1, None, :]
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states, axis=1)


class AutoRegressiveEmissions(_AutoRegressiveEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        # Initialize the subspace with PCA
        from sklearn.decomposition import PCA
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]

        # Solve a linear regression for the AR coefficients.
        from sklearn.linear_model import LinearRegression
        for n in range(self.N):
            lr = LinearRegression()
            lr.fit(np.concatenate([d[:-1, n] for d in datas])[:,None],
                   np.concatenate([d[1:, n] for d in datas]))
            self.As[:,n] = lr.coef_[0]

        # Compute the residuals of the AR model
        pad = np.zeros((1,self.N))
        mus = [np.concatenate((pad, self.As[0] * d[:-1])) for d in datas]
        residuals = [data - mu for data, mu in zip(datas, mus)]

        # Run PCA on the residuals to initialize C and d
        pca = self._initialize_with_pca(residuals, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class AutoRegressiveOrthogonalEmissions(_AutoRegressiveEmissionsMixin, _OrthogonalLinearEmissions):
    pass


class AutoRegressiveIdentityEmissions(_AutoRegressiveEmissionsMixin, _IdentityEmissions):
    pass


class AutoRegressiveNeuralNetworkEmissions(_AutoRegressiveEmissionsMixin, _NeuralNetworkEmissions):
    pass
