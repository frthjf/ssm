import autograd.numpy as np
import autograd.numpy.random as npr

from ssm.primitives import lds_log_probability, lds_sample, lds_mean
from ssm.messages import hmm_expected_states, hmm_sample, kalman_info_sample, kalman_info_smoother, \
    hmm_expected_states_3d, kalman_info_smoother_3d, hmm_sample_3d, kalman_info_sample_3d

from ssm.util import ensure_variational_args_are_lists, trace_product

from autograd.scipy.special import logsumexp
from warnings import warn

import torch


class VariationalPosterior(object):
    """
    Base class for a variational posterior distribution.

        q(z; phi) \approx p(z | x, theta)

    where z is a latent variable and x is the observed data.

    ## Reparameterization Gradients
    We assume that the variational posterior is "reparameterizable"
    in the sense that,

    z ~ q(z; phi)  =d  eps ~ r(eps); z = f(eps; phi).

    where =d denotes equal in distirbution.  If this is the case,
    we can rewrite

    L(phi) = E_q(z; phi) [g(z)] = E_r(eps) [g(f(eps; phi))]

    and

    dL/dphi = E_r(eps) [d/dphi g(f(eps; phi))]
            approx 1/S sum_s [d/dphi g(f(eps_s; phi))]

    where eps_s ~iid r(eps).  In practice, this Monte Carlo estimate
    of dL/dphi is lower variance than alternative approaches like
    the score function estimator.

    ## Amortization
    We also allow for "amortized variational inference," in which the
    variational posterior parameters are a function of the data.  We
    write the posterior as

        q(z; x, phi) approx p(z | x, theta).


    ## Requirements
    A variational posterior must support sampling and point-wise
    evaluation in order to be used for the reparameterization trick.
    """
    # @ensure_variational_args_are_lists
    def __init__(self, model, datas, inputs=None, masks=None, tags=None):
        """
        Initialize the posterior with a ref to the model and datas,
        where datas is a list of data arrays.
        """
        self.model = model
        self.datas = datas

    @property
    def params(self):
        """
        Return phi.
        """
        raise NotImplemented

    def sample(self):
        """
        Return a sample from q(z; x, phi)
        """
        raise NotImplemented

    def log_density(self, sample):
        """
        Return log q(z; x, phi)
        """
        raise NotImplemented


class SLDSMeanFieldVariationalPosterior(VariationalPosterior):
    """
    Mean field variational posterior for the continuous latent
    states of an SLDS.
    """
    @ensure_variational_args_are_lists
    def __init__(self, model, datas,
                 inputs=None, masks=None, tags=None,
                 initial_variance=0.01):

        super(SLDSMeanFieldVariationalPosterior, self).\
            __init__(model, datas, masks, tags)

        # Initialize the parameters
        self.D = model.D
        self.Ts = [data.shape[0] for data in datas]
        self.initial_variance = initial_variance
        self._params = [self._initialize_variational_params(data, input, mask, tag)
                        for data, input, mask, tag in zip(datas, inputs, masks, tags)]

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        assert len(value) == len(self.datas)
        for v, T in zip(value, self.Ts):
            assert len(v) == 2
            q_mu, q_sigma_inv = v
            assert q_mu.shape == q_sigma_inv.shape == (T, self.D)

        self._params = value

    @property
    def mean(self):
        return [mu for mu, _ in self.params]

    def _initialize_variational_params(self, data, input, mask, tag):
        T = data.shape[0]
        q_mu = self.model.emissions.invert(data, input=input, mask=mask, tag=tag)
        q_sigma_inv = np.log(self.initial_variance) * np.ones((T, self.D))
        return q_mu, q_sigma_inv

    def sample(self):
        return [q_mu + np.sqrt(np.exp(q_sigma_inv)) * npr.randn(*q_mu.shape)
                for (q_mu, q_sigma_inv) in self.params]

    def log_density(self, sample):
        assert isinstance(sample, list) and len(sample) == len(self.datas)

        logq = 0
        for s, (q_mu, q_sigma_inv) in zip(sample, self.params):
            assert s.shape == q_mu.shape
            q_sigma = np.exp(q_sigma_inv)
            logq += np.sum(-0.5 * np.log(2 * np.pi * q_sigma))
            logq += np.sum(-0.5 * (s - q_mu)**2 / q_sigma)

        return logq


class SLDSTriDiagVariationalPosterior(VariationalPosterior):
    """
    Gaussian variational posterior for the continuous latent
    states of an SLDS.  The Gaussian is constrained to have
    a block tri-diagonal inverse covariance matrix, as in a
    linear dynamical system.
    """
    @ensure_variational_args_are_lists
    def __init__(self, model, datas,
                 inputs=None, masks=None, tags=None,
                 initial_variance=0.01):

        super(SLDSTriDiagVariationalPosterior, self).\
            __init__(model, datas, masks, tags)

        # Initialize the parameters
        self.D = model.D
        self.Ts = [data.shape[0] for data in datas]
        self.initial_variance = initial_variance
        self._params = [self._initialize_variational_params(data, input, mask, tag)
                        for data, input, mask, tag in zip(datas, inputs, masks, tags)]

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        D = self.D

        # Check the value for correct shape
        assert len(value) == len(self.datas)
        for v, T in zip(value, self.Ts):
            As, bs, Qi_sqrts, ms, Ri_sqrts = v
            assert As.shape == (T-1, D, D)
            assert bs.shape == (T-1, D)
            assert Qi_sqrts.shape == (T-1, D, D)
            assert ms.shape == (T, D)
            assert Ri_sqrts.shape == (T, D, D)

        self._params = value

    @property
    def mean(self):
        return [lds_mean(*prms) for prms in self.params]

    def _initialize_variational_params(self, data, input, mask, tag):
        T = data.shape[0]
        D = self.D

        # Initialize the mean with the linear model, if applicable
        ms = self.model.emissions.invert(data, input=input, mask=mask, tag=tag)

        # Initialize with no covariance between adjacent time steps
        # NOTE: it's important to initialize A and Q to be nonzero,
        # otherwise the gradients wrt them are zero and they never
        # change during optimization!
        As = np.repeat(np.eye(D)[None, :, :], T-1, axis=0)
        bs = np.zeros((T-1, D))
        Qi_sqrts = np.repeat(np.eye(D)[None, :, :], T-1, axis=0)
        Ri_sqrts = 1./np.sqrt(self.initial_variance) * np.repeat(np.eye(D)[None, :, :], T, axis=0)
        return As, bs, Qi_sqrts, ms, Ri_sqrts

    def sample(self):
        return [lds_sample(*prms) for prms in self.params]

    def log_density(self, sample):
        assert isinstance(sample, list) and len(sample) == len(self.datas)

        logq = 0
        for s, prms in zip(sample, self.params):
            logq += lds_log_probability(s, *prms)
        return logq


class SLDSStructuredMeanFieldVariationalPosterior(VariationalPosterior):
    """
    p(z, x | y) \approx q(z) q(x).


    Assume q(z) is a chain-structured discrete graphical model,

        q(z) = exp{log_pi0[z_1] +
                   \sum_{t=2}^T log_Ps[z_{t-1}, z_t] +
                   \sum_{t=1}^T log_likes[z_t]

    parameterized by pi0, Ps, and log_likes.

    Assume q(x) is a Gaussian with a block tridiagonal precision matrix,
    and that we update q(x) via Laplace approximation. Specifically,

        q(x) = N(J, h)

    where J is block tridiagonal precision and h is the linear potential.
    The mapping to mean parameters is mu = J^{-1} h and Sigma = J^{-1}.

    Initial distribution parameters:
    J_ini:     (D, D)       initial state precision
    h_ini:     (D,)         initial state bias

    If time-varying dynamics:
    J_dyn_11:  (T-1, D, D)  upper left block of dynamics precision
    J_dyn_21:  (T-1, D, D)  lower left block of dynamics precision
    J_dyn_22:  (T-1, D, D)  lower right block of dynamics precision
    h_dyn_1:   (T-1, D)     upper block of dynamics bias
    h_dyn_2:   (T-1, D)     lower block of dynamics bias

    Observation distribution parameters
    J_obs:     (T, D, D)    observation precision
    h_obs:     (T, D)       observation bias
    """
    # @ensure_variational_args_are_lists
    def __init__(self, model, datas,
                 inputs=None, masks=None, tags=None,
                 initial_variance=0.01):

        super(SLDSStructuredMeanFieldVariationalPosterior, self).\
            __init__(model, datas, masks, tags)

        # Initialize the parameters
        self.D = model.D
        self.K = model.K
        self.Ts = [data.shape[0] for data in datas]
        self.initial_variance = initial_variance

        self._discrete_state_params = None
        self._discrete_expectations = None
        # self.discrete_state_params = \
        #     [self._initialize_discrete_state_params(data, input, mask, tag)
        #      for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        self.discrete_state_params = self._initialize_discrete_state_params_3d(datas, inputs, masks, tags)

        self._continuous_state_params = None
        self._continuous_expectations = None
        # self.continuous_state_params = \
        #     [self._initialize_continuous_state_params(data, input, mask, tag)
        #      for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        self.continuous_state_params = self._initialize_continuous_state_params_3d(datas, inputs, masks, tags)

    # Parameters
    @property
    def params(self):
        return self.discrete_state_params, self.continuous_state_params

    @property
    def discrete_state_params(self):
        return self._discrete_state_params

    @discrete_state_params.setter
    def discrete_state_params(self, value):
        # assert isinstance(value, list) and len(value) == len(self.datas)
        # for prms in value:
        #     for key in ["pi0", "Ps", "log_likes"]:
        #         assert key in prms
        assert isinstance(value, dict)
        for key in ['pi0', 'Ps', 'log_likes']:
            assert key in value
        self._discrete_state_params = value

        # Rerun the HMM smoother with the updated parameters
        # self._discrete_expectations = \
        #     [hmm_expected_states(prms["pi0"], prms["Ps"], prms["log_likes"])
        #      for prms in self._discrete_state_params]
            
        # pi0 = torch.tensor(np.stack([prms['pi0'] for prms in self._discrete_state_params]))
        # Ps = torch.tensor(np.stack([prms['Ps'] for prms in self._discrete_state_params]))
        # log_likes = torch.tensor(np.stack([prms['log_likes'] for prms in self._discrete_state_params]))
        self._discrete_expectations = hmm_expected_states_3d(
            self._discrete_state_params['pi0'], 
            self._discrete_state_params['Ps'],
            self._discrete_state_params['log_likes']
        )

        # for i in range(len(self._discrete_expectations)):
        #     assert np.all(np.abs(Ezs[i].cpu().numpy() - self._discrete_expectations[i][0]) < 1e-8)
        #     assert np.all(np.abs(Ezzp1s[i].cpu().numpy() - self._discrete_expectations[i][1]) < 1e-8)
        #     assert np.all(np.abs(norm[i].cpu().numpy() - self._discrete_expectations[i][2]) < 1e-8)

    @property
    def continuous_state_params(self):
        return self._continuous_state_params

    @continuous_state_params.setter
    def continuous_state_params(self, value):
        # assert isinstance(value, list) and len(value) == len(self.datas)
        # for prms in value:
        #     for key in ["J_ini", "J_dyn_11", "J_dyn_21", "J_dyn_22", "J_obs",
        #                 "h_ini", "h_dyn_1", "h_dyn_2", "h_obs"]:
        #         assert key in prms
        assert isinstance(value, dict)
        for key in ["J_ini", "J_dyn_11", "J_dyn_21", "J_dyn_22", "J_obs",
                    "h_ini", "h_dyn_1", "h_dyn_2", "h_obs"]:
            assert key in value
        self._continuous_state_params = value

        # Rerun the Kalman smoother with the updated parameters
        # self._continuous_expectations = \
        #     [kalman_info_smoother(prms["J_ini"], prms["h_ini"], 0,
        #                           prms["J_dyn_11"], prms["J_dyn_21"], prms["J_dyn_22"],
        #                           prms["h_dyn_1"], prms["h_dyn_2"], 0,
        #                           prms["J_obs"], prms["h_obs"], 0)
        #      for prms in self._continuous_state_params]
            
        # kwargs = {}
        # for key in ["J_ini", "J_dyn_11", "J_dyn_21", "J_dyn_22", "J_obs",
        #             "h_ini", "h_dyn_1", "h_dyn_2", "h_obs"]:
        #     kwargs[key] = torch.tensor(np.stack([prms[key] for prms in self._continuous_state_params]))
        log_Z_ini = torch.zeros((self.datas.shape[0], 1), dtype=torch.double)
        log_Z_dyn = torch.zeros((self.datas.shape[0], 1), dtype=torch.double)
        log_Z_obs = torch.zeros((self.datas.shape[0], 1), dtype=torch.double)
        self._continuous_expectations = \
            kalman_info_smoother_3d(log_Z_ini=log_Z_ini, log_Z_dyn=log_Z_dyn, log_Z_obs=log_Z_obs, **self._continuous_state_params)

        # for i in range(len(self._continuous_expectations)):
        #     assert np.all(np.abs(log_Z[i,0].cpu().numpy() - self._continuous_expectations[i][0]) < 1e-8)
        #     assert np.all(np.abs(smoothed_mus[i].cpu().numpy() - self._continuous_expectations[i][1]) < 1e-8)
        #     assert np.all(np.abs(smoothed_Sigmas[i].cpu().numpy() - self._continuous_expectations[i][2]) < 1e-8)
        #     assert np.all(np.abs(ExxnT[i].cpu().numpy() - self._continuous_expectations[i][3]) < 1e-8)

    def _initialize_discrete_state_params(self, data, input, mask, tag):
        T = data.shape[0]
        K = self.K

        # Initialize q(z) parameters: pi0, log_likes, transition_matrices
        pi0 = np.ones(K) / K
        Ps = np.ones((T-1, K, K)) / K
        log_likes = np.zeros((T, K))
        return dict(pi0=pi0, Ps=Ps, log_likes=log_likes)
    
    def _initialize_discrete_state_params_3d(self, datas, inputs, masks, tags):
        B, T, _ = datas.shape
        K = self.K

        pi0 = torch.ones((B,K), dtype=torch.double) / K
        Ps = torch.ones((B,T-1,K,K), dtype=torch.double) / K
        log_likes = torch.zeros((B,T,K), dtype=torch.double)
        return dict(pi0=pi0, Ps=Ps, log_likes=log_likes)

    def _initialize_continuous_state_params(self, data, input, mask, tag):
        T = data.shape[0]
        D = self.D

        # Initialize the linear terms
        h_ini = np.zeros(D)
        h_dyn_1 = np.zeros((T - 1, D))
        h_dyn_2 = np.zeros((T - 1, D))

        # Set the posterior mean based on the emission model, if possible.
        # try:
        h_obs = (1.0 / self.initial_variance) * self.model.emissions. \
                invert(data, input=input, mask=mask, tag=tag).cpu().numpy()
        # except:
        #     warn("We can only initialize the continuous states if the emissions support "
        #          "\"inverting\" the observations by mapping them to an estimate of the "
        #          "latent states. Defaulting to a random initialization instead.")
        #     h_obs = (1.0 / self.initial_variance) * np.random.randn(data.shape[0], self.D)

        # Initialize the posterior variance to self.initial_variance * I
        J_ini = np.zeros((D, D))
        J_dyn_11 = np.zeros((T - 1, D, D))
        J_dyn_21 = np.zeros((T - 1, D, D))
        J_dyn_22 = np.zeros((T - 1, D, D))
        J_obs = np.tile(1 / self.initial_variance * np.eye(D)[None, :, :], (T, 1, 1))

        return dict(J_ini=J_ini,
                    h_ini=h_ini,
                    J_dyn_11=J_dyn_11,
                    J_dyn_21=J_dyn_21,
                    J_dyn_22=J_dyn_22,
                    h_dyn_1=h_dyn_1,
                    h_dyn_2=h_dyn_2,
                    J_obs=J_obs,
                    h_obs=h_obs)
    
    def _initialize_continuous_state_params_3d(self, datas, inputs, masks, tags):
        B, T, _ = datas.shape
        D = self.D

        h_ini = torch.zeros((B,D), dtype=torch.double)
        h_dyn_1 = torch.zeros((B,T-1,D), dtype=torch.double)
        h_dyn_2 = torch.zeros((B,T-1,D), dtype=torch.double)
    
        h_obs = (1.0 / self.initial_variance) * self.model.emissions. \
            invert(datas, input=inputs, mask=masks, tag=tags)

        J_ini = torch.zeros((B, D, D), dtype=torch.double)
        J_dyn_11 = torch.zeros((B, T - 1, D, D), dtype=torch.double)
        J_dyn_21 = torch.zeros((B, T - 1, D, D), dtype=torch.double)
        J_dyn_22 = torch.zeros((B, T - 1, D, D), dtype=torch.double)
        J_obs = torch.tensor(np.tile(1 / self.initial_variance * np.eye(D)[None, None, :, :], (B, T, 1, 1)), dtype=torch.double)

        return dict(J_ini=J_ini,
                    h_ini=h_ini,
                    J_dyn_11=J_dyn_11,
                    J_dyn_21=J_dyn_21,
                    J_dyn_22=J_dyn_22,
                    h_dyn_1=h_dyn_1,
                    h_dyn_2=h_dyn_2,
                    J_obs=J_obs,
                    h_obs=h_obs)      

    # Posterior expectations
    @property
    def discrete_expectations(self):
        return self._discrete_expectations

    @property
    def continuous_expectations(self):
        return self._continuous_expectations

    @property
    def mean_discrete_states(self):
        full_expectations = self.discrete_expectations
        return full_expectations[0]
        # return [exp[0] for exp in full_expectations]

    @property
    def mean_continuous_states(self):
        full_expectations = self.continuous_expectations
        return full_expectations[1]
        # return [exp[1] for exp in full_expectations]

    @property
    def mean(self):
        return list(zip(self.discrete_expectations, self.mean_continuous_states))

    # Sample
    def sample_discrete_states(self):
        prms = self._discrete_state_params
        return hmm_sample_3d(prms['pi0'], prms['Ps'], prms['log_likes'])
        # return [hmm_sample(prms["pi0"], prms["Ps"], prms["log_likes"])
        #         for prms in self._discrete_state_params]

    def sample_continuous_states(self):
        prms = self._continuous_state_params
        return kalman_info_sample_3d(prms["J_ini"], prms["h_ini"], torch.zeros((self.datas.shape[0], 1), dtype=torch.double),
                                   prms["J_dyn_11"], prms["J_dyn_21"], prms["J_dyn_22"],
                                   prms["h_dyn_1"], prms["h_dyn_2"], torch.zeros((self.datas.shape[0], 1), dtype=torch.double),
                                   prms["J_obs"], prms["h_obs"], torch.zeros((self.datas.shape[0], 1), dtype=torch.double))
        # return [kalman_info_sample(prms["J_ini"], prms["h_ini"], 0,
        #                            prms["J_dyn_11"], prms["J_dyn_21"], prms["J_dyn_22"],
        #                            prms["h_dyn_1"], prms["h_dyn_2"], 0,
        #                            prms["J_obs"], prms["h_obs"], 0)
        #         for prms in self._continuous_state_params]

    def sample(self):
        return list(zip(self.sample_discrete_states(), self.sample_continuous_states()))

    # Entropy
    def _discrete_entropy(self):
        negentropy = 0
        # discrete_expectations = self.discrete_expectations
        full_prms = self.discrete_state_params
        Ezs, Ezzp1s, normalizers = self.discrete_expectations
        # for prms, (Ez, Ezzp1, normalizer) in \
        #         zip(self.discrete_state_params, discrete_expectations):
        for i in range(Ezs.shape[0]):
            Ez = Ezs[i].cpu().numpy()
            Ezzp1 = Ezzp1s[i].cpu().numpy()
            normalizer = normalizers[i].cpu().numpy()
            prms = {key: val[i].cpu().numpy() for key, val in full_prms.items()}

            log_pi0 = np.log(prms["pi0"] + 1e-16) - logsumexp(prms["pi0"])
            log_Ps = np.log(prms["Ps"] + 1e-16) - logsumexp(prms["Ps"], axis=1, keepdims=True)
            negentropy -= normalizer  # -log Z
            negentropy += np.sum(Ez[0] * log_pi0)  # initial factor
            negentropy += np.sum(Ez * prms["log_likes"])  # unitary factors
            negentropy += np.sum(Ezzp1 * log_Ps)  # pairwise factors
        return -negentropy
    
    def _discrete_entropy_3d(self):
        negentropy = 0
        prms = self.discrete_state_params
        Ezs, Ezzp1s, normalizers = self.discrete_expectations
        log_pi0 = torch.log(prms['pi0'] + 1e-16) - torch.logsumexp(prms['pi0'], dim=1, keepdim=True)
        log_Ps = torch.log(prms['Ps'] + 1e-16) - torch.logsumexp(prms['Ps'], dim=2, keepdim=True)
        negentropy -= torch.sum(normalizers)
        negentropy += torch.sum(Ezs[:,0] * log_pi0)
        negentropy += torch.sum(Ezs * prms['log_likes'])
        negentropy += torch.sum(Ezzp1s * log_Ps)
        # for i in range(Ez.shape[0]):
        #     log_pi0 = torch.log(prms['pi0'][i] + 1e-16) - torch.logsumexp(prms['pi0'][i])
        #     log_Ps = torch.log(prms['Ps'][i] + 1e-16) - torch.logsumexp(prms['Ps'][i], dim=1, keepdim=True)
        #     negentropy -= normalizer[i]
        #     negentropy += torch.sum(Ezs[i][0] * log_pi0)
        #     negentropy += torch.sum(Ezs[i] * prms['log_likes'][i])
        #     negentropy += torch.sum(Ezzp1s[i] * log_Ps)
        return -negentropy

    def _continuous_entropy(self):
        negentropy = 0
        # continuous_expectations = self.continuous_expectations
        full_prms = self.continuous_state_params
        log_Zs, Exs, smoothed_sigmass, ExxnTs = self.continuous_expectations
        # for prms, (log_Z, Ex, smoothed_sigmas, ExxnT) in \
        #         zip(self.continuous_state_params, continuous_expectations):
        for i in range(log_Zs.shape[0]):
            # import pdb; pdb.set_trace()
            Ex = Exs[i].cpu().numpy()
            log_Z = log_Zs[i].cpu().numpy()[0]
            smoothed_sigmas = smoothed_sigmass[i].cpu().numpy()
            ExxnT = ExxnTs[i].cpu().numpy()
            prms = {key: val[i].cpu().numpy() for key, val in full_prms.items()}

            # Kalman smoother outputs the smoothed covariance matrices. Add
            # back the mean to get E[x_t x_{t+1}^T]
            mumuT = np.swapaxes(Ex[:, None], 2,1) @ Ex[:, None]
            ExxT = smoothed_sigmas + mumuT

            # Pairwise terms
            negentropy += np.sum(-0.5 * trace_product(prms["J_ini"], ExxT[0]))
            negentropy += np.sum(-0.5 * trace_product(prms["J_dyn_11"], ExxT[:-1]))
            negentropy += np.sum(-0.5 * trace_product(prms["J_dyn_22"], ExxT[1:]))
            negentropy += np.sum(-0.5 * trace_product(prms["J_obs"], ExxT))
            negentropy += np.sum(-1.0 * trace_product(prms["J_dyn_21"], ExxnT))

            # Unary terms
            negentropy += np.sum(prms["h_ini"] * Ex[0])
            negentropy += np.sum(prms["h_dyn_1"] * Ex[:-1])
            negentropy += np.sum(prms["h_dyn_2"] * Ex[1:])
            negentropy += np.sum(prms["h_obs"] * Ex)

            # Log normalizer
            negentropy -= log_Z
        return -negentropy
    
    def _continuous_entropy_3d(self):
        negentropy = 0
        prms = self.continuous_state_params
        log_Zs, Exs, smoothed_sigmas, ExxnTs = self.continuous_expectations
        mumuT = Exs[:,:,:,None] @ Exs[:,:,None,:]
        ExxT = smoothed_sigmas + mumuT

        negentropy += torch.sum(-0.5 * torch.sum(prms['J_ini'] * ExxT[:,0].permute(0,2,1), dim=(1,2)))
        negentropy += torch.sum(-0.5 * torch.sum(prms['J_dyn_11'] * ExxT[:,:-1].permute(0,1,3,2), dim=(2,3)))
        negentropy += torch.sum(-0.5 * torch.sum(prms['J_dyn_22'] * ExxT[:,1:].permute(0,1,3,2), dim=(2,3)))
        negentropy += torch.sum(-0.5 * torch.sum(prms['J_obs'] * ExxT.permute(0,1,3,2), dim=(2,3)))
        negentropy += torch.sum(-1.0 * torch.sum(prms['J_dyn_21'] * ExxnTs.permute(0,1,3,2), dim=(2,3)))

        negentropy += torch.sum(prms['h_ini'] * Exs[:,0])
        negentropy += torch.sum(prms['h_dyn_1'] * Exs[:,:-1])
        negentropy += torch.sum(prms['h_dyn_2'] * Exs[:,1:])
        negentropy += torch.sum(prms['h_obs'] * Exs)

        negentropy -= torch.sum(log_Zs)

        return -negentropy
        # for i in range(log_Z.shape[0]):
        #     mumuT = Exs[i][:, None].permute(0, 2, 1) @ Exs[i][:, None]
        #     ExxT = smoothed_sigmas[i] + mumuT

        #     # Pairwise terms
        #     negentropy += np.sum(-0.5 * trace_product(prms["J_ini"], ExxT[0]))
        #     negentropy += np.sum(-0.5 * trace_product(prms["J_dyn_11"], ExxT[:-1]))
        #     negentropy += np.sum(-0.5 * trace_product(prms["J_dyn_22"], ExxT[1:]))
        #     negentropy += np.sum(-0.5 * trace_product(prms["J_obs"], ExxT))
        #     negentropy += np.sum(-1.0 * trace_product(prms["J_dyn_21"], ExxnT))

        #     # Unary terms
        #     negentropy += np.sum(prms["h_ini"] * Ex[0])
        #     negentropy += np.sum(prms["h_dyn_1"] * Ex[:-1])
        #     negentropy += np.sum(prms["h_dyn_2"] * Ex[1:])
        #     negentropy += np.sum(prms["h_obs"] * Ex)

        #     # Log normalizer
        #     negentropy -= log_Z

    def entropy(self):
        """
        Compute the entropy of the variational posterior distirbution.

        Recall that under the structured mean field approximation

        H[q(z)q(x)] = -E_{q(z)q(x)}[log q(z) + log q(x)] = -E_q(z)[log q(z)] -
                    E_q(x)[log q(x)] = H[q(z)] + H[q(x)].

        That is, the entropy separates into the sum of entropies for the
        discrete and continuous states.

        For each one, we have

        E_q(u)[log q(u)] = E_q(u) [log q(u_1) + sum_t log q(u_t | u_{t-1}) + loq
                         q(u_t) - log Z] = E_q(u_1)[log q(u_1)] + sum_t
                         E_{q(u_t, u_{t-1}[log q(u_t | u_{t-1})] + E_q(u_t)[loq
                         q(u_t)] - log Z

        where u \in {z, x} and log Z is the log normalizer.  This shows that we
        just need the posterior expectations and potentials, and the log
        normalizer of the distribution.

        """
        # continuous_entropy = self._continuous_entropy()
        # discrete_entropy = self._discrete_entropy()
        continuous_entropy = self._continuous_entropy_3d()
        discrete_entropy = self._discrete_entropy_3d()
        # import pdb; pdb.set_trace()
        return discrete_entropy + continuous_entropy
