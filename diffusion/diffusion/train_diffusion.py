from abc import abstractmethod
from .base_diffusion import BaseDiffusion

class TrainDiffusion(BaseDiffusion):

    @abstractmethod
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        """
        pass
    
    @abstractmethod
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).
        """
        pass

    @abstractmethod
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        pass

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        pass

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        pass
