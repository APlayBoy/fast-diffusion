from .base_diffusion import BaseDiffusion

class MeasuredDim(BaseDiffusion):


    def _prior_bpd(self, x_start):
        pass

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        pass
