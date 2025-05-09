from .faa_model_base import Faa_model_base

class faa_fold_model(Faa_model_base):
    def __init__(self, config):
        self.config = config

    def transform_antenna_positions(self, antenna_positions_original, psi):
        pass

    def directional_channel_gain(self, antenna_positions_rotated):
        pass

    def omni_channel_gain(self, antenna_positions_rotated):
        pass
