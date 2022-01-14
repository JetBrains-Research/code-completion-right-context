import torch
from catalyst.callbacks import Callback
from catalyst.dl import IRunner


class SaveCheckpointCallback(Callback):
    def __init__(self, n_epochs, log_dir):
        self.n_epochs = n_epochs
        self.counter = 0
        self.log_dir = log_dir

    # override
    def on_batch_end(self, state: IRunner):
        # only in the infer stage
        if not state.is_infer_stage:
            return

        self.counter += 1

        if self.counter % self.n_epochs == 0:
            model_state_dict = state.model.state_dict()
            with open(f'{self.log_dir}/{self.counter}', 'wb') as f:
                torch.save(model_state_dict, f)

