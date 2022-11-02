from pytorch_lightning import LightningModule


class BaseTrainService(LightningModule):
    """
    Interface for PyTorch Lightning Module
    """

    def forward(self):
        pass

    def configure_optimizers(self):
        pass

    def shared_step(self, x, y):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        pass
