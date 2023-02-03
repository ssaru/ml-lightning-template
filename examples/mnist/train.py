from container import MNISTModelContainer
from model import MnistModel

from ml_training_template.application import TrainApplication

if __name__ == "__main__":
    train_app = TrainApplication()
    train_app.run()
