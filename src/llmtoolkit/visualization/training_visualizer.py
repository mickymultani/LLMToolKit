import matplotlib.pyplot as plt

class TrainingVisualizer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def update(self, train_loss, val_loss, epoch):
        """
        Updates the records with new data.

        Args:
        - train_loss (float): The training loss for the current epoch.
        - val_loss (float): The validation loss for the current epoch.
        - epoch (int): The current epoch number.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.epochs.append(epoch)

    def plot(self):
        """
        Plots the training and validation losses over epochs.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()