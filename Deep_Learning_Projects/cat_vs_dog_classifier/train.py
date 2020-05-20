import os
import matplotlib.pyplot as plt

from main import CatVsDog

def accuracy_graph(history):
    plot_dir = "accuracy_plot"
    plt.title('Classification Acuraccy')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.1, 1])
    plt.legend(loc='lower right')
    
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plt.savefig(os.path.join(plot_dir, "model_accuracy.png"))

def loss_graph(history):
    plot_dir = "loss_plot"
    plt.title("Cross Entropy Loss")
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0.1, 1])
    plt.legend(loc='lower right')

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plt.savefig(os.path.join(plot_dir, "model_loss.png"))


if __name__ == "__main__":
    main = CatVsDog()
    main.data_visualization()
    history = main.train()
    accuracy_graph(history)
    loss_graph(history)
