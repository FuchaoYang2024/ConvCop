from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np
import argparse
import logging
import os
from torch.utils.data import Subset
import torch
import torch.optim as optim

from logger import setup_logging
from utils import (
    dataset,
    models,
    test,
    train,
    utils,
    visualisation,
)


LOG_CONFIG_PATH = os.path.join(os.path.abspath("."), "logger", "logger_config.json")
LOG_DIR   = os.path.join(os.path.abspath("."), "logs")
DATA_DIR  = os.path.join(os.path.abspath('.'), "data")
IMAGE_DIR = os.path.join(os.path.abspath("."), "images")
MODEL_DIR = os.path.join(os.path.abspath("."), "checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure that all operations are deterministic for reproducibility, even on GPU (if used)
utils.set_seed(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

def main():


    # Configure logging module
    utils.mkdir(LOG_DIR)
    setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)

    logging.info(f'######## Training the {config["name"]} model ########')
    model = models.load_model(model_name=config["model"]["type"], params=config["model"]["args"])
    model.to(DEVICE)

    logging.info("Loading dataset...")

    

    # Get the datasets
    train_data, val_data, test_data = dataset.get_dataset(data_path=DATA_DIR, balanced=True)
    logging.info("Dataset loaded!")


    #选择一部分作为呀征集采样
    utils.set_seed(42)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


    #选择一部分作为呀征集采样
    val_subset_indices = np.random.choice(len(val_data), size=int(0.3 * len(train_data)), replace=False)
    test_subset_indices = np.random.choice(len(test_data), size=int(0.3 * len(train_data)), replace=False)

    val_subset = Subset(val_data, val_subset_indices)
    test_subset = Subset(test_data, test_subset_indices)
    # How many instances have we got?
    print('# instances in training set: ', len(train_data))
    print('# instances in validation set: ', len(val_data))
    print('# instances in testing set: ', len(test_data))
    print('# instances in validation set: ', len(val_subset))
    print('# instances in testing set: ', len(test_subset))
    batch_size = 128#之前是64，我感觉改成256也可以

    # Create the dataloaders - for training, validation and testing
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)




    model = CNN2ID()
    # Out loss function
    criterion = nn.CrossEntropyLoss()

    # Our optimizer
    learning_rate = 0.001
    #optimizer1 = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Epochs
    num_epochs = 5
    logging.info('start to train')
    device = 'cpu'
    history = train2(model, criterion, optimizer, train_loader, valid_loader, num_epochs, device)

    training_loss = history['train']['loss']
    training_accuracy = history['train']['accuracy']
    train_output_true = history['train']['output_true']
    train_output_pred = history['train']['output_pred']

    validation_loss = history['valid']['loss']
    validation_accuracy = history['valid']['accuracy']
    valid_output_true = history['valid']['output_true']
    valid_output_pred = history['valid']['output_pred']



    logging.info('start to Plot loss vs iterations')
    fig = plt.figure(figsize=(12, 8))
    plt.plot(training_loss, label='train - loss')
    plt.plot(validation_loss, label='validation - loss')
    plt.title("Train and Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="best")
    plt.savefig('Train and Validation Loss2.pdf')
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    plt.plot(training_accuracy, label='train - accuracy')
    plt.plot(validation_accuracy, label='validation - accuracy')
    plt.title("Train and Validation Accuracy")
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(loc="best")
    plt.savefig('Train and Validation Accuracy2.pdf')
    plt.show()

    logging.info('plot confusion matrix')
    labels = ['Benign', 'Botnet ARES', 'Brute Force', 'DoS', 'DDoS', 'PortScan', 'Web Attack']
    visualisation.plot_confusion_matrix(y_true=train_output_true,
                                    y_pred=train_output_pred,
                                    labels=labels,
                                    save=True,
                                    save_dir=IMAGE_DIR,
                                    filename="cnn2ids_train_confusion_matrix2.pdf")
    print("Training Set -- Classification Report", end="\n\n")
    print(classification_report(train_output_true, train_output_pred, target_names=labels))

    visualisation.plot_confusion_matrix(y_true=valid_output_true,
                                    y_pred=valid_output_pred,
                                    labels=labels,
                                    save=True,
                                    save_dir=IMAGE_DIR,
                                    filename="cnn2ids_valid_confusion_matrix2.pdf")
    print("Validation Set -- Classification Report", end="\n\n")
    print(classification_report(valid_output_true, valid_output_pred, target_names=labels))

    logging.info('test it')
    history = test(model, criterion, test_loader, device)

    test_output_true = history['test']['output_true']
    test_output_pred = history['test']['output_pred']
    test_output_pred_prob = history['test']['output_pred_prob']

    visualisation.plot_confusion_matrix(y_true=test_output_true,
                                    y_pred=test_output_pred,
                                    labels=labels,
                                    save=True,
                                    save_dir=IMAGE_DIR,
                                    filename="cnn2ids_test_confusion_matrix2.pdf")
    print("Testing Set -- Classification Report", end="\n\n")
    print(classification_report(test_output_true, test_output_pred, target_names=labels))
    
    logging.info('plot ROC')
    visualisation.plot_roc_curve(y_test=y_test,
                             y_score=y_score,
                             labels=labels,
                             save=True,
                             save_dir=IMAGE_DIR,
                             filename="cnn2ids_roc_curve2.pdf")
    visualisation.plot_roc_curve(y_test=y_test,
                             y_score=y_score,
                             labels=labels,
                             save=True,
                             save_dir=IMAGE_DIR,
                             filename="cnn2ids_roc_curve2.pdf")


if __name__ == "__main__":
    main()
