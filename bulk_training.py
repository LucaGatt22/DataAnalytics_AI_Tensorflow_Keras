# Assuming numpy archive with X and y is saved already, this Python file is independent of the Jupyter notebook `DataAnalyticsAI_Tensorflow_Keras.ipynb`
import os
from os.path import exists as path_exists
from os import makedirs
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split # to split the dataset into training (80%) and testing (20%) sets

from random import seed as random_seed

# imports for AI model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.random import set_seed as set_tensorflow_seed
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from itertools import permutations

# Load the data from the .npz (numpy archive) file
numpy_archive = np.load("Xy.npz")

# Extract X and y
X = numpy_archive['X']
y = numpy_archive['y']

'''
Stage 2 – Creation of the MLP using Keras , and
Stage 3 – Post Factum Data Analysis 

# The following code is similar (many of it copied) to that in the Jupyter notebook but makes Post Factum Analysis in a function and automates it for all the model structures, learning rates and random seeds
'''

# random seeds with sample values
randomSeeds = dict()
randomSeeds['sklearn'] = 20
randomSeeds['tensorflow'] = 32
randomSeeds['random'] = 42 # random seed of random module

# same Callback to store test loss and test accuracy after each epoch
class TestSetEvaluatorCallback(Callback):
    def __init__(self, X_test, y_test):
        super(TestSetEvaluatorCallback, self).__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.test_loss_values = []  # To store test losses for each epoch
        self.test_accuracy_values = []  # To store test accuracies for each epoch
    
    def on_epoch_end(self, epoch, logs=None):
        # Evaluate on the test set at the end of each epoch
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # Store the test loss and test accuracy in the history dictionary
        if 'test_loss' not in self.model.history.history:
            self.model.history.history['test_loss'] = []
        if 'test_accuracy' not in self.model.history.history:
            self.model.history.history['test_accuracy'] = []
        
        # Append the current epoch's test loss and accuracy
        self.test_loss_values.append(test_loss)
        self.test_accuracy_values.append(test_accuracy)

## same Model structures
# Model with a single hidden layer using ReLU activation
def modelStructure1HiddenReLULayer(X_train):
    return Sequential([
        Input(shape=(X_train.shape[1],)),  # Define the input shape
        Dense(32, activation='relu'),  # Single hidden layer with ReLU activation
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

# Model with a single hidden layer using ReLU activation and Dropout layer
def modelStructure1HiddenReLULayerWithDropoutLayer(X_train):
    return Sequential([
        Input(shape=(X_train.shape[1],)),  # Define the input shape
        Dense(32, activation='relu'),  # Single hidden layer with ReLU activation
        Dropout(0.2),  # Dropout layer to prevent overfitting
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

# Model with multiple hidden layers using ReLU activation
def modelStructureMultipleHiddenReLULayer(X_train):
    return Sequential([
        Input(shape=(X_train.shape[1],)),  # Define the input shape
        Dense(64, activation='relu'),  # First hidden layer with ReLU activation
        Dense(32, activation='relu'),  # Second hidden layer with ReLU activation
        Dense(16, activation='relu'),  # Third hidden layer with ReLU activation
        Dense(4, activation='relu'),   # Fourth hidden layer with ReLU activation
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

# Model with a single hidden layer using Tanh activation
def modelStructure1HiddenTanhLayer(X_train):
    return Sequential([
        Input(shape=(X_train.shape[1],)),  # Define the input shape
        Dense(32, activation='tanh'),  # Single hidden layer with tanh activation
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

# same Wrapper function to call one of the models based on a string parameter
def get_model_structure(model_type, X_train):
    """
    Returns a Keras model based on the specified model type.

    :param model_type: str, the model type to create. Options are:
        'single_hidden_relu', 'single_hidden_relu_dropout', 
        'multiple_hidden_relu', 'single_hidden_tanh'
    :return: Keras Sequential model
    """
    if model_type == modelStructure1HiddenReLULayer.__name__:
        return modelStructure1HiddenReLULayer(X_train)
    elif model_type == modelStructure1HiddenReLULayerWithDropoutLayer.__name__:
        return modelStructure1HiddenReLULayerWithDropoutLayer(X_train)
    elif model_type == modelStructureMultipleHiddenReLULayer.__name__:
        return modelStructureMultipleHiddenReLULayer(X_train)
    elif model_type == modelStructure1HiddenTanhLayer.__name__:
        return modelStructure1HiddenTanhLayer(X_train)
    else:
        # Raise an error with valid options listed
        valid_model_types = [
            modelStructure1HiddenReLULayer.__name__,
            modelStructure1HiddenReLULayerWithDropoutLayer.__name__,
            modelStructureMultipleHiddenReLULayer.__name__,
            modelStructure1HiddenTanhLayer.__name__
        ]
        raise ValueError(f"Invalid model type: {model_type}. Choose from {', '.join(valid_model_types)}.")


# Helper functions end here (same as those in Jupyter notebook). Primary functions start here

# New post_factum_analysis function with similar code inside it
def post_factum_analysis(model_structure: str, learning_rate: float, history, test_set_evaluator, X_test, y_test, y_train, model, filepath: str):
    """
    Perform post-factum data analysis after training, and save the results to files.
    """

    # File paths for saving graphs and confusion matrix
    graph_filepath = f"{filepath}_analysis.png"
    cm_filepath = f"{filepath}_confusion_matrix.png"

    # Stage 3 – Post Factum Data Analysis: Plot Bad Facts vs Epoch
    train_accuracy = np.array(history.history['accuracy'])
    total_training_samples = len(y_train)
    bad_facts_per_epoch = total_training_samples * (1 - train_accuracy)

    # Plotting Bad Facts vs Epoch
    plt.figure(figsize=(10, 12))

    # Plot 1: Bad Facts vs Epoch (Training Set)
    plt.subplot(3, 1, 1)
    plt.plot(range(1, len(bad_facts_per_epoch) + 1), bad_facts_per_epoch, color='red', label="Bad Facts")
    plt.xlabel("Epoch")
    plt.ylabel("Bad Facts")
    plt.title("Bad Facts vs Epoch (Training Set)")
    # plt.xticks(range(1, len(bad_facts_per_epoch) + 1))
    plt.legend()
    plt.grid(True)

    # Loss and Accuracy Curves
    epochs = np.arange(1, len(history.history['loss']) + 1)
    train_loss = history.history['loss']
    test_loss = test_set_evaluator.test_loss_values
    train_accuracy = history.history['accuracy']
    test_accuracy = test_set_evaluator.test_accuracy_values

    # Plot 2: Loss vs Epoch
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot 3: Accuracy vs Epoch
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the analysis plot
    plt.savefig(graph_filepath)
    plt.close()

    # Confusion Matrix
    y_pred = model.predict(X_test, batch_size=500)
    y_pred = (y_pred > 0.5).astype(int)  # Threshold for binary classification

    cm = confusion_matrix(y_test, y_pred)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_disp.plot(cmap="Blues", values_format="d")

    # Save the confusion matrix
    cm_disp.figure_.savefig(cm_filepath)
    plt.close()


    # Final Results File - Saving to final-results folder
    final_results_dir = "final-results"
    final_results_filepath = filepath.split('graphs')[0] + final_results_dir + filepath.split('graphs')[1] # incomplete path

    # Define the file path for saving the final results
    final_results_filepath = f"{final_results_filepath}_final_results.txt" # path till text file

    # Write the final results to the file
    with open(final_results_filepath, "w") as f:
        f.write(f"Model Structure: {model_structure}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Random seed - sklearn: {randomSeeds['sklearn']}\n")
        f.write(f"Random seed - random: {randomSeeds['random']}\n")
        f.write(f"Random seed - tensorflow: {randomSeeds['tensorflow']}\n")

        f.write(f"Final Training Loss: {train_loss[-1]:.4f}\n")
        f.write(f"Final Training Accuracy: {train_accuracy[-1] * 100:.2f}%\n")
        f.write(f"Final Test Loss: {test_loss[-1]:.4f}\n")
        f.write(f"Final Test Accuracy: {test_accuracy[-1] * 100:.2f}%\n")

    # Print final results for comparison (also to console)
    print(f"Final Training Loss: {train_loss[-1]:.4f}")
    print(f"Final Training Accuracy: {train_accuracy[-1] * 100:.2f}%")
    print(f"Final Test Loss: {test_loss[-1]:.4f}")
    print(f"Final Test Accuracy: {test_accuracy[-1] * 100:.2f}%")

def bulkTraining(randomSeedsTuple, model_structure: str, learning_rate: float, filepath: str):
    # Set randomSeeds dictionary
    randomSeeds['sklearn'] = randomSeedsTuple[0]
    randomSeeds['random'] = randomSeedsTuple[1]
    randomSeeds['tensorflow'] = randomSeedsTuple[2]

    # Open the file for writing
    filepathOutput = filepath
    if 'randomSeeds' in filepathOutput: filepathOutput = 'randomSeeds/output-random-seeds-0to2' # .txt
    with open(f"{filepathOutput}.txt", "a") as f:
        f.write(f'model_structure = {model_structure}\n')
        f.write(f'learning_rate = {learning_rate}\n')

        f.write(f"randomSeeds['sklearn'] = {randomSeeds['sklearn']}\n")
        f.write(f"randomSeeds['random'] = {randomSeeds['random']}\n")
        f.write(f"randomSeeds['tensorflow'] = {randomSeeds['tensorflow']}\n")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomSeeds['sklearn'], shuffle=True)

        # Set random seed for reproducibility
        random_seed(randomSeeds['random'])
        set_tensorflow_seed(randomSeeds['tensorflow'])

        # Build a simple MLP model (you can replace this with any of your model functions)
        model = get_model_structure(model_structure, X_train)

        # Compile the model directly here with the learning rate parameter
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',  # Assuming binary classification task
            metrics=['accuracy']
        )

        # Write the start of training to the file
        f.write("Train the model\n")

        # Initialise the TestSetEvaluator callback
        test_set_evaluator = TestSetEvaluatorCallback(X_test, y_test)

        # Train the model (with silent output)
        history = model.fit(X_train, y_train, epochs=1000, batch_size=500, shuffle=True, verbose=0, callbacks=[test_set_evaluator])

        # Test the model
        f.write("\nTest the model\n")

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=7032) # totalNumRows - numRowsDropped = 7032

        # Write the test results to the file
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")

        # Evaluate the accuracy thresholds
        if test_accuracy == 1:
            f.write(f'Accuracy 100%\n')
        elif test_accuracy >= 0.85:
            f.write(f'Accuracy 85% or more\n')
        elif test_accuracy >= 0.8:
            f.write(f'Accuracy 80% or more\n')

        # Add spacing and separator for clarity
        f.write('\n' + '-' * 15 + '\n\n')  # Spacing between function calls


        filepath = filepath.split('output')[0] + 'graphs' + filepath.split('output')[1]
        # Call post_factum_analysis for the saved graphs and confusion matrix
        post_factum_analysis(model_structure, learning_rate, history, test_set_evaluator, X_test, y_test, y_train, model, filepath=filepath)

    return test_accuracy


# List of model structures
model_structures = [
    "modelStructure1HiddenReLULayer",
    "modelStructure1HiddenReLULayerWithDropoutLayer",
    "modelStructureMultipleHiddenReLULayer",
    "modelStructure1HiddenTanhLayer"
]

# List of learning rates chosen
learning_rates = [
    0.001,
    0.01,
    0.2
]

# Subset of the combinations of model structures and learning rates above
tuples_model_structures_learning_rates = [
    ("modelStructure1HiddenReLULayer", 0.001),
    ("modelStructure1HiddenReLULayer", 0.01),
    ("modelStructure1HiddenReLULayer", 0.2),
    ("modelStructure1HiddenReLULayerWithDropoutLayer", 0.2),
    ("modelStructureMultipleHiddenReLULayer", 0.2),
    ("modelStructure1HiddenTanhLayer", 0.2)
]

# Ensure the required directories/folders exist
foldersRequired = []
for subfolder in ['', 'output', 'graphs', 'final-results']:
    foldersRequired.append('modelStructures_learningRates/' + subfolder)
    foldersRequired.append('randomSeeds/' + subfolder)
for folderRequired in foldersRequired:
    if not path_exists(folderRequired): makedirs(folderRequired)


# Driver code
# Run with different model_structures and learning_rates
for model_structure, learning_rate in tuples_model_structures_learning_rates:
    print('model_structure =', model_structure, ', learning_rate =', learning_rate)
    filepath = f"modelStructures_learningRates/output/{model_structure}_{learning_rate}" # .txt
    with open(filepath + '.txt', 'w') as file:
        file.write('')
    test_accuracy = bulkTraining(randomSeedsTuple=(0,1,2), model_structure=model_structure, learning_rate=learning_rate, filepath=filepath)
    if test_accuracy == 1: break

# Run with different random seeds
with open('randomSeeds/output-random-seeds-0to2.txt', 'w') as file:  file.write('') # list results of each set of random seeds in the same file for `output_model_different_random_seeds.py`
for seeds in permutations(range(3), len(randomSeeds)):
    filepath = f"randomSeeds/output/sk{seeds[0]}_random{seeds[1]}_tf{seeds[2]}" # .txt
    test_accuracy = bulkTraining(seeds, model_structure=model_structures[0], learning_rate=learning_rates[0], filepath=filepath)
    if test_accuracy == 1: break

# Generate graphs that give overview of performance of all runs of different random seeds
import output_model_different_random_seeds # it has code outside i.e. executed upon import, so important to import it in last line