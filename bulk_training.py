# Load the data from the .npz (numpy archive) file
numpy_archive = np.load("Xy.npz")

# Extract X and y
X = numpy_archive['X']
y = numpy_archive['y']

'''
Stage 2 – Creation of the MLP using Keras , and
Stage 3 – Post Factum Data Analysis 

# The following code is similar to that in the Jupyter notebook but makes Post Factum Analysis in a function and automates it for all the model structures, learning rates and random seeds
'''
from os.path import exists as path_exists
from os import makedirs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def post_factum_analysis(model_structure: str, learning_rate: float, history, test_set_evaluator, X_test, y_test):
    """
    Perform post-factum data analysis after training, and save the results to files.
    """
    # Ensure the graph directory exists
    graph_dir = "graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    # File paths for saving graphs and confusion matrix
    graph_filepath = os.path.join(graph_dir, f"{model_structure}_{learning_rate}_analysis.png")
    cm_filepath = os.path.join(graph_dir, f"{model_structure}_{learning_rate}_confusion_matrix.png")

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
    plt.xticks(range(1, len(bad_facts_per_epoch) + 1))
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

    # Print final results for comparison
    print(f"Final Training Loss: {train_loss[-1]:.4f}")
    print(f"Final Training Accuracy: {train_accuracy[-1] * 100:.2f}%")
    print(f"Final Test Loss: {test_loss[-1]:.4f}")
    print(f"Final Test Accuracy: {test_accuracy[-1] * 100:.2f}%")

def bulkTraining(randomSeedsTuple, model_structure: str, learning_rate: float):
    # Set randomSeeds dictionary
    randomSeeds['sklearn'] = randomSeedsTuple[0]
    randomSeeds['random'] = randomSeedsTuple[1]
    randomSeeds['tensorflow'] = randomSeedsTuple[2]

    # Open the file for writing
    with open(f"output/{model_structure}_{learning_rate}.txt", "a") as f:
        f.write(f"randomSeeds['sklearn'] = {randomSeeds['sklearn']}\n")
        f.write(f"randomSeeds['random'] = {randomSeeds['random']}\n")
        f.write(f"randomSeeds['tensorflow'] = {randomSeeds['tensorflow']}\n")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomSeeds['sklearn'], shuffle=True)

        # Set random seed for reproducibility
        random_seed(randomSeeds['random'])
        set_tensorflow_seed(randomSeeds['tensorflow'])

        # Build a simple MLP model (you can replace this with any of your model functions)
        model = get_model_structure(model_structure)

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
        test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=totalNumRows - numRowsDropped)

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

        # Call post_factum_analysis for the saved graphs and confusion matrix
        post_factum_analysis(model_structure, learning_rate, history, test_set_evaluator, X_test, y_test)

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

# Driver code
for model_structure, learning_rate in tuples_model_structures_learning_rates:
    print('model_structure =', model_structure, ', learning_rate =', learning_rate)
    with open(f"output/{model_structure}_{learning_rate}.txt", "w") as f: # clear file
        f.write("")

    for seeds in permutations(range(3), len(randomSeeds)):
        test_accuracy = bulkTraining(seeds, model_structure=model_structure, learning_rate=learning_rate)
        if test_accuracy == 1:
            break
