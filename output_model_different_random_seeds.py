import pandas as pd
import re

# base = 'output-model-different-random-seeds/'
# modelCharacteristic = '1-hidden-ReLU-layer'

text = None
with open(f'randomSeeds/output-random-seeds-0to2.txt', 'r') as file:
    text = file.read()
if not text: raise Exception('Variable text not populated. Maybe file is open - If yes, close it.')

# Initialize a list to store the parsed data
data = []

# Define a regular expression pattern to extract the relevant data
pattern = r"randomSeeds\['sklearn'\] = (\d+)\s+randomSeeds\['random'\] = (\d+)\s+randomSeeds\['tensorflow'\] = (\d+).*?Test Accuracy: ([\d\.]+).*?Test Loss: ([\d\.]+)"

# Use the findall method to find all matches of the pattern
matches = re.findall(pattern, text, re.DOTALL)

# Loop through the matches and store them in the data list
for match in matches:
    sklearn_seed, random_seed, tensorflow_seed, accuracy, loss = match
    data.append({
        'sklearn_seed': int(sklearn_seed),
        'random_seed': int(random_seed),
        'tensorflow_seed': int(tensorflow_seed),
        'accuracy': float(accuracy),
        'loss': float(loss)
    })

# Create a pandas DataFrame from the collected data
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Visualisations of test accuracies and losses over runs

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Create a PdfPages object to save multiple plots in one PDF
with PdfPages(f'randomSeeds/graphs-random-seeds-0to2_model_visualizations_based_on_output.pdf') as pdf:

    # Test Accuracy vs Run Number
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['accuracy'], marker='o', linestyle='-', color='b', label='Test Accuracy')
    plt.title('Test Accuracy Over Different Runs', fontsize=14)
    plt.xlabel('Run Number', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.grid(True)
    plt.legend()
    pdf.savefig()  # Save the current figure to the PDF
    plt.close()

    # Test Loss vs Run Number
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['loss'], marker='o', linestyle='-', color='r', label='Test Loss')
    plt.title('Test Loss Over Different Runs', fontsize=14)
    plt.xlabel('Run Number', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.grid(True)
    plt.legend()
    pdf.savefig()  # Save the current figure to the PDF
    plt.close()

    # Accuracy and Loss Comparison in One Plot (Dual Y-Axis)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot accuracy
    ax1.plot(df.index, df['accuracy'], marker='o', linestyle='-', color='b', label='Test Accuracy')
    ax1.set_xlabel('Run Number', fontsize=12)
    ax1.set_ylabel('Test Accuracy', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis to plot loss
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['loss'], marker='o', linestyle='-', color='r', label='Test Loss')
    ax2.set_ylabel('Test Loss', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')

    # Title and grid
    plt.title('Test Accuracy and Loss Over Different Runs', fontsize=14)
    ax1.grid(True)

    pdf.savefig()  # Save the current figure to the PDF
    plt.close()

    # Accuracy vs Loss Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['accuracy'], df['loss'], color='purple', alpha=0.7)
    plt.title('Accuracy vs Loss for Different Runs', fontsize=14)
    plt.xlabel('Test Accuracy', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.grid(True)
    pdf.savefig()  # Save the current figure to the PDF
    plt.close()

    # Visualizing Seed Effects on Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(df['sklearn_seed'], df['accuracy'], marker='o', linestyle='-', color='b', label='sklearn_seed')
    plt.plot(df['random_seed'], df['accuracy'], marker='o', linestyle='-', color='g', label='random_seed')
    plt.plot(df['tensorflow_seed'], df['accuracy'], marker='o', linestyle='-', color='r', label='tensorflow_seed')
    plt.title('Accuracy Comparison for Different Seeds', fontsize=14)
    plt.xlabel('Seed Value', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.grid(True)
    plt.legend()
    pdf.savefig()  # Save the current figure to the PDF
    plt.close()

    # Visualizing Seed Effects on Accuracy (Scatter)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['sklearn_seed'], df['accuracy'], color='b', label='sklearn_seed')
    plt.scatter(df['random_seed'], df['accuracy'], color='g', label='random_seed')
    plt.scatter(df['tensorflow_seed'], df['accuracy'], color='r', label='tensorflow_seed')
    plt.title('Accuracy Comparison for Different Seeds (Scatter)', fontsize=14)
    plt.xlabel('Seed Value', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.grid(True)
    plt.legend()
    pdf.savefig()  # Save the current figure to the PDF
    plt.close()

    # Bar Chart for Accuracies and Losses
    df.plot(kind='bar', x='sklearn_seed', y=['accuracy', 'loss'], figsize=(12, 6))
    plt.title('Accuracy and Loss for Each Run (Based on sklearn_seed)', fontsize=14)
    plt.xlabel('Run Number (sklearn_seed)', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    pdf.savefig()  # Save the current figure to the PDF
    plt.close()


# Get the 5 rows with the highest accuracy
top_5_accuracy = df.nlargest(5, 'accuracy')
print('\ntop_5_accuracy', top_5_accuracy, sep='\n')
# Get the 5 rows with the highest accuracy
top_5_loss = df.nsmallest(5, 'loss')
print('\ntop_5_loss', top_5_loss, sep='\n')

print('\nSelected row (the one having the lowest loss from the Top 5 Accuracy): ', top_5_accuracy.nsmallest(1, 'loss'), sep='\n')