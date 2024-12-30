import pandas as pd
import re

filename = '1-hidden-ReLU-layer'

text = None
with open(f'output-model-different-random-seeds/{filename}.txt', 'r') as file:
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