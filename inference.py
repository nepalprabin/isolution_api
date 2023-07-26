import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import json

# Load models
identification_model = keras.models.load_model('models/identification_model.h5')
resolution_model = keras.models.load_model('models/resolution_model.h5')

# Load data from json
with open('input.json') as json_file:
    data = json.load(json_file)
input_data = pd.DataFrame([data])

# Ensure the input data is in the correct format and type
input_data = input_data.fillna(0).astype('int')

# Inference for 'problem_rating_identifitication'
preds_identification = np.argmax(identification_model.predict(input_data), axis = 1)

# Inference for 'problem_rating_resolution'
preds_resolution = np.argmax(resolution_model.predict(input_data), axis = 1)

# Create output json
output = {
    "problem_rating_identifitication": preds_identification.tolist(),
    "problem_rating_resolution": preds_resolution.tolist()
}

# Save output to json
with open('output.json', 'w') as json_file:
    json.dump(output, json_file)

print("Predictions saved to output.json")
