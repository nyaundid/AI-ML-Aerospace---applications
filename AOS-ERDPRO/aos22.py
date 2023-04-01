import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# Load the data into a pandas dataframe
data = pd.read_csv(r"C:\Users\darius.nyaundi\Documents\aos123.csv")

# Extract the text data for the 'Description' and 'Event ATA' columns
text_data = data[['Description', 'Event ATA']].apply(lambda x: ' '.join(x), axis=1).values

# Use CountVectorizer to create a bag of words representation of the text
vectorizer = CountVectorizer()
text_data_vectorized = vectorizer.fit_transform(text_data).toarray()

# One-hot encode the target "Event Problem"
target = data['Event Problem'].values
target = np.array(target)
y = np.zeros((len(target), np.unique(target).shape[0]))
y[np.arange(len(target)), target] = 1

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(text_data_vectorized, y, test_size=0.2, random_state=0)

# Create a sequential neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=text_data_vectorized.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)

# Create a function to make predictions based on a new text input
def predict_event_problem(text, event_ata):
    text_vectorized = vectorizer.transform([f"{text} {event_ata}"]).toarray()
    prediction = model.predict(text_vectorized)
    return np.argmax(prediction[0])

# GUI code

import tkinter as tk

def predict_event_problem_GUI():
    text = text_entry.get()
    event_ata = event_ata_entry.get()
    prediction = predict_event_problem(text, event_ata)
    prediction_label.config(text=f"Event Problem: {prediction}")

# Create the GUI window
root = tk.Tk()
root.title("Event Problem Predictor")

# Create the text entry widget
text_entry = tk.Entry(root)
text_entry.grid(row=0, column=0)

# Create the event ATA entry widget
event_ata_entry = tk.Entry(root)
event_ata_entry.grid(row=1, column=0)

# Create the prediction button
predict_button = tk.Button(root, text="Predict", command=predict_event_problem_GUI)
predict_button.grid(row=2, column=0)

# Create the prediction label
prediction_label = tk.Label(root, text="")
prediction_label.grid(row=3, column=0)

# Start the GUI event loop
root.mainloop()