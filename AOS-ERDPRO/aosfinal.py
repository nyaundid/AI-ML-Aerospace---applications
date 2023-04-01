import tkinter as tk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the data into a pandas dataframe
data = pd.read_csv(r"C:\Users\darius.nyaundi\Documents\aos123.csv")

# Extract the text data for the 'Description' column
text_data = data['Description'].values

# Use CountVectorizer to create a bag of words representation of the text
vectorizer = CountVectorizer()
text_data_vectorized = vectorizer.fit_transform(text_data)

# Train a Multinomial Naive Bayes classifier on the text data
nb = MultinomialNB()
nb.fit(text_data_vectorized, data['Event Problem'].values)

# Create a function to make predictions based on a new text input
def predict_event_problem(text):
    text_vectorized = vectorizer.transform([text])
    prediction = nb.predict(text_vectorized)[0]
    return prediction

# Create a GUI for the sample input text and event problem prediction
root = tk.Tk()
root.title("Event Problem Predictor")

label1 = tk.Label(root, text="Sample Input:")
label1.pack()

entry1 = tk.Entry(root)
entry1.pack()

label2 = tk.Label(root, text="Event Problem Prediction:")
label2.pack()

def predict():
    sample_input = entry1.get()
    event_problem = predict_event_problem(sample_input)
    label2.config(text=f"Event Problem Prediction: {event_problem}")

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

root.mainloop()