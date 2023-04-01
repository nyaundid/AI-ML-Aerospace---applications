import tkinter as tk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data into a pandas dataframe
data = pd.read_csv(r"C:\Users\105146\Documents\New2023\ALLAOS.csv")

#data = pd.read_csv(r"C:\Users\darius.nyaundi\Documents\aos123.csv")

# Convert 'Description', 'Event ATA', and 'Event Problem' columns to strings
data['Description'] = data['Description'].astype(str)
data['Event ATA'] = data['Event ATA'].astype(str)
data['Event Problem'] = data['Event Problem'].astype(str)

# Extract the text data for the 'Description' and 'Event ATA' columns
text_data = data[['Description', 'Event ATA']].apply(lambda x: ' '.join(x), axis=1).values

# Use CountVectorizer to create a bag of words representation of the text
vectorizer = CountVectorizer()
text_data_vectorized = vectorizer.fit_transform(text_data)

# Train a Random Forest classifier on the text data with 100 trees
rf = RandomForestClassifier(n_estimators=100)
rf.fit(text_data_vectorized, data['Event Problem'].values)

# Create a GUI for the sample input text and event problem prediction
root = tk.Tk()
root.title("Event Problem Predictor")

label1 = tk.Label(root, text="Sample Input (Description):")
label1.pack()

entry1 = tk.Entry(root)
entry1.pack()

label2 = tk.Label(root, text="Sample Input (Event ATA):")
label2.pack()

entry2 = tk.Entry(root)
entry2.pack()

label3 = tk.Label(root, text="Event Problem Predictions:")
label3.pack()

label4 = tk.Label(root, text="Top 3 Predictions with Accuracy Scores:")
label4.pack()

def predict():
    # Get the input values from the text boxes
    sample_input_desc = entry1.get()
    sample_input_event_ata = entry2.get()
    
    # Vectorize the input text using the trained CountVectorizer
    text_vectorized = vectorizer.transform([f"{sample_input_desc} {sample_input_event_ata}"])
    
    # Make predictions using the trained model
    predictions = rf.predict(text_vectorized)
    
    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(data['Event Problem'].values, rf.predict(text_data_vectorized))
    
    # Get the top 3 predictions with their probability scores
    prob_scores = rf.predict_proba(text_vectorized)
    top_3_indices = prob_scores.argsort()[0][-3:][::-1]
    top_3_probs = prob_scores[0][top_3_indices]
    top_3_predictions = rf.classes_[top_3_indices]
    
    # Update the output labels
    label3.config(text=f"Event Problem Prediction: {predictions[0]}")
    label4.config(text=f"Top 3 Predictions with Accuracy Scores: {top_3_predictions} {top_3_probs.round(2)} ({accuracy:.2f} accuracy)")

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

root.mainloop()