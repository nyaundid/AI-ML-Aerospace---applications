import tkinter as tk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data into a pandas dataframe
data = pd.read_csv(r"C:\Users\105146\Documents\New2023\ALLERD.csv")

#data = pd.read_csv(r"C:\Users\darius.nyaundi\Documents\aos123.csv")
#data = pd.read_csv(r"C:\Users\105146\Documents\New2023\ALLERD.csv")

# Extract the text data for the 'Occurrence' and 'ATA-SUB' columns
text_data = data[['Occurrence', 'ATA-SUB']].apply(lambda x: ' '.join(x), axis=1).values

# Use CountVectorizer to create a bag of words representation of the text
vectorizer = CountVectorizer()
text_data_vectorized = vectorizer.fit_transform(text_data)

# Train a Random Forest classifier on the text data with 100 trees
rf = RandomForestClassifier(n_estimators=100)
rf.fit(text_data_vectorized, data['Bucket 1'].values)

# Create a GUI for the sample input text and event problem prediction
root = tk.Tk()
root.title("Bucket 1 Predictor")

label1 = tk.Label(root, text="Sample Input (Occurrence):")
label1.pack()

entry1 = tk.Entry(root)
entry1.pack()

label2 = tk.Label(root, text="Sample Input (ATA-SUB):")
label2.pack()

entry2 = tk.Entry(root)
entry2.pack()

label3 = tk.Label(root, text="Bucket 1 Predictions:")
label3.pack()

def predict():
    # Get the input values from the text boxes
    sample_input_desc = entry1.get()
    sample_input_event_ata = entry2.get()
    
    # Vectorize the input text using the trained CountVectorizer
    text_vectorized = vectorizer.transform([f"{sample_input_desc} {sample_input_event_ata}"])
    
    # Make predictions using the trained model
    predictions = rf.predict(text_vectorized)
    
    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(data['Bucket 1'].values, rf.predict(text_data_vectorized))
    
    # Get the top 3 predictions with their corresponding accuracy scores
    proba = rf.predict_proba(text_vectorized)
    top_3 = sorted(zip(rf.classes_, proba[0]), key=lambda x: x[1], reverse=True)[:3]
    top_3_str = '\n'.join([f"{p[0]} ({p[1]:.2f} accuracy)" for p in top_3])
    
    # Update the output label
    label3.config(text=f"Bucket 1 Predictions:\n{top_3_str}")

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

root.mainloop()