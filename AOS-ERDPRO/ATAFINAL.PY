import tkinter as tk
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score

import openpyxl 

data = pd.read_csv("C:/Users/darius.nyaundi/Documents/PIREPS.csv")
#data = pd.read_csv("C:/Users/105146/Documents/New2023/PIREPS.csv")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['Defect Description'])
X_test = vectorizer.transform(test_data['Defect Description'])
y_train = train_data['ATA-Sub']
y_test = test_data['ATA-Sub']
clf = RandomForestClassifier(n_estimators=100) # set n_estimators to 100
clf.fit(X_train, y_train)

def predict_label():
    new_text = entry.get()
    new_text_transformed = vectorizer.transform([new_text])
    predictions_proba = clf.predict_proba(new_text_transformed)
    predictions_proba = predictions_proba[0]
    top3_indices = predictions_proba.argsort()[-3:][::-1]
    top3_accuracies = predictions_proba[top3_indices]
    top3_predictions = [clf.classes_[i] for i in top3_indices]
    label["text"] = f"Predicted labels: {top3_predictions} with accuracy: {top3_accuracies}"

# GUI
root = tk.Tk()
root.title("Text Classification")
root.geometry("400x200")

label = tk.Label(root, text="Enter a new text sample:", width=300)
label.pack()

entry = tk.Entry(root)
entry.pack()

predict_button = tk.Button(root, text="Predict Label", command=predict_label)
predict_button.pack()

label = tk.Label(root)
label.pack()

root.mainloop()