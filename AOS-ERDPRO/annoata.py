import tkinter as tk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
# GUI
root = tk.Tk()
root.title("Text Classification")
root.geometry("400x300")

label = tk.Label(root, text="Enter the path to the CSV file:", width=300)
label.pack()

entry = tk.Entry(root)
entry.pack()

def predict_accuracy():
    filepath = entry.get()
    data = pd.read_csv(filepath)
    X_test = vectorizer.transform(data['Defect Description'])
    y_test = data['ATA-Sub']
    y_pred = clf.predict(X_test)
    result = ""
    for i, row in data.iterrows():
        result += f"{row['ATA-Sub']}-{row['Defect Description']}:\n"
        proba = clf.predict_proba(vectorizer.transform([row['Defect Description']]))[0]
        top3_indices = proba.argsort()[-3:][::-1]
        top3_accuracies = proba[top3_indices]
        top3_predictions = [clf.classes_[i] for i in top3_indices]
        for j in range(3):
            result += f"\t{top3_predictions[j]}: {top3_accuracies[j]:.4f}\n"
    output_label["text"] = result

predict_button = tk.Button(root, text="Predict Accuracy", command=predict_accuracy)
predict_button.pack(side=tk.LEFT)

copy_button = tk.Button(root, text="Copy", command=lambda: root.clipboard_append(output_label["text"]))
copy_button.pack(side=tk.LEFT)

output_label = tk.Label(root, text="", justify="left")
output_label.pack()

# Train the model
data = pd.read_csv("C:/Users/darius.nyaundi/Documents/ATAANNO.csv")
train_data, _ = train_test_split(data, test_size=0.2, random_state=1)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['Defect Description'])
y_train = train_data['ATA-Sub']
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

root.mainloop()