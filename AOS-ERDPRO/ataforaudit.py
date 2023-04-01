import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load training data
train_data = pd.read_csv("C:/Users/darius.nyaundi/Documents/PIREPS.csv")
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['Defect Description'])
y_train = train_data['ATA-Sub']
clf = RandomForestClassifier(n_estimators=100) # set n_estimators to 100
clf.fit(X_train, y_train)

# Load new data and predict labels
new_data = pd.read_csv("C:/Users/darius.nyaundi/Documents/justnewp.csv")
X_new = vectorizer.transform(new_data['Defect Description'])
y_new = new_data['ATA-Sub']
predictions = clf.predict(X_new)
probabilities = clf.predict_proba(X_new)
probabilities_max = probabilities.max(axis=1)
results = pd.DataFrame({'Prediction': predictions, 'Accuracy': probabilities_max})

# Show results in the console
for i, row in results.iterrows():
    print(f"Row {i+2}: Prediction={row['Prediction']}, Accuracy={row['Accuracy']:.2f}")



