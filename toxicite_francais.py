import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import tkinter as tk
from tkinter import messagebox

# Charger le fichier CSV
file_path = '/Users/maxwolfers/Documents/toxicity-french-upsampled.csv'
data = pd.read_csv(file_path)

# Afficher les premières lignes pour comprendre la structure
print(data.head())

# Supprimer les lignes avec des valeurs manquantes dans la colonne 'Texte'
data = data.dropna(subset=['Texte'])

# Fonction de nettoyage du texte
def clean_text(text):
    text = str(text).lower()  # Convertir en minuscules
    text = re.sub(f"[{string.punctuation}]", "", text)  # Supprimer la ponctuation
    text = re.sub("\n", " ", text)  # Supprimer les sauts de ligne
    text = re.sub("\d+", "", text)  # Supprimer les chiffres
    return text

# Appliquer le nettoyage du texte
data['Texte'] = data['Texte'].apply(lambda x: clean_text(x) if isinstance(x, str) else '')

# Diviser les données en caractéristiques (X) et labels (y)
X = data['Texte']
y = data['oh_label']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir les textes en vecteurs TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entraîner un modèle de régression logistique
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print(report)

# Fonction pour prédire la toxicité d'une nouvelle phrase
def predict_toxicity(text):
    text = clean_text(text)
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return "Toxique" if prediction[0] == 1 else "Non toxique"


# Interface utilisateur avec tkinter
def check_toxicity():
    user_input = entry.get()
    result = predict_toxicity(user_input)
    messagebox.showinfo("Résultat", f"La phrase est : {result}")

# Configuration de la fenêtre principale
root = tk.Tk()
root.title("Détecteur de Toxicité")

# Champ de saisie
entry_label = tk.Label(root, text="Entrez une phrase :")
entry_label.pack(pady=5)
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

# Bouton de vérification
check_button = tk.Button(root, text="Vérifier Toxicité", command=check_toxicity)
check_button.pack(pady=20)

# Lancement de la boucle principale de l'interface utilisateur
root.mainloop()