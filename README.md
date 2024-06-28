# Détecteur de Toxicité des Commentaires en Français

## Description

Ce projet utilise des techniques de machine learning pour détecter la toxicité dans les commentaires en français. Un modèle de régression logistique est entraîné à partir de commentaires prétraités et vectorisés à l'aide de TF-IDF. Une interface utilisateur simple, créée avec `tkinter`, permet d'entrer des phrases et de vérifier leur toxicité.

## Fonctionnalités

- **Préparation des données** : Nettoyage et prétraitement des commentaires pour éliminer les ponctuations, les chiffres, et mettre en minuscules.
- **Vectorisation** : Conversion des textes en vecteurs numériques à l'aide de la méthode TF-IDF.
- **Modélisation** : Utilisation d'un modèle de régression logistique pour classifier les commentaires en toxiques ou non toxiques.
- **Interface Utilisateur** : Interface graphique pour saisir des phrases et vérifier leur toxicité.

## Résultats

Le modèle atteint une précision de 93.7% sur l'ensemble de test, démontrant une forte capacité de prédiction.

## Prérequis

- Python 3.x
- pandas
- numpy
- scikit-learn
- tkinter

## Installation

1. Clonez le dépôt :
    ```sh
    git clone https://github.com/maxwolfers/toxicite_francais.git
    cd toxicite_francais
    ```

2. Installez les dépendances :
    ```sh
    pip install pandas numpy scikit-learn
    ```
3. Installez la base de donnés en fichier CSV :
   
   https://www.data.gouv.fr/fr/datasets/r/385c7a35-61cd-48e6-8eb2-7d975be6349f

   

## Utilisation

1. Assurez-vous que le fichier CSV contenant les données se trouve dans le répertoire du projet et modifiez le chemin du fichier dans `toxicite_francais.py` :
    ```python
    file_path = '/path/to/your/toxicity-french-upsampled.csv'
    ```

2. Exécutez le script `toxicite_francais.py` pour lancer l'interface utilisateur :
    ```sh
    toxicite_francais.py
    ```

3. Entrez une phrase dans le champ de saisie et cliquez sur "Vérifier Toxicité" pour voir si la phrase est toxique ou non.

