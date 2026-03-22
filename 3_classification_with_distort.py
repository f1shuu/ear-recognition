import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import sys
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def classification_LOPO_CV(df, df_dist, classifier):
    real = []
    prediction = []
    training_times = []
    prediction_times = []
    for id_person in range(1, 49):
        # Podziel zbiór rekordów na rekordy użytkownika (o numerze id_person) i resztę.
        df_person = df[df[df.columns[-1]] == id_person].reset_index(drop=True)
        df_rest = df[df[df.columns[-1]] != id_person].reset_index(drop=True)

        df_person_dist = df_dist[df_dist[df_dist.columns[-1]] == id_person].reset_index(drop=True)
        df_rest_dist = df_dist[df_dist[df_dist.columns[-1]] != id_person].reset_index(drop=True)

        # Podziel zbiór użytkownika (id_person) na dane i klasę. Nadaj klasie wartość 1.
        df_person = df_person.drop(df_person.columns[-1], axis=1)
        df_person['class'] = 1

        df_person_dist = df_person_dist.drop(df_person_dist.columns[-1], axis=1)
        df_person_dist['class'] = 1

        # Weź losowo 9 rekordów innych użytkowników (ze zbioru reszty) i nadaj im klasę 0.
        random_rest = df_rest.sample(n=9, random_state=0)
        random_rest = random_rest.drop(random_rest.columns[-1], axis=1)
        random_rest['class'] = 0

        random_rest_dist = df_rest_dist.sample(n=9, random_state=0)
        random_rest_dist = random_rest_dist.drop(random_rest_dist.columns[-1], axis=1)
        random_rest_dist['class'] = 0

        # Podziel zbiór reszty użytkowników na dane i klasę.
        X_random_rest = random_rest.drop(columns=['class'])
        y_random_rest = random_rest['class']

        X_random_rest_dist = random_rest_dist.drop(columns=['class'])
        y_random_rest_dist = random_rest_dist['class']

        # Test dla próbek pozytywnych.
        for i in range(0, len(df_person)):
            # Weź pierwszy rekord użytkownika do testów, z pozostałych utwórz zbiór treningowy.
            df_test = df_person_dist.iloc[[i]]
            df_train = df_person.drop(i)

            # Dzielę rekord testowy na dane i klasę.
            X_df_test = df_test.drop(columns=['class'])
            #y_df_test = df_test['class']

            # Dzielę rekordy treningowe na dane i klasy.
            X_df_train = df_train.drop(columns=['class'])
            y_df_train = df_train['class']

            # Łączę rekordy treningowe z i wybranymi losowo 9 rekordami reszty i dzielę na dane i klasy.
            X_df_train = pd.concat([X_df_train, X_random_rest], ignore_index=True)
            y_df_train = pd.concat([y_df_train, y_random_rest], ignore_index=True)

            # Uczę, a następnie klasyfikuję wybrany rekord.
            # Zbieram czasy.
            start_train = time.perf_counter()
            classifier.fit(X_df_train, y_df_train)
            end_train = time.perf_counter()
            training_times.append((end_train - start_train) * 1000)  # ms

            start_pred = time.perf_counter()
            predict = classifier.predict(X_df_test)
            end_pred = time.perf_counter()
            prediction_times.append((end_pred - start_pred) * 1000)  # ms

            # Dodaje wyniki do tablicy.
            real.append(1)
            prediction.append(predict[0])

        # Test dla próbek negatywnych.
        for i in range(0, len(df_person)):
            # Weź losowy rekord z reszty użytkowników do testów i podziel go na dane i klasę.
            random_record = df_rest_dist.sample(n=1, random_state=0)
            X_df_test = random_record.iloc[:, :-1]

            # Ze zbioru użytkownika usuń rekord i-ty (tak żeby zostało 9 rekordów).
            # Podziel ten zbiór na dane i klasy.
            df_train = df_person.drop(i)
            X_df_train = df_train.drop(columns=['class'])
            y_df_train = df_train['class']

            # Połącz rekordy użytkownika z losowo wybranymi 9 rekordami reszty i dzielę na dane i klasy.
            X_df_train = pd.concat([X_df_train, X_random_rest], ignore_index=True)
            y_df_train = pd.concat([y_df_train, y_random_rest], ignore_index=True)

            # Uczę, a następnie klasyfikuję wybrany rekord.
            # Zbieram czasy.
            start_train = time.perf_counter()
            classifier.fit(X_df_train, y_df_train)
            end_train = time.perf_counter()
            training_times.append((end_train - start_train) * 1000)

            start_pred = time.perf_counter()
            predict = classifier.predict(X_df_test)
            end_pred = time.perf_counter()
            prediction_times.append((end_pred - start_pred) * 1000)

            # Dodaje wyniki do tablicy.
            real.append(0)
            prediction.append(predict[0])


    cm = confusion_matrix(real, prediction)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)

    return pd.Series({
        "ACC": accuracy_score(real, prediction),
        "F1 ": f1_score(real, prediction),
        "PPV": precision_score(real, prediction),
        "FAR": far,
        "FRR": frr,
        "TPR": tpr, #Recall
        "TNR": tnr,
        "TrainTime [ms]": round(sum(training_times) / len(training_times), 2),
        "PredictTime [ms]": round(sum(prediction_times) / len(prediction_times), 2)
    })

# Wczytanie danych do DataFrame.
image_path = "data/"
file_name_1 = "ear_data_aug_denoise_bright.csv"
file_name_2 = "ear_data_aug_denoise_bright_light150.csv"

df_RAW_1 = pd.read_csv(image_path + file_name_1)
df_StandardScaler_1 = pd.concat([pd.DataFrame(StandardScaler().fit_transform(df_RAW_1.iloc[:, :-1]), columns=df_RAW_1.columns[:-1]), df_RAW_1.iloc[:, -1]], axis=1)
df_RAW_2 = pd.read_csv(image_path + file_name_2)
df_StandardScaler_2 = pd.concat([pd.DataFrame(StandardScaler().fit_transform(df_RAW_2.iloc[:, :-1]), columns=df_RAW_2.columns[:-1]), df_RAW_2.iloc[:, -1]], axis=1)

per = Perceptron(max_iter=1000, random_state=0)
mlp = MLPClassifier(max_iter=1000, random_state=0)
svc = SVC(kernel='rbf', gamma='auto', random_state=0)
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()
sgd = SGDClassifier(max_iter=1000, random_state=0)
rf = RandomForestClassifier(random_state=0)
qd = QuadraticDiscriminantAnalysis()

data_1 = [df_StandardScaler_1]
data_2 = [df_StandardScaler_2]

data_names = ["Scaled"]
classifiers = [per, mlp, svc, dt, knn, nb, sgd, rf, qd]
classifiers_names = [
    "Perceptron (Single-layer Neural Network)",
    "Multi-layer Perceptron",
    "SVC (Support Vector Classifier / Support Vector Machine)",
    "Decision Tree Classifier",
    "KNeighbors Classifier (kNN)",
    "GaussianNB (Gaussian Naive Bayes)",
    "SGD Classifier (Stochastic Gradient Descent Classifier)",
    "Random Forest Classifier",
    "Quadratic Discriminant Analysis (QDA)"
]

for j in range(0, len(classifiers)):
    results_all = []
    name = classifiers_names[j]
    #print(name)
    for i in range(0, len(data_1)):
        results = classification_LOPO_CV(data_1[i], data_2[i], classifiers[j])
        #results_all.append(results.rename(data_names[i]))
        results_all.append(results[0])

    print(name,' ',results_all[0].round(3))

    '''
    df_results = pd.concat(results_all, axis=1).round(3)
    df_display = df_results.copy()
    df_display.reset_index(inplace=True)
    df_display.insert(0, 'Classifier', '')

    row_for_name = df_display[df_display['index'] == 'FAR'].index[0]
    df_display.loc[row_for_name, 'Classifier'] = name
    df_display.rename(columns={'index': 'Metrics'}, inplace=True)
    df_display.loc[df_display["Classifier"] == name, "Classifier parameters"] = "---"
    df_display.fillna('', inplace=True)

    print(df_display.to_string(index=False, justify='left'))
    '''

    '''
    latex = df_display.to_latex(index=False, column_format='llrrrl', multicolumn=False)
    print(latex)
    '''
