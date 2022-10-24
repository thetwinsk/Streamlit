import pandas as pd
import numpy as np
import streamlit as st
import sklearn

from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Loan Prediction")
#st.write("""
# Explore your approval loan
#Which one is the best?
#""")

#dataset_name = st.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))
#st.write(dataset_name)

#classifier_name = st.selectbox("Select Classifier", ("Naive Bayes -Bernoulli",
#                                                     "KNN", "SVM", "Random Forest")) #"Logistic Regression",

path = '/Users/kaylanguyen/Documents/babydragon/OurProjects/LoanPrediction/'
X = pd.read_csv(path + 'X_loan.csv')

from sklearn.utils.validation import column_or_1d
y = pd.read_csv(path + 'y_loan.csv')
y = column_or_1d(y, warn=False)

# def get_dataset(dataset_name):
#     if dataset_name == "Iris":
#         data = datasets.load_iris()
#     elif dataset_name == "Breast Cancer":
#         data = datasets.load_breast_cancer()
#     else:
#         data = datasets.load_wine()
#     X = data.data
#     y = data.target
#     return X, y

# X, y = get_dataset(dataset_name)
# st.write("shape of dataset", X.shape)
# st.write("number of classes", len(np.unique(y)))
#
def add_parameter_ui(clf_name):
    params = dict()
    # if clf_name == "Logistic Regression":
    #     pass
    if clf_name == "Naive Bayes -Bernoulli":
        pass
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params
#
#params = add_parameter_ui(classifier_name)
#
def get_classifier(clf_name, params):
    # if clf_name == "Logistic Regression":
    #     pass
    if clf_name == "Naive Bayes -Bernoulli":
        clf = BernoulliNB()
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], random_state=42)
    return clf
#
#clf = get_classifier(classifier_name, params)
#
# Classification
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

clf = BernoulliNB()
clf.fit(X_train, y_train)
#y_pred = clf.predict(X_valid)
#clf.score(X_train, y_train)
#acc = accuracy_score(y_valid, y_pred)
#st.write(f"accuracy = {round(acc*100, 2)}%")
#st.write(f"train score = {round(clf.score(X_train, y_train)*100, 2)}%")

income = st.number_input('Your income per month: $US', min_value=150, max_value=None, step=1)
coincome = st.number_input('Your spouse income per month: $US', min_value=0, max_value=None, step=1)
loanamount = st.number_input('How much would you like to loan? $US', min_value=0, max_value=None, step=1)
loanterm = st.number_input('Which term (months) would you like to tend? ',
                           min_value=36, max_value=480, step=12)

X_input = [[income, coincome, loanamount, loanterm]]
y_result = clf.predict(X_input)
st.write('Your result is...')
if y_result == 1:
    st.write('Approval.')
else:
    st.write('Disapproval!')

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
# acc = accuracy_score(y_test, y_pred)
# st.write(f"classifier = {classifier_name}")
# st.write(f"accuracy = {acc}")
#
# PLOT
# pca = PCA(2)
# X_projected = pca.fit_transform(X)
#
# x1 = X_projected[:, 0]
# x2 = X_projected[:, 1]
#
# fig = plt.figure()
# plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
# plt.xlabel("Principle Component 1")
# plt.ylabel("Principle Component 2")
# plt.colorbar()
# #plt.show()
# st.pyplot(fig)
