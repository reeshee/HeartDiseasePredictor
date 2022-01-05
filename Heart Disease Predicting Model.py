import sklearn
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn import linear_model, preprocessing
import pandas as pd

data = pd.read_csv("heart.csv")
print(data.head())

le = preprocessing.LabelEncoder()
Age = le.fit_transform(list(data["Age"]))
Sex = le.fit_transform(list(data["Sex"]))
ChestPainType = le.fit_transform((list(data["ChestPainType"])))
RestingBP = le.fit_transform(list(data["RestingBP"]))
Cholesterol = le.fit_transform(list(data["Cholesterol"]))
FastingBS = le.fit_transform(list(data["FastingBS"]))
RestingECG = le.fit_transform(list(data["RestingECG"]))
MaxHR = le.fit_transform(list(data["MaxHR"]))
ExerciseAngina = le.fit_transform(list(data["ExerciseAngina"]))
Oldpeak = le.fit_transform(list(data["Oldpeak"]))
ST_Slope = le.fit_transform(list(data["ST_Slope"]))
HeartDisease = le.fit_transform(list(data["HeartDisease"]))

X = list(zip(Age, Sex, ChestPainType, RestingBP, Cholesterol, RestingECG, FastingBS, MaxHR, ExerciseAngina, Oldpeak, ST_Slope))
Y = list(HeartDisease)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

##SVM
clf = svm.SVC(kernel='linear', C=3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = str(round(metrics.accuracy_score(y_pred, y_test)*100)) + "%"
print("SVM Model Accuracy: ", accuracy)

##KNN
k = 15
model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_train, y_train)
accuracy = str(round(model.score(x_test, y_test)*100)) + "%"
print("KNN Model Accuracy: ", accuracy)

##Linear Regression
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = str(round(linear.score(x_test, y_test)*100)) + "%"
print("Linear Regression Model Accuracy: ", accuracy)
