# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
from sklearn import svm, metrics, preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from joblib import dump, load

class SVM_Arr:
    def __init__(self, kernel = "rbf", data_csv = "../Data/train.csv", model_file = "../Saved Models/svm_"):
        # Possible kernel functions: "rbf", "poly", "linear", "sigmoid"
        self.kernel = kernel
        self.data_csv = data_csv
        self.model_file = model_file + kernel + ".joblib"
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.svm, self.dataframe, self.X_train, self.y_train, self.X_test, self.y_test, self.y_pred = [], [], [], [], [], [], []
        self.selected_devices = [7, 8, 9, 10, 11]
        self.accuracy = ""

    def setDataCsv(data_csv):
        """ Sets the path to the data csv file """
        self.data_csv = data_csv

    def setModelFile(model_file):
        """ Sets the name of the file that stores
            the saved model """
        self.model_file = model_file

    def loadData(self):
        """ Loads data into self.dataset """
        self.dataframe = pd.read_csv(self.data_csv)

    def partitionData(self):
        """ Splits the data into training/testing """
        self.loadData()
        smaller_data = self.dataframe.sample(frac = 0.00001)
        smaller_data = smaller_data.loc[smaller_data['Device'].isin(self.selected_devices)]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(smaller_data[["T", "X", "Y", "Z"]], smaller_data.Device, test_size=0.3, random_state = 1)

    def fitScaler(self):
        """ Fit the scaler to training data """
        self.scaler.fit(self.X_train)

    def preprocessTrainData(self):
        """ preprocess training data """
        self.partitionData()
        self.fitScaler()
        self.X_train = self.scaler.transform(self.X_train)

    def preprocessTestData(self):
        """ preprocess training data """
        self.X_test = self.scaler.transform(self.X_test)

    def buildSVM(self):
        """ Builds the RNN. """
        if (self.kernel == "rbf"):
            self.svm = svm.SVC(C = 512, kernel = self.kernel, gamma = 2)
        else:
            self.svm = svm.SVC(kernel = self.kernel)

    def crossValidateKernel(self):
        self.preprocessTrainData()
        self.preprocessTestData()

        C_list = self.getCList()
        parameters = {"kernel":["linear", "rbf", "poly", "sigmoid"], "C": C_list}

        clf = RandomizedSearchCV(SVC(), parameters)
        clf.fit(self.X_train, self.y_train)
        print('score', clf.score(self.X_test, self.y_test))
        print(clf.best_params_)

    def crossValidateRBF(self):
        print("beginning preprocessing")
        self.preprocessTrainData()
        self.preprocessTestData()

        print("finished preprocessing")

        C_list = self.getCList()
        gamma_list = self.getGammaList()

        print("got lists")

        parameters = {"kernel": ["rbf"], "C": C_list, "gamma": gamma_list}
        
        print("made parameters")

        clf = RandomizedSearchCV(SVC(), param_distributions = parameters, random_state = 42)

        print("executed randomized search")

        clf.fit(self.X_train, self.y_train)

        print("fitted")

        print('score', clf.score(self.X_test, self.y_test))

        print(clf.best_params_)

    def getCList(self):
        """ Returns a list of values to cross-validate
        C on: {2^-5, 2^-3, ..., 2^15} """
        C_list = [2**(-5), 2**(-3), 2**(-1)]

        for i in range(1, 16, 2):
            C_list.append(2**i)

        return C_list

    def getGammaList(self):
        """ Returns a list of values to cross-validate
        Gamma on: {2^-15, 2^-13, ..., 2^3} """
        gamma_list = []
        for i in range(15, 0, -2):
            gamma_list.append(2**(-i))

        gamma_list.append(2**1)
        gamma_list.append(2**3)

        return gamma_list

    def fitSVM(self):
        """ Fits the RNN to the data. """
        self.preprocessTrainData()
        self.buildSVM()
        self.svm.fit(self.X_train, self.y_train)

    def predict(self):
        """ Gets predictions for the test data. """
        self.preprocessTestData()

        predictions = []

        for index, row in (self.X_test.loc[self.X_test['Device'] == device]).iterrows():
            predictions.append(self.svm.predict(row))

        self.y_pred = predictions
        # self.y_pred = self.svm.predict(self.X_test)

    def computeAccuracy(self):
        """ Computes the accuracy of the predictions. """
        self.accuracy = "Accuracy: " + str(metrics.accuracy_score(self.y_test, self.y_pred))

    def getResults(self):
        """ Returns the accuracy of the predictions. """
        self.computeAccuracy()
        return self.accuracy

    def visualize(self):
        """ Plot the results """
        # TODO

    def saveModel(self):
        """ Saves SVM to joblib file """
        dump(self.svm, self.model_file)

    def loadModel(self):
        self.preprocessTrainData()
        self.preprocessTestData()
        self.svm = load(self.model_file)

    def run(self):
        print("Importing the data.")
        print("Preprocessing the data.")
        print("Building and fitting the SVM.")
        self.fitSVM()

        print("Getting the predictions of the model.")
        self.predict()

        print(self.getResults())

        print("Saving model into " + self.model_file + ".")
        self.saveModel()

        print("Done.")