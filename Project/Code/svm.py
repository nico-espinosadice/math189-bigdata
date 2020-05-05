# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
from sklearn import svm, metrics, preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from joblib import dump, load

class SVM:
    def __init__(self, num_devices = 5, kernel = "rbf", data_csv = "../Data/train.csv", model_file = "../Saved Models/svm_"):
        # Possible kernel functions: "rbf", "poly", "linear", "sigmoid"
        self.kernel = kernel
        self.data_csv = data_csv
        self.model_file = model_file + kernel + ".joblib"
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.svm, self.dataframe, self.X_train, self.y_train, self.X_test, self.y_test, self.y_pred = [], [], [], [], [], [], []
        self.accuracy = ""
        self.kernel_cv, self.rbf_cv = [], []
        self.devices = [7, 8, 9, 12, 23, 25, 26, 27, 33, 37, 39, 45, 47, 51, 52, 57, 58, 65, 67, 68, 70, 71, 73, 74, 75, 78, 79, 81, 87, 89, 90, 91, 92, 94, 95, 96, 99, 104, 105, 108, 110, 111, 116, 117, 120, 122, 124, 126, 127, 129, 134, 137, 142, 145, 148, 149, 152, 156, 157, 158, 159, 162, 163, 168, 169, 174, 175, 177, 183, 187, 188, 189, 190, 194, 196, 204, 206, 207, 211, 213, 216, 219, 222, 224, 229, 232, 233, 234, 236, 237, 239, 240, 261, 263, 268, 269, 270, 271, 273, 274, 275, 277, 281, 282, 283, 284, 285, 289, 290, 291, 294, 296, 297, 298, 299, 302, 306, 309, 312, 313, 314, 323, 325, 333, 335, 338, 341, 343, 344, 345, 350, 360, 361, 366, 369, 370, 371, 376, 378, 381, 390, 394, 398, 399, 401, 404, 411, 412, 413, 415, 417, 421, 422, 423, 425, 433, 438, 447, 448, 455, 461, 463, 466, 471, 473, 477, 478, 479, 482, 485, 486, 487, 491, 492, 494, 501, 503, 505, 507, 509, 514, 515, 518, 520, 523, 524, 528, 531, 533, 534, 536, 537, 539, 547, 550, 552, 553, 554, 556, 557, 562, 568, 571, 573, 574, 575, 577, 579, 580, 581, 583, 589, 593, 594, 595, 596, 600, 601, 607, 610, 611, 612, 613, 614, 617, 621, 622, 626, 627, 629, 632, 634, 638, 640, 642, 643, 646, 647, 650, 653, 656, 658, 660, 661, 663, 664, 665, 666, 667, 669, 670, 671, 674, 675, 676, 678, 679, 680, 681, 682, 683, 684, 687, 688, 690, 691, 692, 694, 696, 698, 699, 700, 703, 705, 706, 709, 710, 711, 713, 714, 720, 721, 722, 727, 728, 729, 730, 732, 735, 736, 738, 739, 745, 746, 750, 751, 754, 755, 757, 761, 762, 763, 764, 768, 770, 774, 776, 781, 782, 784, 789, 792, 793, 795, 801, 802, 804, 805, 806, 810, 812, 814, 818, 820, 823, 824, 827, 834, 836, 838, 839, 841, 842, 846, 847, 848, 854, 857, 859, 860, 862, 864, 868, 870, 871, 877, 880, 882, 883, 887, 890, 895, 897, 900, 911, 912, 913, 919, 933, 941, 943, 945, 953, 955, 956, 967, 973, 977, 979, 983, 987, 991, 992, 996, 997, 998, 1000, 1006, 1015, 1017, 1027, 1029, 1031, 1033, 1035, 1036, 1037]
        self.selected_devices = []

        for i in range(num_devices):
            self.selected_devices.append(self.devices[i])

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
        smaller_data = self.dataframe.sample(frac = 0.1)
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
        """ Cross-validates kernel function and C hyperparameters """
        self.preprocessTrainData()
        self.preprocessTestData()

        C_list = self.getCList()
        parameters = {"kernel":["linear", "rbf", "poly", "sigmoid"], "C": C_list}

        self.kernel_cv = RandomizedSearchCV(SVC(), parameters)
        self.kernel_cv.fit(self.X_train, self.y_train)
        print('score', self.kernel_cv.score(self.X_test, self.y_test))
        print(self.kernel_cv.best_params_)

    def crossValidateRBF(self):
        """ Cross-validates the gamma and C hyperparameters using an RBF kernel """
        self.preprocessTrainData()
        self.preprocessTestData()

        C_list = self.getCList()
        gamma_list = self.getGammaList()

        parameters = {"kernel": ["rbf"], "C": C_list, "gamma": gamma_list}
        self.rbf_cv = RandomizedSearchCV(SVC(), param_distributions = parameters, random_state = 42)
        self.rbf_cv.fit(self.X_train, self.y_train)
        print('score', self.rbf_cv.score(self.X_test, self.y_test))
        print(self.rbf_cv.best_params_)

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
        self.y_pred = self.svm.predict(self.X_test)

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
        """ Loads a model from a saved file """
        self.preprocessTrainData()
        self.preprocessTestData()
        self.svm = load(self.model_file)

    def run(self):
        """ Preprocesses data; builds and trains the SVM; outputs results """
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

def getAccuracies():
    """ Measures the accuracy of SVMs tested on varying numbers of devices """
    accuracies = []
    for i in range(5, 21, 5):
        new_svm = SVM(num_devices = i)
        new_svm.run()

        accuracies.append([i, new_svm.getResults()])

    for i in range(30, 71, 10):
        new_svm = SVM(num_devices = i)
        new_svm.run()

        accuracies.append([i, new_svm.getResults()])

    df = pd.DataFrame(accuracies, columns = ['num_devices', 'Accuracy'])
    df.to_excel("accuracies.xlsx")