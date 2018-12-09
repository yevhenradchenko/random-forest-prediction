import csv
import random
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE


class CSV_Data:

    def __init__(self, file_name):
        self.file_name = file_name
        self.csv_header = []
        self.data_matrix = np.array([])

    def read_data(self):

        print("Reading started...")

        rows_readed = 0

        with open(self.file_name, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            header_skipped = False
            tmp = []
            for row in csv_reader:

                l = len(row)

                if header_skipped == False:
                    self.csv_header = row
                    header_skipped = True
                else:
                    if row[1] == "yes":
                        row[1] = 1
                    else:
                        row[1] = 0
                    for i in range(2, l):
                        row[i] = float(row[i])
                    tmp.append(row[1:])

            self.data_matrix = np.array(tmp)

        print("Reading finished. Records number: %d" % self.data_matrix.shape[0])

    def get_data_matrix(self):
        return self.data_matrix

    def get_y_vector(self):
        return self.data_matrix[0:, 0]

    def get_X_matrix(self):
        return self.data_matrix[0:, 1:]

    def get_header(self):
        return self.csv_header


def get_positive_class_objects(m):
    l = []
    for row in m:
        if row[0] == 1:
            l.append(row[0:])
    return np.array(l)


def get_negative_class_objects(m):
    l = []
    for row in m:
        if row[0] == 0:
            l.append(row[0:])
    return np.array(l)


def select_n_random_elements(l, n):
    np.random.shuffle(l)
    return l[0:n], l[n:]


def extract_y_X(m):
    return m[0:, 0], m[0:, 1:]


data = CSV_Data("data.csv")
data.read_data()

acc_train = []
accs_test = []

precs = []
recalls = []
f1s = []

y_X_data = data.get_data_matrix()
del data


print("Removing examples started")

data_dict = {}

lst = []

for row in y_X_data:
    s = np.sum(row[1:])
    if s not in data_dict and row[0] == 0:
        data_dict[s] = row
        lst.append(row)
    elif row[0] == 1:
        lst.append(row)

y_X_data = np.array(lst)

del data_dict


print("Removing examples finished")

y_data, X_data = extract_y_X(y_X_data)

print("Searching feature importance started")
clf = RandomForestClassifier(n_estimators=150, max_depth=15)

clf.fit(X_data, y_data)
print("Searching feature importance finished")
print("\n-- Feature importance: %r" % clf.feature_importances_)

print("\nSelecting started.")
X_data = SelectFromModel(clf, prefit=True, threshold="0.75*median").transform(X_data)  # medial - 0.75-1
print("Selecting finished.")

print("--X_data shape(after feature selection): (%d, %d)" % (X_data.shape[0], X_data.shape[1]))

categorical_f = []

for i in range(X_data.shape[1]):

    lst = len(np.unique(X_data[:, i]))

    if lst <= 8 and lst > 2:
        categorical_f.append(i)

print("Encoding started.")
X_data = OneHotEncoder(categorical_features=categorical_f).fit_transform(X_data, y_data).toarray()
print("Encoding finished.")

print("--X_data shape(after encoding): (%d, %d)" % (X_data.shape[0], X_data.shape[1]))

# print("SMOTE started")
# sm = SMOTE(kind='regular', ratio=0.1, k_neighbors=10, random_state=42)
# X_data, y_data = sm.fit_sample(X_data, y_data)
# print("SMOTE finished")

y_data = y_data.reshape((y_data.shape[0], 1))

y_X_data = np.hstack((y_data, X_data))

print("--y_X_data shape: (%d, %d)" % (y_X_data.shape[0], y_X_data.shape[1]))

num_of_iterations = 10

for i in range(num_of_iterations):

    print("Experiment %d" % (i + 1))
    print("\t\tGenerating train and test data...")

    pos_objects = get_positive_class_objects(y_X_data)
    neg_objects = get_negative_class_objects(y_X_data)

    pos_objects_train, pos_objects_test = select_n_random_elements(pos_objects, len(pos_objects) * 2 // 3)
    neg_objects_train, neg_objects_test = select_n_random_elements(neg_objects, len(neg_objects) * 2 // 3)

    train_data = np.vstack((pos_objects_train, neg_objects_train))
    test_data = np.vstack((pos_objects_test, neg_objects_test))

    np.random.shuffle(train_data)

    y_train, X_train = extract_y_X(train_data)

    y_test, X_test = extract_y_X(test_data)

    y_test_pos, X_test_pos = extract_y_X(get_positive_class_objects(test_data))
    y_test_neg, X_test_neg = extract_y_X(get_negative_class_objects(test_data))

    print("\t\tGenerating finished.")
    print("\t\t -- Train Examples: %d" % X_train.shape[0])
    print("\t\t -- Test Examples: %d" % X_test.shape[0])

    print("\t\tTraining started")

    clf = RandomForestClassifier(n_estimators=150, max_depth=15, class_weight={1: 1, 0: 1})  # 0.25-1

    clf.fit(X_train, y_train)
    print("\t\tTraining finished")

    print("\t\tTesting started")
    prediction = clf.predict(X_test)
    accuracy_train = clf.score(X_train, y_train)
    accuracy_test = accuracy_score(y_test, prediction)
    accuracy_test_pos = clf.score(X_test_pos, y_test_pos)
    accuracy_test_neg = clf.score(X_test_neg, y_test_neg)
    print("\t\tTesting finished")

    prec = precision_score(y_test, prediction) * 100
    rcl = recall_score(y_test, prediction) * 100
    f1 = f1_score(y_test, prediction) * 100

    print("\t\tAccuracy - train: %f" % (accuracy_train * 100))
    print("\t\tAccuracy - test: %f" % (accuracy_test * 100))
    print("\t\tAccuracy - test positive: %f" % (accuracy_test_pos * 100))
    print("\t\tAccuracy - test negative: %f" % (accuracy_test_neg * 100))
    print("\t\tPrecision - test: %f" % prec)
    print("\t\tRecall - test: %f" % rcl)
    print("\t\tF1 - test: %f" % f1)

    acc_train.append(accuracy_train * 100)
    accs_test.append(accuracy_test * 100)
    precs.append(prec)
    recalls.append(rcl)
    f1s.append(f1)

accuracy_train_average = np.average(acc_train)
accuracy_test_average = np.average(accs_test)

print("Average Accuracy - train: %f" % (accuracy_train_average))
print("Average Accuracy - test: %f" % (accuracy_test_average))
print("Average Precision - train: %f" % (np.average(precs)))
print("Average Recall - test: %f" % (np.average(recalls)))
print("Average F1 - test: %f" % (np.average(f1s)))

plt.plot(range(1, len(accs_test) + 1), accs_test, "ro")
plt.plot(range(1, len(acc_train) + 1), acc_train, "go")
plt.xlim(0, num_of_iterations + 2)
plt.ylim(90, 101)

plt.axhline(accuracy_test_average, xmax=(num_of_iterations + 2), color='r', linestyle='dashed', linewidth=2)
plt.axhline(accuracy_train_average, xmax=(num_of_iterations + 2), color='g', linestyle='dashed', linewidth=2)

plt.show()
