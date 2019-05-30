from imageProcessModules import *
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score

feature_vector = loadObject(file_name = "feature_vector.cs484")
label_lookup = loadObject(file_name="label_lookup.cs484")
label_int = loadObject(file_name="label_int.cs484")
label_int = np.float64(label_int)


x_train, x_test, y_train, y_test = train_test_split(feature_vector, label_int, test_size = 0.2)
from sklearn.preprocessing import RobustScaler
#rbX = RobustScaler()
#x_train = rbX.fit_transform(x_train)

#rbY = RobustScaler()
#Y = rbY.fit_transform(Y)

svm_model = svm.SVC(kernel = 'rbf', gamma = 0.5, C = 500.0, max_iter = -1)

svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
svm_accuracy = accuracy_score(y_test, y_pred)

saveObject(svm_model,file_name="svm_model.cs484")

