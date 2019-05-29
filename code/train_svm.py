from imageProcessModules import *
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score

feature_vector = loadObject(file_name = "feature_vector.cs484")
label_lookup = loadObject(file_name="label_lookup.cs484")
label_int = loadObject(file_name="label_int.cs484")


svm_model = svm.SVC(C=500.0,
                    cache_size= 100,
                    class_weight= None,
                    coef0= 0.0,
                    decision_function_shape= 'ovr',
                    degree= 10,
                    gamma= 0.5,
                    kernel= 'rbf',
                    max_iter= -1,
                    probability= False,
                    random_state= 1,
                    shrinking= True,
                    tol= 0.001,
                    verbose= False)


svm_model.fit(feature_vector, label_int)
svm_accuracy = accuracy_score(label_int, svm_model.predict(feature_vector))

y_pred = svm_model.predict(feature_vector)
confusion_matrix = metrics.confusion_matrix(label_int, y_pred)

saveObject(svm_model,file_name="svm_model.cs484")
