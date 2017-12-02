import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, precision_recall_fscore_support
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import logging
import time
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
start_time = time.time()
logging.info(" ----------- Start Detector ------------")

data = pd.read_csv('../../data/combined_features.csv')


f = open("results.log", "a")
f.write("\n\n ---------------------------- BEGIN EXECUTION ---------------------------- ")

notes_features = []
notes_features.append(np.array(data.Topic1))
notes_features.append(np.array(data.Topic2))
notes_features.append(np.array(data.Topic3))
notes_features.append(np.array(data.Topic4))
notes_features.append(np.array(data.Topic5))
notes_features.append(np.array(data.Topic6))
notes_features.append(np.array(data.Topic7))
notes_features.append(np.array(data.Topic8))
notes_features.append(np.array(data.Topic9))
notes_features.append(np.array(data.Topic10))
notes_features.append(np.array(data.Topic11))
notes_features.append(np.array(data.Topic12))
notes_features.append(np.array(data.Topic13))
notes_features.append(np.array(data.Topic14))
notes_features.append(np.array(data.Topic15))
notes_features.append(np.array(data.Topic16))
notes_features.append(np.array(data.Topic17))
notes_features.append(np.array(data.Topic18))
notes_features.append(np.array(data.Topic19))
notes_features.append(np.array(data.Topic20))
notes_features.append(np.array(data.Topic21))
notes_features.append(np.array(data.Topic22))
notes_features.append(np.array(data.Topic23))
notes_features.append(np.array(data.Topic24))
notes_features.append(np.array(data.Topic25))
notes_features.append(np.array(data.Topic26))
notes_features.append(np.array(data.Topic27))
notes_features.append(np.array(data.Topic28))
notes_features.append(np.array(data.Topic29))
notes_features.append(np.array(data.Topic30))
notes_features.append(np.array(data.Topic31))
notes_features.append(np.array(data.Topic32))
notes_features.append(np.array(data.Topic33))
notes_features.append(np.array(data.Topic34))
notes_features.append(np.array(data.Topic35))
notes_features.append(np.array(data.Topic36))
notes_features.append(np.array(data.Topic37))
notes_features.append(np.array(data.Topic38))
notes_features.append(np.array(data.Topic39))
notes_features.append(np.array(data.Topic40))
notes_features.append(np.array(data.Topic41))
notes_features.append(np.array(data.Topic42))
notes_features.append(np.array(data.Topic43))
notes_features.append(np.array(data.Topic44))
notes_features.append(np.array(data.Topic45))
notes_features.append(np.array(data.Topic46))
notes_features.append(np.array(data.Topic47))
notes_features.append(np.array(data.Topic48))
notes_features.append(np.array(data.Topic49))
notes_features.append(np.array(data.Topic50))


notes_features_list = list(map(list, zip(*notes_features)))

age = np.array(data.age)
gender = np.array(data.gender)
saps2 = np.array(data.saps2)
oasis = np.array(data.oasis)
apsiii = np.array(data.apsiii)

#Y
total_output = np.array(data.expired)

#X
total_data_features_list = [age, gender, saps2, oasis, apsiii]
#training_data_features_list = normalize(training_data_features_list)


total_data_features_list = list(map(list, zip(*total_data_features_list)))

for i in range(len(total_data_features_list)):
    total_data_features_list[i].extend(notes_features_list[i])


X = np.array(total_data_features_list)
Y = np.array(total_output)

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]


    #Classifiers
    svm = svm.SVC(C=0.3,kernel='linear',probability=True, verbose=True)
    lr = LogisticRegression(random_state=1, verbose=True)
    rfc = RandomForestClassifier(random_state=1, n_estimators=10, verbose=True)

    # Ensemble Classifier
    eclf = VotingClassifier(estimators=[
        ('svm', svm), ('lr', lr)
    ], voting='soft', weights=[0.5, 0.5])


    # Training
    eclf.fit(X_train, Y_train)

    # Accuracy score
    accuracy_score = eclf.score(X_test, Y_test)

    score_str = "Accuracy Score = %s" % accuracy_score
    f.write("\n" + score_str)

    print(score_str)

    predictions = eclf.predict_proba(X_test)
    predict_ans = eclf.predict(X_test)
    expired_predictions = predictions[:, 1]
    print(eclf.classes_)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, expired_predictions)
    auc_roc = auc(false_positive_rate, true_positive_rate)
    auc_str = "AUC_ROC = %s" % auc_roc
    f.write("\n" + auc_str)
    print("Test data auc(roc curve) : ", auc_roc)

    precision, recall, thresholds = precision_recall_curve(Y_test, expired_predictions)
    print("Test data auc(PR curve) : ", auc(recall, precision))
    print("(PRF)macro : ", precision_recall_fscore_support(Y_test, predict_ans, average='macro'))
    print("(PRF)micro : ", precision_recall_fscore_support(Y_test, predict_ans, average='micro'))
    print("(PRF) : ", precision_recall_fscore_support(Y_test, predict_ans, labels=[0, 1]))

f.write("\n\n ---------------------------- END EXECUTION ---------------------------- \n\n")
f.close()
logging.info(" ----------- End Detector ------------")
total_time = time.time() - start_time
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)
print("Program run time: %d:%02d:%02d" % (h, m, s))
