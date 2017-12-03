import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn import svm
import logging
import time
from sklearn.model_selection import KFold

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
start_time = time.time()
logging.info(" ----------- Start Classifier ------------")

# combined features has both structured and unstructured features with hadm in the form of csv
data = pd.read_csv('../../data/combined_features.csv')

f = open("results.log", "a")
f.write("\n\n ---------------------------- BEGIN EXECUTION ---------------------------- ")

# 50 lda topics distribution from notes.
notes_features = [np.array(data.Topic1), np.array(data.Topic2), np.array(data.Topic3), np.array(data.Topic4),
                  np.array(data.Topic5), np.array(data.Topic6), np.array(data.Topic7), np.array(data.Topic8),
                  np.array(data.Topic9), np.array(data.Topic10), np.array(data.Topic11), np.array(data.Topic12),
                  np.array(data.Topic13), np.array(data.Topic14), np.array(data.Topic15), np.array(data.Topic16),
                  np.array(data.Topic17), np.array(data.Topic18), np.array(data.Topic19), np.array(data.Topic20),
                  np.array(data.Topic21), np.array(data.Topic22), np.array(data.Topic23), np.array(data.Topic24),
                  np.array(data.Topic25), np.array(data.Topic26), np.array(data.Topic27), np.array(data.Topic28),
                  np.array(data.Topic29), np.array(data.Topic30), np.array(data.Topic31), np.array(data.Topic32),
                  np.array(data.Topic33), np.array(data.Topic34), np.array(data.Topic35), np.array(data.Topic36),
                  np.array(data.Topic37), np.array(data.Topic38), np.array(data.Topic39), np.array(data.Topic40),
                  np.array(data.Topic41), np.array(data.Topic42), np.array(data.Topic43), np.array(data.Topic44),
                  np.array(data.Topic45), np.array(data.Topic46), np.array(data.Topic47), np.array(data.Topic48),
                  np.array(data.Topic49), np.array(data.Topic50)]

# stacking them horizontally
notes_features_list = list(map(list, zip(*notes_features)))

## structured features
age = np.array(data.age)
gender = np.array(data.gender)
saps2 = np.array(data.saps2)
oasis = np.array(data.oasis)
apsiii = np.array(data.apsiii)


#X -- combining structured and unstructured features
total_data_features_list = [age, gender, saps2, oasis, apsiii]
total_data_features_list = list(map(list, zip(*total_data_features_list)))
for i in range(len(total_data_features_list)):
    total_data_features_list[i].extend(notes_features_list[i])

#Y
total_output = np.array(data.expired)

X = np.array(total_data_features_list)
Y = np.array(total_output)

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    #Classifiers
    svmClassifier = svm.SVC(C=0.3,kernel='linear',probability=True, verbose=True)
    lr = LogisticRegression(random_state=1, verbose=True)
    rfc = RandomForestClassifier(random_state=1, n_estimators=10, verbose=True)

    # Ensemble Classifier
    eclf = VotingClassifier(estimators=[
        ('svmClassifier', svmClassifier), ('lr', lr)
    ], voting='soft', weights=[0.5, 0.5])

    # Training
    eclf.fit(X_train, Y_train)

    # Accuracy score
    accuracy_score = eclf.score(X_test, Y_test)

    score_str = "Accuracy Score = %s" % accuracy_score
    f.write("\n" + score_str)
    logging.info(score_str)

    prediction_probs = eclf.predict_proba(X_test)
    expired_predictions = prediction_probs[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, expired_predictions)
    auc_roc = auc(false_positive_rate, true_positive_rate)
    auc_str = "AUC_ROC = %s" % auc_roc
    f.write("\n" + auc_str)
    logging.info(auc_str)

    predictions = eclf.predict(X_test)
    logging.info(classification_report(Y_test, predictions, labels=[0, 1], target_names=["class living", "class expired"]))

f.write("\n\n ---------------------------- END EXECUTION ---------------------------- \n\n")
f.close()
logging.info(" ----------- End Detector ------------")
total_time = time.time() - start_time
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)
logging.info("Program run time: %d:%02d:%02d" % (h, m, s))
