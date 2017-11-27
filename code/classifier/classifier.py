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
import csv
import scipy.sparse as sp
from sklearn.preprocessing import normalize


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
start_time = time.time()
logging.info(" ----------- Start Detector ------------")

train_data = pd.read_csv('../../data/combined_features.csv')


f = open("results.log", "a")
f.write("\n\n ---------------------------- BEGIN EXECUTION ---------------------------- ")

train_notes_features = [[] for i in range(1000)]

with open("../../data/combined_features.csv") as combined_features:
    csv_reader = csv.reader(combined_features)
    first_row = True
    for row in csv_reader:
        if first_row:
            first_row = False
            continue
        for i in range(7, 1007):
            train_notes_features[i - 7].append(row[i])

for i in range(len(train_notes_features)):
    train_notes_features[i] = np.array(train_notes_features[i])

training_notes_features_list = list(map(list, zip(*train_notes_features)))

train_age = np.array(train_data.age)
train_gender = np.array(train_data.gender)
train_saps2 = np.array(train_data.saps2)
train_oasis = np.array(train_data.oasis)
train_apsiii = np.array(train_data.apsiii)
train_output = np.array(train_data.expired)

training_data_features_list = [train_age[:30000], train_gender[:30000], train_saps2[:30000], train_oasis[:30000], train_apsiii
                               [:30000]]
training_output = train_output[:30000]

test_data_features_list = [train_age[30000:], train_gender[30000:], train_saps2[30000:], train_oasis[30000:], train_apsiii
                               [30000:]]
test_output = train_output[30000:]

training_data_features_list = list(map(list, zip(*training_data_features_list)))

for i in range(len(training_data_features_list)):
    training_data_features_list[i].extend(training_notes_features_list[i])

test_data_features_list = list(map(list, zip(*test_data_features_list)))


for i in range(len(test_data_features_list)):
    test_data_features_list[i].extend(training_notes_features_list[i])


training_data_features_list = normalize(training_data_features_list)
test_data_features_list = normalize(test_data_features_list)

# stacking selected features.
#final_training_data = sp.hstack(training_data_features_list)
#final_test_data = sp.hstack(test_data_features_list)

#Classifiers
svm = svm.SVC(C=0.3,kernel='linear',probability=True, verbose=True)
lr = LogisticRegression(random_state=1, verbose=True)
rfc = RandomForestClassifier(random_state=1, verbose=True)
dt = tree.DecisionTreeClassifier()

# Ensemble Classifier
eclf = VotingClassifier(estimators=[
    ('svm', svm), ('lr', lr), ('rfc', rfc)
], voting='soft', weights=[0.3, 0.3, 0.4])

#Testing LR
eclf = lr

# Training
eclf.fit(training_data_features_list, training_output)

# Accuracy score
accuracy_score = eclf.score(test_data_features_list, test_output)

score_str = "Accuracy Score = %s" % accuracy_score
f.write("\n" + score_str)

print(score_str)

predictions = eclf.predict_proba(test_data_features_list)
predict_ans = eclf.predict(test_data_features_list)
expired_predictions = predictions[:, 1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_output, expired_predictions)
auc_roc = auc(false_positive_rate, true_positive_rate)
auc_str = "AUC_ROC = %s" % auc_roc
f.write("\n" + auc_str)
print("Test data auc(roc curve) : ", auc_roc)

precision, recall, thresholds = precision_recall_curve(test_output, expired_predictions)
print("Test data auc(PR curve) : ", auc(recall, precision))
print("(PRF)macro : ", precision_recall_fscore_support(test_output, predict_ans, average='macro'))
print("(PRF)micro : ", precision_recall_fscore_support(test_output, predict_ans, average='micro'))
print("(PRF) : ", precision_recall_fscore_support(test_output, predict_ans, labels=[0, 1]))
f.write("\n\n ---------------------------- END EXECUTION ---------------------------- \n\n")
f.close()
logging.info(" ----------- End Detector ------------")
total_time = time.time() - start_time
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)
print("Program run time: %d:%02d:%02d" % (h, m, s))
