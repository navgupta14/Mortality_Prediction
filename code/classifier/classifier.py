import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, precision_recall_fscore_support
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import logging
import time
import scipy.sparse as sp

start_time = time.time()
train_data = pd.read_csv('../../data/final_baseline_features.csv')

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
test_data_features_list = list(map(list, zip(*test_data_features_list)))
# stacking selected features.
#final_training_data = sp.hstack(training_data_features_list)
#final_test_data = sp.hstack(test_data_features_list)

#Classifiers
svm = svm.SVC(C=0.3,kernel='linear',probability=True, verbose=True)
lr = LogisticRegression(random_state=1, verbose=True)

# Ensemble Classifier
eclf = VotingClassifier(estimators=[
    ('svm', svm), ('lr', lr)
], voting='soft', weights=[0.6, 0.4])

# Training
#eclf.fit(training_data_features_list, training_output)
svm.fit(training_data_features_list, training_output)
# Accuracy score
accuracy_score = svm.score(test_data_features_list, test_output)
score_str = "Score = %s" % accuracy_score

print(score_str)

