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
train_data = pd.read_csv('../../data/combined_features.csv')

train_notes_features = []
train_notes_features.append(np.array(train_data.Topic1))
train_notes_features.append(np.array(train_data.Topic2))
train_notes_features.append(np.array(train_data.Topic3))
train_notes_features.append(np.array(train_data.Topic4))
train_notes_features.append(np.array(train_data.Topic5))
train_notes_features.append(np.array(train_data.Topic6))
train_notes_features.append(np.array(train_data.Topic7))
train_notes_features.append(np.array(train_data.Topic8))
train_notes_features.append(np.array(train_data.Topic9))
train_notes_features.append(np.array(train_data.Topic10))
train_notes_features.append(np.array(train_data.Topic11))
train_notes_features.append(np.array(train_data.Topic12))
train_notes_features.append(np.array(train_data.Topic13))
train_notes_features.append(np.array(train_data.Topic14))
train_notes_features.append(np.array(train_data.Topic15))
train_notes_features.append(np.array(train_data.Topic16))
train_notes_features.append(np.array(train_data.Topic17))
train_notes_features.append(np.array(train_data.Topic18))
train_notes_features.append(np.array(train_data.Topic19))
train_notes_features.append(np.array(train_data.Topic20))
train_notes_features.append(np.array(train_data.Topic21))
train_notes_features.append(np.array(train_data.Topic22))
train_notes_features.append(np.array(train_data.Topic23))
train_notes_features.append(np.array(train_data.Topic24))
train_notes_features.append(np.array(train_data.Topic25))
train_notes_features.append(np.array(train_data.Topic26))
train_notes_features.append(np.array(train_data.Topic27))
train_notes_features.append(np.array(train_data.Topic28))
train_notes_features.append(np.array(train_data.Topic29))
train_notes_features.append(np.array(train_data.Topic30))
train_notes_features.append(np.array(train_data.Topic31))
train_notes_features.append(np.array(train_data.Topic32))
train_notes_features.append(np.array(train_data.Topic33))
train_notes_features.append(np.array(train_data.Topic34))
train_notes_features.append(np.array(train_data.Topic35))
train_notes_features.append(np.array(train_data.Topic36))
train_notes_features.append(np.array(train_data.Topic37))
train_notes_features.append(np.array(train_data.Topic38))
train_notes_features.append(np.array(train_data.Topic39))
train_notes_features.append(np.array(train_data.Topic40))
train_notes_features.append(np.array(train_data.Topic41))
train_notes_features.append(np.array(train_data.Topic42))
train_notes_features.append(np.array(train_data.Topic43))
train_notes_features.append(np.array(train_data.Topic44))
train_notes_features.append(np.array(train_data.Topic45))
train_notes_features.append(np.array(train_data.Topic46))
train_notes_features.append(np.array(train_data.Topic47))
train_notes_features.append(np.array(train_data.Topic48))
train_notes_features.append(np.array(train_data.Topic49))
train_notes_features.append(np.array(train_data.Topic50))

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
