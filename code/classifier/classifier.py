import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn import svm
import logging
import time
from sklearn.model_selection import cross_validate

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
start_time = time.time()
logging.info(" ----------- Start Classifier ------------")

# combined features has both structured and unstructured features with hadm in the form of csv
data = pd.read_csv('../../data/combined_features.csv')

f = open("results.log", "a")
f.write("\n\n ---------------------------- BEGIN EXECUTION ---------------------------- \n")

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

#Comment this block for only baseline features
f.write("Include LDA.\n")
logging.info("Include LDA")
for i in range(len(total_data_features_list)):
    total_data_features_list[i].extend(notes_features_list[i])

#Y
total_output = np.array(data.expired)

X = np.array(total_data_features_list)
Y = np.array(total_output)

#Classifiers
svmClassifier = svm.SVC(C=0.3,kernel='linear',probability=True, verbose=True)
lr = LogisticRegression(random_state=1, verbose=True)
rfc = RandomForestClassifier(random_state=1, n_estimators=10, verbose=True)

# Ensemble Classifier
eclf = VotingClassifier(estimators=[
    ('svmClassifier', svmClassifier), ('lr', lr), ('rfc', rfc)
], voting='soft', weights=[0.5, 0.5])


scoring_metrics = {'acc': 'accuracy',
                   'prec': 'precision',
                   'rec': 'recall',
                   'f1': 'f1',
                   'aucroc': 'roc_auc'}

#Uncomment the below line to only enable LR
eclf = lr

logging.info("Classifier information: " + str(eclf))
f.write("Classifier information: " + str(eclf) + "\n")

scores = cross_validate(eclf, X, Y, cv=5, scoring=scoring_metrics, n_jobs=-1)


#Print Scores for different metrics
acc_score = scores['test_acc']
prec_score = scores['test_prec']
recall_score = scores['test_rec']
f1_score = scores['test_f1']
aucroc_score = scores['test_aucroc']

acc_str = "Accuracy: " + str(acc_score) + "\n"
prec_str = "Precision: " + str(prec_score) + "\n"
recall_str = "Recall: " + str(recall_score) + "\n"
f1_str = "F1: " + str(f1_score) + "\n"
aucroc_str = "AUCROC: " + str(aucroc_score) + "\n"

f.write(acc_str)
f.write(prec_str)
f.write(recall_str)
f.write(f1_str)
f.write(aucroc_str)

logging.info(acc_str)
logging.info(prec_str)
logging.info(recall_str)
logging.info(f1_str)
logging.info(aucroc_str)

#Print max and average scores for different metrics
max_acc, avg_acc = str(max(acc_score)), str(np.mean(np.array(acc_score)))
max_prec, avg_prec = str(max(prec_score)), str(np.mean(np.array(prec_score)))
max_recall, avg_recall = str(max(recall_score)), str(np.mean(np.array(recall_score)))
max_f1, avg_f1 = str(max(f1_score)), str(np.mean(np.array(f1_score)))
max_aucroc, avg_aucroc = str(max(aucroc_score)), str(np.mean(np.array(aucroc_score)))

acc_str = "Max Accuracy is " + str(max_acc) + ". Average Accuracy is " + str(avg_acc) + "\n"
prec_str = "Max Precision is " + str(max_prec) + ". Average Precision is " + str(avg_prec) + "\n"
recall_str = "Max Recall is " + str(max_recall) + ". Average Recall is " + str(avg_recall) + "\n"
f1_str = "Max F1 is " + str(max_f1) + ". Average F1 is " + str(avg_f1) + "\n"
aucroc_str = "Max AUCROC is " + str(max_aucroc) + ". Average AUCROC is " + str(avg_aucroc) + "\n"

f.write(acc_str)
f.write(prec_str)
f.write(recall_str)
f.write(f1_str)
f.write(aucroc_str)

logging.info(acc_str)
logging.info(prec_str)
logging.info(recall_str)
logging.info(f1_str)
logging.info(aucroc_str)

f.write("\n\n ---------------------------- END EXECUTION ---------------------------- \n\n")
f.close()
logging.info(" ----------- End Detector ------------")
total_time = time.time() - start_time
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)
logging.info("Program run time: %d:%02d:%02d" % (h, m, s))
