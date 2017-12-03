from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import csv
import sys
from time import time

csv.field_size_limit(sys.maxsize)
n_features = 1000
n_topics = 50
n_top_words = 20
hadm_id_list = []
notes_list = []


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


t0 = time()
print("Creating hadm and notes lists from consolidated_notes...")
with open("../../data/consolidated_notes.csv") as consolidated_notes:
    csv_reader = csv.reader(consolidated_notes)
    for row in csv_reader:
        if len(row) != 2:
            print("ERROR - consolidated_notes seem to be corrupt. Each row should have 2 elements(hadm, notes)")
        hadm_id_list.append(row[0])
        notes_list.append(row[1])
print("Time for creating hadm and notes lists = %0.3fs" % (time() - t0))

t0 = time()
print("Extracting tf features for LDA...")

# TODO - building vocab over all words, can see if shorter vocab helps like top 5k words
# getting term count for each note over vocab to use in LDA
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)

tf = tf_vectorizer.fit_transform(notes_list)
print("Time for tf fit transform = %0.3fs" % (time() - t0))

t0 = time()
print("Fitting LDA models with tf features")
lda = LatentDirichletAllocation(n_topics=n_topics,
                                learning_method='batch')

# getting topic_distributions for all the notes
topic_distributions = lda.fit_transform(tf)
print("Time for lda fit transform = %0.3fs" % (time() - t0))

t0 = time()
# text_features.csv -> for each hadm, stores {hadm, [topic distributions]}
with open("../../data/notes_features.csv", "w") as text_features:
    csv_writer = csv.writer(text_features)
    row_header = ["hadm_id"]
    for i in range(1, 51):
        row_header.append("Topic" + str(i))

    csv_writer.writerow(row_header)

    for i in range(len(hadm_id_list)):
        row = [hadm_id_list[i]]
        for dstr in topic_distributions[i]:
            row.append(dstr)
        csv_writer.writerow(row)

print("Time for writing LDA distributions = %0.3fs" % (time() - t0))

# for debug/info purpose
print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
