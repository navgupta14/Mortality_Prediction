#http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import csv
import sys
from collections import defaultdict
from time import time

csv.field_size_limit(sys.maxsize)
final_words = []
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        final_words.extend([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print final_words
    return

n_features = 2000
n_topics = 50
n_top_words = 20

hadm_id_list = []
notes_list = []

t0 = time()
print("Creating hadm and notes lists...")
with open("../../data/consolidated_notes.csv") as consolidated_notes:
    csv_reader = csv.reader(consolidated_notes)
    for row in csv_reader:
        if (len(row) != 2):
            print("ERROR!!!!!")
        hadm_id_list.append(row[0])
        notes_list.append(row[1])
print("Time for creating hadm and notes lists = %0.3fs" % (time() - t0))


t0 = time()
print("Extracting tf features for LDA...")


tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')


tf = tf_vectorizer.fit_transform(notes_list)
print("Time for tf fit transform = %0.3fs" % (time() - t0))


t0 = time()
print("Fitting LDA models with tf features")
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

predict = lda.fit_transform(tf)
print("Time for lda fit transform = %0.3fs" % (time() - t0))

t0 = time()
with open("../../data/text_features.csv", "w") as text_features:
    csv_writer = csv.writer(text_features)
    row_header = ["hadm_id"]

    for i in range(1, 51):
        row_header.append("Topic"+str(i))

    csv_writer.writerow(row_header)

    for i in range(len(hadm_id_list)):
        row = []
        row.append(hadm_id_list[i])
        for distr in predict[i]:
            row.append(distr)
        csv_writer.writerow(row)

print("Time for writing LDA distributions = %0.3fs" % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

with open("../../data/lda_words.csv", "w") as lda_words:
    csv_writer = csv.writer(lda_words)
    with open("../../data/consolidated_notes.csv") as consolidated_notes:
        csv_reader = csv.reader(consolidated_notes)
        for row in csv_reader:
            hadm = row[0]
            notes = row[1]
            row_vec = [hadm]
            notes_words = notes.split()
            dict = defaultdict(lambda : 0)
            for word in notes_words:
                dict[word] += 1
            for word in final_words:
                row_vec.append(dict[word])
            csv_writer.writerow(row_vec)
