import csv
import re
from collections import defaultdict
import operator


# generating set of stop words using ONIX lists.
def generate_stopwords(stopwords_set):
    with open("../../data/Stopwords1.txt", "r") as stopwords1:
        stopwords1_lines = stopwords1.read().splitlines()
        for word in stopwords1_lines:
            if not word.isspace() and len(word) > 0:
                stopwords_set.add(word)
    with open("../../data/Stopwords2.txt", "r") as stopwords2:
        stopwords2_lines = stopwords2.read().splitlines()
        for word in stopwords2_lines:
            if not word.isspace() and len(word) > 0:
                stopwords_set.add(word)


def main():
    output_csv_file = "../../data/final_notes.csv"
    input_csv_file = "../../data/raw_notes.csv"
    stopwords_set = set()
    # Populate the above set with stopwords from both Stopwords1.txt and Stopwords2.txt
    generate_stopwords(stopwords_set)
    first_line = True

    wordCountDict = defaultdict(lambda: 0)

    with open(input_csv_file) as csv_input:
        csv_reader = csv.reader(csv_input)
        with open(output_csv_file, "w") as csv_output:
            csv_writer = csv.writer(csv_output)
            for row in csv_reader:
                # ignoring header line
                if first_line:
                    csv_writer.writerow(row)
                    first_line = False
                    continue

                note = row[10]
                # replacing non-alphanumeric by spaces
                note = re.sub(r"[^0-9a-zA-Z]+", " ", note)
                # replacing multiple spaces by a single space
                note = re.sub(r"\s", " ", note)
                words = re.split(r" ", note)

                words_list = []
                for word in words:
                    # ignoring words of len < 3
                    if len(word) > 3:
                        words_list.append(word)
                        wordCountDict[word] += 1

                # uniformly spaced, each word >= 3 size and no non-alphanumeric chars
                final_note = " ".join(words_list)

                row[10] = final_note
                csv_writer.writerow(row)

    # getting top 100 common words - prune them, so they may not carry any imp info
    most_common_words = sorted(wordCountDict.items(), key=operator.itemgetter(1))[-100:]
    illegal_words = stopwords_set
    for tupl in most_common_words:
        illegal_words.add(tupl[0])

    illegal_words_string = " ".join(list(illegal_words))
    with open("../../data/illegal_word_list", "w") as illegal_words:
        illegal_words.write(illegal_words_string)


if __name__ == "__main__":
    main()
