import csv
import re
from collections import Counter
from collections import defaultdict
import operator

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
    output_csv_file = "../../data/final_notes_features.csv"
    input_csv_file = "../../data/raw_notes_features.csv"
    stopwords_set = set()

    #Populate the above set with stopwords from both Stopwords1.txt and Stopwords2.txt
    generate_stopwords(stopwords_set)
    first_line = True

    wordCountDict = defaultdict(lambda : 0)

    with open(input_csv_file) as csv_input:
        csv_reader = csv.reader(csv_input)
        with open(output_csv_file, "w") as csv_output:
            csv_writer = csv.writer(csv_output)
            for row in csv_reader:
                if (first_line):
                    csv_writer.writerow(row)
                    first_line = False
                    continue


                #print(row)
                note = row[10]

                note = re.sub(r"[^0-9a-zA-Z]+", " ", note)
                note = re.sub(r"\s", " ", note)
                split_note = re.split(r" ", note)

                non_stopword_list = []
                for word in split_note:
                    if len(word) > 3:
                        non_stopword_list.append(word)
                        wordCountDict[word] += 1

                final_note = " ".join(non_stopword_list)

                row[10] = final_note
                csv_writer.writerow(row)


    sorted_x = sorted(wordCountDict.items(), key=operator.itemgetter(1))[-100:]
    illegal_words = stopwords_set
    for tuple in sorted_x:
        illegal_words.add(tuple[0])

    illegal_words_string = " ".join(list(illegal_words))
    #string_to_write = " ".join(uncommon_non_stop_words)
    with open("../../data/illegal_word_list", "w") as illegal_words:
        illegal_words.write(illegal_words_string)

if __name__ == "__main__":
    main()
