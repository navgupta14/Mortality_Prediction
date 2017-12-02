import csv
import re
from collections import Counter

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
    uncommon_non_stop_words = set()
    all_notes_blob = ""
    first_line = True
    all_notes_blob_split = []
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
                #note = re.sub(r"\d+", "", note)
                #note = re.sub(r"_", "", note)
                note = re.sub(r"[^0-9a-zA-Z]+", " ", note)
                note = re.sub(r"\s", " ", note)
                split_note = re.split(r" ", note)
		
                #print(split_note)
                #print(len(split_note))
                non_stopword_list = []
                for word in split_note:
                    if len(word) > 3:
                        non_stopword_list.append(word)

                all_notes_blob_split.extend(non_stopword_list)
                final_note = " ".join(non_stopword_list)
                #all_notes_blob += final_note + " "
                row[10] = final_note
                csv_writer.writerow(row)
                #print(row)
                #break

    #all_notes_blob_split = re.split(r" ", all_notes_blob)
    word_counts = Counter(all_notes_blob_split)
    
    print("Word Count Size = ", len(word_counts.most_common())) 
    print(word_counts.most_common()) 
    current_count = 0
    for w in word_counts.most_common():
        if (current_count < 100):
            current_count += 1
            continue


        if w[0] not in stopwords_set:
            uncommon_non_stop_words.add(w[0])
    
    print(len(uncommon_non_stop_words))
    string_to_write = " ".join(uncommon_non_stop_words)
    with open("../../data/legal_word_list", "w") as legal_words:
        legal_words.write(string_to_write)

if __name__ == "__main__":
    main()
