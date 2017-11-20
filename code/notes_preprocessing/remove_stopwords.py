import csv
import re

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
                split_note = re.split(r"\s|,|\.", note)
                #print(split_note)
                #print(len(split_note))
                non_stopword_list = []
                for word in split_note:
                    if word not in stopwords_set:
                        non_stopword_list.append(word)

                final_note = " ".join(non_stopword_list)
                row[10] = final_note
                csv_writer.writerow(row)
                #print(row)
                #break

if __name__ == "__main__":
    main()
