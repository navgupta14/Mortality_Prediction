from collections import defaultdict
import csv

def main():
    hadm_id_notes_dict = defaultdict(lambda :"")
    baseline_hadm_id_set = set()
    legal_words = []
    with open("../../data/legal_word_list") as legal_words_file:
	legal_words_string = legal_words_string.readline()
	legal_words = legal_words_string.split()

    with open("../../data/final_baseline_features.csv") as baseline:
        csv_reader = csv.reader(baseline)
        for row in csv_reader:
            baseline_hadm_id_set.add(row[1])

    with open("../../data/final_notes_features.csv") as notes:
        csv_reader = csv.reader(notes)
        for row in csv_reader:
            hadm_id = row[2]
            note = row[10]
            if (hadm_id in baseline_hadm_id_set):
                hadm_id_notes_dict[hadm_id] += note + " "
    
    with open("../../data/consolidated_notes.csv", "w") as consolidated_notes:
        csv_writer = csv.writer(consolidated_notes)
        for key, value in hadm_id_notes_dict.items():
            	split_notes = value.split()
                legal_notes = []
		for w in split_notes:
			if w in legal_words:
				legal_notes.append(w)

		value = " ".join(legal_notes)
		csv_writer.writerow([key, value])

if __name__ == "__main__":
    main()
