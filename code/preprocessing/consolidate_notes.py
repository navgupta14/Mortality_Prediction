from collections import defaultdict
import csv


def main():
    hadm_id_notes_dict = defaultdict(lambda: "")
    baseline_hadm_id_set = set()
    # generating list of stopwords + common words.
    with open("../../data/illegal_word_list") as illegal_words_file:
        illegal_words_string = illegal_words_file.readline()
        illegal_words = illegal_words_string.split()

    # noting hadms for which we have baseline features
    # Idea is that we want only those cases where we have both baseline and notes
    with open("../../data/baseline_features.csv") as baseline:
        csv_reader = csv.reader(baseline)
        for row in csv_reader:
            baseline_hadm_id_set.add(row[1])

    with open("../../data/final_notes.csv") as notes:
        csv_reader = csv.reader(notes)
        for row in csv_reader:
            hadm_id = row[2]
            note = row[10]
            # process only if we have baseline features for this admission id
            # also combining notes if there multiple notes for a single admission
            if hadm_id in baseline_hadm_id_set:
                hadm_id_notes_dict[hadm_id] += note + " "

    # writing <hadm_id, note> in consolidated_notes.csv.
    # note here is the final note after removing stopwords, common words etc
    with open("../../data/consolidated_notes.csv", "w") as consolidated_notes:
        csv_writer = csv.writer(consolidated_notes)
        for hadm_id, note in hadm_id_notes_dict.items():
            words = note.split()
            legit_words = []
            for word in words:
                if word not in illegal_words:
                    legit_words.append(word)
            note = " ".join(legit_words)
            csv_writer.writerow([hadm_id, note])


if __name__ == "__main__":
    main()
