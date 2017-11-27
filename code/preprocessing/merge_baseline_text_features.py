import csv
from collections import defaultdict

def main():
    combined_features_dict = defaultdict(lambda : [])

    with open("../../data/final_baseline_features.csv") as final_baseline_features:
        csv_reader = csv.reader(final_baseline_features)
        first_row = True
        for row in csv_reader:
            if first_row:
                first_row = False
                continue

            hadm_id = row[1]
            combined_features_dict[hadm_id].extend([hadm_id, row[2], row[3], row[4], row[5], row[6], row[7]])

    with open("../../data/lda_words.csv") as text_features:
        csv_reader = csv.reader(text_features)
        for row in csv_reader:
            hadm_id = row[0]
            if (len(combined_features_dict[hadm_id]) == 7):
                for i in range(1, 1001):
                    combined_features_dict[hadm_id].append(row[i])

    with open("../../data/combined_features.csv", "w") as combined_features:
        csv_writer = csv.writer(combined_features)
        header_row = ["hadm_id","age","gender","saps2","oasis","apsiii","expired"]
        topicStr = "Word"
        for i in range(1, 1001):
            header_row.append(topicStr+str(i))

        csv_writer.writerow(header_row)

        for key in combined_features_dict.keys():
            row = []
            row.extend(combined_features_dict[key])
            csv_writer.writerow(row)

if __name__ == "__main__":
    main()