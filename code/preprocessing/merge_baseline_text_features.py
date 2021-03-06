import csv
from collections import defaultdict

baseline_features_count = 7 # hadm_id, age, gender, saps2, aps3, oasis, expired_flag
total_features_count = 57 # 7 baseline + 50 topics distributions

# it merges baseline features and notes features(topic distributions)
def main():
    # stores all features(including output) with hadm_id as key
    combined_features_dict = defaultdict(lambda: [])

    with open("../../data/baseline_features.csv") as final_baseline_features:
        csv_reader = csv.reader(final_baseline_features)
        first_row = True
        for row in csv_reader:
            if first_row:
                first_row = False
                continue
            hadm_id = row[1]
            combined_features_dict[hadm_id].extend([hadm_id, row[2], row[3], row[4], row[5], row[6], row[7]])

    with open("../../data/notes_features.csv") as text_features:
        csv_reader = csv.reader(text_features)
        first_row = True
        for row in csv_reader:
            if first_row:
                first_row = False
                continue
            hadm_id = row[0]
            for i in range(1, 51):
                combined_features_dict[hadm_id].append(row[i])

    with open("../../data/combined_features.csv", "w") as combined_features:
        csv_writer = csv.writer(combined_features)
        header_row = ["hadm_id", "age", "gender", "saps2", "oasis", "apsiii", "expired"]
        topicStr = "Topic"
        for i in range(1, 51):
            header_row.append(topicStr + str(i))
        csv_writer.writerow(header_row)

        for key in combined_features_dict.keys():
            if len(combined_features_dict[key]) != total_features_count:
                continue
            csv_writer.writerow(combined_features_dict[key])


if __name__ == "__main__":
    main()
