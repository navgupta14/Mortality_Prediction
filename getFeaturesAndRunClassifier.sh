## Assuming that data has already been extracted and relevant csv files are in place.
# Preprocess the notes
cd code/preprocessing
python cleanup_notes.py
echo "---cleaned--"
# Consolidate the notes
python consolidate_notes.py
echo "----consolidated--------"
# Get text features (topic distributions)
cd ../classifier/;
python lda_module.py
echo "------lda done-------"
# Merge structured features with unstructured features
cd ../preprocessing/
python merge_baseline_text_features.py
echo "merged baseline with notes features"
# Run classifier
cd ../classifier/
python classifier.py
echo "all done"
