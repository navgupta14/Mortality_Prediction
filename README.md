# Mortality_Prediction
cd code/preprocessing; python cleanup_notes.py ; echo "---cleaned--" ; python consolidate_notes.py ; echo "----consolidated--------"; cd ../classifier/; python lda_module.py; echo "------lda done-------"; cd ../preprocessing/; python merge_baseline_text_features.py; echo "merged baseline with notes features"; cd ../classifier/; python classifier.py; echo "all done"
