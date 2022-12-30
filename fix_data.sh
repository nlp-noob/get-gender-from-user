python ngender_and_keywords.py \
  --mode count_data_tags \
  --input_file_path "./data/per_json/dup_fixxedtrain0000_01.json" \
  --realtime_save_path "./data/per_json/tagged_train0000_01.json" \
  --save_every_step 20 \
  --fix_while_count True \
  --fix_no_label True \
  --fix_ambiguous True \
