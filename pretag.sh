python pretag.py \
    --model_name "./saved_model/pretag_01/" \
    --data_to_be_pretagged "data/splitted_data/no_label_orders.json" \
    --tagged_data_output_path "data/splitted_data/pretagged_no_label_orders.json" \
    --num_labels 3 \
    --max_line_num 20 \
    --device "cuda" \

