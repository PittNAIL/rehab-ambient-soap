python johnsnowlabs_eval.py \
--input_path ./cache/json_dataset.json \
--output_path ./cache/outputs/jsl_text2soap_predictions.json \
--text_col conversation \
--id_col idx \
--reference_col full_note \
--n_predict 2048 \
--batch_size 4 \
--temperature 0.0