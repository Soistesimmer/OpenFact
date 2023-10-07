# KGC + OpenFact
This directory contains the code for implementing the KGC results in the paper.

## Dataset
We study on [CoDEx](https://github.com/tsafavi/codex). For training and validating, instances should be processed as a jsonline file:
> {"triple": (s, p, o), "kn": [...]}
> {"triple": (s, p, o), "kn": [...]}
> ...

where "kn" item is the returned knowledge piece from OpenFact DB.
To prepare the link prediction evaluation file, we first use the ComplEx results provided in [CoDEx](https://github.com/tsafavi/codex) and take their top 200 candidates for each instance and get the final predictions after reranking them using our trained models.

## Training
Run the following command to train the model. Activate "add_kn" as True to use OpenFact KN. 
>python run_reranker_concat.py \  
  --codex_dir data/codex-m \  
  --data_dir saved_data/data-reranker-codex-m-triple-classification \  
  --model_name_or_path bert-base-cased \  
  --do_train \  
  --do_eval \  
  --max_seq_length 512 \  
  --per_device_train_batch_size 32 \  
  --gradient_accumulation_steps 32  \  
  --learning_rate 5e-5 \  
  --num_train_epochs 4 \  
  --remove_unused_columns false \  
  --output_dir saved_data/bert-base-cased-codex-m-reranker-kn-top1 \  
  --overwrite_output_dir \  
  --use_segment_ids true \  
  --negative_weight 0.2 \  
  --top_1_only true \  
  --add_kn true \ 
  --save_steps 500

## Evaluation
First run the evaluation script
>python run_link_prediction.py \  
  --codex_dir harddisk/data/codex-m \  
  --data_dir harddisk/data/codex-m_link_prediction \  
  --data_file test.json \  
  --model_name_or_path saved_data/bert-base-cased-codex-m-reranker-kn-top1 \  
  --do_predict \  
  --max_seq_length 512 \  
  --per_device_eval_batch_size 1024 \  
  --gradient_accumulation_steps 16  \  
  --learning_rate 5e-5 \  
  --num_train_epochs 8 \  
  --remove_unused_columns false \  
  --output_dir saved_data/codex-m_results \  
  --output_file lp_test_predict_results.txt \  
  --overwrite_output_dir \  
  --use_segment_ids true \  
  --negative_weight 0.25 \  
  --add_kn true \  
  --save_steps 500

Then, get the final link prediction score (MRR) with script 'eval_link_prediction.py'