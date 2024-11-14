SPLIT="test"
DATASET_LIST="RoG-webqsp RoG-cwq"
BEAM_LIST="3" # "1 2 3 4 5"
MODEL_LIST="llama2-chat-hf"
PROMPT_LIST="prompts/llama2_predict.txt"
SORTED_BY="roberta"
set -- $PROMPT_LIST

for DATA_NAME in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        RULE_PATH=results/gen_rule_path/${DATA_NAME}/RoG/test/predictions_${N_BEAM}_False.jsonl
        for i in "${!MODEL_LIST[@]}"; do
        
            MODEL_NAME=${MODEL_LIST[$i]}
            PROMPT_PATH=${PROMPT_LIST[$i]}
            
            python src/qa_prediction/predict_answer.py \
                --model_name ${MODEL_NAME} \
                -d ${DATA_NAME} \
                --prompt_path ${PROMPT_PATH} \
                --add_rule \
                --rule_path ${RULE_PATH} \
                --sorted_by ${SORTED_BY}
        done
    done
done