dataset_name="Amazon"
batch_size=1
retrieval_type="Topk"
ddp="True"

# model_list=("gpt2-large" "gpt-neo-2.7B")
# sbert_list=("all-mpnet-base-v2" "paraphrase-mpnet-base-v2")
# ice_num_list=(1 2 4 8 16 32)

model_list=("gpt-neo-2.7B")
sbert_list=("paraphrase-mpnet-base-v2")
ice_num_list=(8)

for ice_num in "${ice_num_list[@]}"; do
  for model in "${model_list[@]}"; do
    for sbert in "${sbert_list[@]}"; do
        echo -e "\n\n-${dataset_name}-${model}-${sbert}-${retrieval_type}-${ice_num}-\n\n"
        accelerate launch case_test.py \
           --dataset_name ${dataset_name} \
           --model_name ${model} \
           --sentence_transformers_name ${sbert} \
           --ice_num ${ice_num} \
           --batch_size ${batch_size} \
           --retriever_type ${retrieval_type} \
           --output_json_filepath './icl_inference_output_ascending' \
           --ascending_order \
           --ddp \
           --debug
    done
  done
done