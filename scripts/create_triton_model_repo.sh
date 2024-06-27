#!/bin/bash

# Check if the number of arguments is 3
if [ "$#" -ne 3 ]; then
    echo "Error: Please provide exactly 3 arguments."
    echo "Usage: $0 <tokenizer_dir> <llm_engine_dir> <visual_engine_dir>"
    exit 1
else
    echo "Number of arguments is correct."
    echo "Arguments passed: $@"
fi

triton_model_dir=./triton_model_repo
tokenizer_dir=$1

if [ ! -d ${triton_model_dir} ]; then
  cp -r tensorrtllm_backend/all_models/internlm_xcomposer2 ${triton_model_dir}
  echo "${triton_model_dir} has been created."
else
  rm -rf ${triton_model_dir}
  cp -r tensorrtllm_backend/all_models/internlm_xcomposer2 ${triton_model_dir}
  echo "${triton_model_dir} already exists but we remove and re-created it!"
fi

fill_template=tensorrtllm_backend/tools/fill_template.py

triton_max_batch_size=32
kv_cache_free_gpu_mem_fraction=0.9
max_beam_width=1
max_queue_delay_microseconds=0
backend=tensorrtllm

engine_path=$2
vit_engine_path=$3/visual_encoder.engine
engine_config_path=${triton_model_dir%/}/tensorrt_llm/config.pbtxt
preprocess_config_path=${triton_model_dir%/}/preprocessing/config.pbtxt
postprocess_config_path=${triton_model_dir%/}/postprocessing/config.pbtxt
ensemble_config_path=${triton_model_dir%/}/ensemble/config.pbtxt
bls_config_path=${triton_model_dir%/}/tensorrt_llm_bls/config.pbtxt

# copy tokenizer model to the target path
#cp ${tokenizer_dir%/}/tokeniz* ${triton_model_dir}

# fill config.pbtxt
python ${fill_template} --in_place ${engine_config_path} \
  triton_backend:${backend},triton_max_batch_size:${triton_max_batch_size},batching_strategy:inflight_fused_batching,engine_dir:${engine_path},batch_scheduler_policy:max_utilization,decoupled_mode:False,kv_cache_free_gpu_mem_fraction:${kv_cache_free_gpu_mem_fraction},max_beam_width:${max_beam_width},max_queue_delay_microseconds:${max_queue_delay_microseconds},exclude_input_in_output:True,lora_cache_optimal_adapter_size:256,lora_cache_max_adapter_size:256,lora_cache_gpu_memory_fraction:0.3,lora_cache_host_memory_bytes:2147483648


python ${fill_template} --in_place ${preprocess_config_path} \
  tokenizer_dir:${tokenizer_dir},triton_max_batch_size:${triton_max_batch_size},preprocessing_instance_count:1,engine_dir:${engine_path},vit_plan_dir:${vit_engine_path}

python ${fill_template} --in_place ${postprocess_config_path} \
  tokenizer_dir:${tokenizer_dir},triton_max_batch_size:${triton_max_batch_size},postprocessing_instance_count:1,skip_special_tokens:True

python ${fill_template} --in_place ${ensemble_config_path} \
  triton_max_batch_size:${triton_max_batch_size}

python ${fill_template} --in_place ${bls_config_path} \
  triton_max_batch_size:${triton_max_batch_size},decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False

python tensorrtllm_backend/scripts/launch_triton_server.py --model_repo ${triton_model_dir}

python tensorrtllm_backend/tools/internlm_xcomposer2/client.py --text "Please describe this image in detail." --request-output-len 200