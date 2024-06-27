tokenizer_dir=/tmp/yanghao/TensorRT-LLM/examples/internlm/models--internlm--internlm-xcomposer2-vl-7b/snapshots/358caed4fa8e8c8c18b5a6724e986b879a9c9c8e
llm_engine_dir=/tmp/yanghao/TensorRT-LLM/examples/internlm/trt_engines/internlm-xcomposer2-vl-7b/fp16/1-gpu
visual_engine_dir=/tmp/yanghao/TensorRT-LLM/examples/internlm/visual_engines
#sh create_triton_model_repo.sh $tokenizer_dir $llm_engine_dir $visual_engine_dir

triton_model_dir=./triton_model_repo
python tensorrtllm_backend/scripts/launch_triton_server.py --model_repo ${triton_model_dir}