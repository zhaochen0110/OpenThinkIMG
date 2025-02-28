cd /mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/src/eval
source activate vllm
# swatch -n SH-IDCA1404-10-140-54-81 nv_always
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export DEBUG_MODE="true"

# FILE_PATH="/mnt/petrelfs/share_data/mmtool/ChartQA/ChartQA Dataset/test/test_augmented.json"
FILE_PATH="/mnt/petrelfs/share_data/suzhaochen/datasets/chartgemma_cot/chartgemma_sample_test.json"
# MODEL_PATH="/mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/src/output_path/caogao/checkpoint-1000"
MODEL_PATH="/mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2-VL-2B-Instruct"
IMAGE_BASE_PATH="/mnt/petrelfs/share_data/suzhaochen/datasets/chart_cot"
# IMAGE_BASE_PATH="/mnt/petrelfs/share_data/mmtool/ChartQA/ChartQA Dataset/test/png"
OUTPUT_PATH="/mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/src/eval/output/chartqa/qwen2vl_chartgemma_default.json"
NUM_SAMPLES=3313231 # Set to null or omit to use all samples (None in Python)
TENSOR_PARALLEL_SIZE=4

quotatype="reserved"
jobname="r1v"
srun --partition=MoE --job-name=${jobname} --job-name="eval" --mpi=pmi2  --gres=gpu:0 -w SH-IDCA1404-10-140-54-81 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python3 inference_qwen2vl_chartqa.py \
  --file_path "$FILE_PATH" \
  --model_path "$MODEL_PATH" \
  --image_base_path "$IMAGE_BASE_PATH" \
  --output_path "$OUTPUT_PATH" \
  --num_samples "$NUM_SAMPLES" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"


