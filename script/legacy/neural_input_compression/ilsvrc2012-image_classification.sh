BASE_NAME=${1}
MAX_QUALITY=${2}

if [ $# -ne 2 ]; then
  echo "Illegal number of arguments"
  exit 2
fi

for quality in $(seq 1 1 ${MAX_QUALITY});
do
  json_str='{"models": {"model": {"compression_model": {"params": {"quality": '
  json_str+=${quality}
  json_str+='}}}}}'
  python script/legacy/task/image_classification.py \
  --config configs/legacy/ilsvrc2012/input_compression/${BASE_NAME}.yaml \
  --log log/legacy/input_compression/${BASE_NAME}-quality${quality}.txt \
  --json "${json_str}" -student_only -test_only -no_dp_eval
done
