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
  pipenv run python script/task/object_detection.py \
  --config configs/coco2017/input_compression/${BASE_NAME}.yaml \
  --log log/input_compression/${BASE_NAME}-quality${quality}.txt \
  --json "${json_str}" -student_only -test_only
done
