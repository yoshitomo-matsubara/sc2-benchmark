BASE_NAME=${1}
FORMAT_NAME=${2}

if [ $# -eq 5 ]
then
  MIN_QUALITY=${3}
  STEP_SIZE=${4}
  MAX_QUALITY=${5}
else
  MIN_QUALITY=10
  STEP_SIZE=10
  MAX_QUALITY=100
fi


for quality in $(seq ${MIN_QUALITY} ${STEP_SIZE} ${MAX_QUALITY});
do
  sed -i "s/quality:.*/quality: ${quality}/g" configs/ilsvrc2012/input_compression/${BASE_NAME}.yaml
  pipenv run python script/image_classification/input_compression.py \
  --config configs/ilsvrc2012/input_compression/${BASE_NAME}.yaml \
  --log log/${FORMAT_NAME}_compression/${BASE_NAME}-quality${quality}.txt
done

sed -i "s/quality:.*/quality:/g" configs/ilsvrc2012/input_compression/${BASE_NAME}.yaml