BASE_NAME=${1}

for quality in $(seq 10 10 100);
do

  sed -i "s/quality:.*/quality: ${quality}/g" configs/ilsvrc2012/input_compression/${BASE_NAME}.yaml
  pipenv run python script/image_classification/input_compression.py \
  --config configs/ilsvrc2012/input_compression/${BASE_NAME}.yaml \
  --log log/jpeg_compression/${BASE_NAME}-quality${quality}.txt
done

sed -i "s/quality:.*/quality:/g" configs/ilsvrc2012/input_compression/${BASE_NAME}.yaml