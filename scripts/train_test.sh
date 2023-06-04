CUDA_VISIBLE_DEVICES=1 \
python src/train.py \
--data_path datasets/apple2orange

python src/test.py \
--data_path datasets/apple2orange