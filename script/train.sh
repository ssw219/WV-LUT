python train.py \
  --batchSize 8 \
  --cropSize 96 \
  -e ../models/WVLUT_shared \
  --model_type shared \
  --totalIter 150000 \
  --displayStep 100 \
  --valStep 500 \
  --saveStep 2000 \
  --startIter 0 \
  --lr0 1e-3 \
  --lr1 1e-5 \
  --trainDir ../data/LOL_v1 \
  --valDataset LOL_v1_val
