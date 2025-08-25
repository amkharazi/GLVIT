#python  TEST_glvit_v17.py    --TEST_ID EXP_V2_ID014    --dataset oxford_pets    --batch_size 256 --n_epoch 600 --image_size 32  --patch_size 4  --num_classes 37    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
python  TEST_glvit_v17_val.py    --TEST_ID EXP_V2_ID014    --dataset oxford_pets    --batch_size 256 --n_epoch 600 --image_size 32  --patch_size 4  --num_classes 37    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/EXP_V2_ID014/accuracy_stats/output.txt 

