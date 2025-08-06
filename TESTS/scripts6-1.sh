
python  TEST_basic_v4.py    --TEST_ID FINAL_V2_ID013    --dataset oxford_pets    --batch_size 256 --n_epoch 600 --image_size 32  --patch_size 4  --num_classes 37    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
python  TEST_basic_v4_val.py    --TEST_ID FINAL_V2_ID013    --dataset oxford_pets    --batch_size 256 --n_epoch 600 --image_size 32  --patch_size 4  --num_classes 37    --dim 192    --depth 9   --heads 12   --mlp_dim 384 

# python  TEST_glvit_v14.py    --TEST_ID FINAL_V2_ID014    --dataset oxford_pets    --batch_size 256 --n_epoch 600 --image_size 32  --patch_size 4  --num_classes 37    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_glvit_v14_val.py    --TEST_ID FINAL_V2_ID014    --dataset oxford_pets    --batch_size 256 --n_epoch 600 --image_size 32  --patch_size 4  --num_classes 37    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/FINAL_V1_ID014/accuracy_stats/output.txt 