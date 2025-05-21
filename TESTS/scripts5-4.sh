# python  TEST_glvit_v11.py    --TEST_ID EXP_ID001    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_glvit_v11_val.py    --TEST_ID EXP_ID001    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 --print_w True > ../results/EXP_ID001/accuracy_stats/output.txt 

# python  TEST_basic_v2.py    --TEST_ID EXP_ID002    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_basic_v2_val.py    --TEST_ID EXP_ID002    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 

# python  TEST_glvit_v11.py    --TEST_ID EXP_ID003    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_glvit_v11_val.py    --TEST_ID EXP_ID003    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/EXP_ID003/accuracy_stats/output.txt 

python  TEST_basic_v2.py    --TEST_ID EXP_ID004    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
python  TEST_basic_v2_val.py    --TEST_ID EXP_ID004    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 192    --depth 9   --heads 12   --mlp_dim 384 

# python  TEST_glvit_v12.py    --TEST_ID EXP_ID005    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_glvit_v12_val.py    --TEST_ID EXP_ID005    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 --print_w True > ../results/EXP_ID005/accuracy_stats/output.txt 

# python  TEST_basic_v3.py    --TEST_ID EXP_ID006    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_basic_v3_val.py    --TEST_ID EXP_ID006    --dataset cifar100    --batch_size 256 --n_epoch 300 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
