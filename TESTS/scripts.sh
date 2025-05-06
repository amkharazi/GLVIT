# # MNIST

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID0    --dataset mnist   --batch_size 32 --n_epoch 5 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID0    --dataset mnist   --batch_size 32 --n_epoch 5 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID00    --dataset mnist   --batch_size 32 --n_epoch 5 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID00    --dataset mnist   --batch_size 32 --n_epoch 5 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/TEST_FINAL_ID00/accuracy_stats/output.txt  


# # MNIST

python  TEST_basic.py    --TEST_ID TEST_FINAL_ID001    --dataset mnist   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID001    --dataset mnist   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 

python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID002    --dataset mnist   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID002    --dataset mnist   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/TEST_FINAL_ID002/accuracy_stats/output.txt  

# # FASHION MNIST

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID003    --dataset fashionmnist   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID003    --dataset fashionmnist   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID004    --dataset fashionmnist   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID004    --dataset fashionmnist   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/TEST_FINAL_ID004/accuracy_stats/output.txt 

# # CIFAR10

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID005    --dataset cifar10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID005    --dataset cifar10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID006    --dataset cifar10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID006    --dataset cifar10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/TEST_FINAL_ID006/accuracy_stats/output.txt

# # STL10

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID007    --dataset stl10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID007    --dataset stl10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID008    --dataset stl10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID008    --dataset stl10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/TEST_FINAL_ID008/accuracy_stats/output.txt 

# # FLOWERS 102

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID009    --dataset flowers102   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 102    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID009    --dataset flowers102   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 102    --dim 192    --depth 9   --heads 12   --mlp_dim 384 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID010    --dataset flowers102   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 102    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID010    --dataset flowers102   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 102    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/TEST_FINAL_ID010/accuracy_stats/output.txt 

# # OXFORD PETS

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID011    --dataset oxford_pets   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 37    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID011    --dataset oxford_pets   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 37    --dim 192    --depth 9   --heads 12   --mlp_dim 384 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID012    --dataset oxford_pets   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 37    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID012    --dataset oxford_pets   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 37    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/TEST_FINAL_ID012/accuracy_stats/output.txt

# # CIFAR100

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID013    --dataset cifar100   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 100    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID013    --dataset cifar100   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 100    --dim 192    --depth 9   --heads 12   --mlp_dim 384 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID014    --dataset cifar100   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 100    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID014    --dataset cifar100   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 100    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/TEST_FINAL_ID014/accuracy_stats/output.txt 

# # TINY IMAGENET 200

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID015    --dataset tiny_imagenet   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 200    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID015    --dataset tiny_imagenet   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 200    --dim 192    --depth 9   --heads 12   --mlp_dim 384 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID016    --dataset tiny_imagenet   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 200    --dim 192    --depth 9   --heads 12   --mlp_dim 384 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID016    --dataset tiny_imagenet   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 200    --dim 192    --depth 9   --heads 12   --mlp_dim 384 --print_w True > ../results/TEST_FINAL_ID016/accuracy_stats/output.txt   
