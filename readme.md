# How to Run ? 


To run a test, simple open the directory inside your console or terminal and run

```bash
python  TEST_basic.py    --TEST_ID TEST_ID001    --dataset cifar10   --batch_size 32 --n_epoch 10    --image_size 32 --train_size 40000  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
```

> Make sure you change the TEST_ID as it will overwrite your previous results.

> You also need the datasets, so please use :
```bash
python download_dataset.py
```


```bash
# # MNIST

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID0    --dataset mnist   --batch_size 32 --n_epoch 5 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID0    --dataset mnist   --batch_size 32 --n_epoch 5 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID00    --dataset mnist    --batch_size 32 --n_epoch 5 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID00    --dataset mnist   --batch_size 32 --n_epoch 5 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 --print_w True > ../results/TEST_FINAL_ID00/accuracy_stats/output.txt  


# # MNIST

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID017    --dataset mnist    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID017    --dataset mnist    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID018    --dataset mnist    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID018    --dataset mnist    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 --print_w True > ../results/TEST_FINAL_ID002/accuracy_stats/output.txt  

# # FASHION MNIST

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID019    --dataset fashionmnist    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID019    --dataset fashionmnist    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID020    --dataset fashionmnist    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID020    --dataset fashionmnist    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 --print_w True > ../results/TEST_FINAL_ID004/accuracy_stats/output.txt 

# # CIFAR10

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID021    --dataset cifar10    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID021    --dataset cifar10    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID022    --dataset cifar10    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID022    --dataset cifar10    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 --print_w True > ../results/TEST_FINAL_ID006/accuracy_stats/output.txt

# # STL10

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID023    --dataset stl10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID023    --dataset stl10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID024    --dataset stl10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID024    --dataset stl10   --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 --print_w True > ../results/TEST_FINAL_ID008/accuracy_stats/output.txt 

# # FLOWERS 102

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID025    --dataset flowers102    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 102    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID025    --dataset flowers102    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 102    --dim 64    --depth 6   --heads 8   --mlp_dim 128 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID026    --dataset flowers102    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 102    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID026    --dataset flowers102    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 102    --dim 64    --depth 6   --heads 8   --mlp_dim 128 --print_w True > ../results/TEST_FINAL_ID010/accuracy_stats/output.txt 

# # OXFORD PETS

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID027    --dataset oxford_pets    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 37    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID027    --dataset oxford_pets    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 37    --dim 64    --depth 6   --heads 8   --mlp_dim 128 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID028    --dataset oxford_pets    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 37    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID028    --dataset oxford_pets    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 37    --dim 64    --depth 6   --heads 8   --mlp_dim 128 --print_w True > ../results/TEST_FINAL_ID012/accuracy_stats/output.txt

# # CIFAR100

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID029    --dataset cifar100    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID029    --dataset cifar100    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID030    --dataset cifar100    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID030    --dataset cifar100    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 --print_w True > ../results/TEST_FINAL_ID014/accuracy_stats/output.txt 

# # TINY IMAGENET 200

# python  TEST_basic.py    --TEST_ID TEST_FINAL_ID031    --dataset tiny_imagenet    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 200    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_basic_val.py    --TEST_ID TEST_FINAL_ID031    --dataset tiny_imagenet    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 200    --dim 64    --depth 6   --heads 8   --mlp_dim 128 

# python  TEST_glvit_v7.py    --TEST_ID TEST_FINAL_ID032    --dataset tiny_imagenet    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 200    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
# python  TEST_glvit_v7_val.py    --TEST_ID TEST_FINAL_ID032    --dataset tiny_imagenet    --batch_size 32 --n_epoch 200 --image_size 32  --patch_size 4  --num_classes 200    --dim 64    --depth 6   --heads 8   --mlp_dim 128 --print_w True > ../results/TEST_FINAL_ID016/accuracy_stats/output.txt   

```
