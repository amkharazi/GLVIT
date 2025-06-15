# # MNIST

python  TEST_basic_timm_v2.py    --TEST_ID TEST_SCRATCH_v2_ID001    --dataset mnist    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
python  TEST_basic_timm_v2_val.py    --TEST_ID TEST_SCRATCH_v2_ID001    --dataset mnist    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
sleep 60
# # FASHION MNIST

python  TEST_basic_timm_v2.py    --TEST_ID TEST_SCRATCH_v2_ID002    --dataset fashionmnist    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
python  TEST_basic_timm_v2_val.py    --TEST_ID TEST_SCRATCH_v2_ID002    --dataset fashionmnist    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
sleep 60
# # CIFAR10

python  TEST_basic_timm_v2.py    --TEST_ID TEST_SCRATCH_v2_ID003    --dataset cifar10    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
python  TEST_basic_timm_v2_val.py    --TEST_ID TEST_SCRATCH_v2_ID003    --dataset cifar10    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
sleep 60
# # STL10   

python  TEST_basic_timm_v2.py    --TEST_ID TEST_SCRATCH_v2_ID004    --dataset stl10   --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
python  TEST_basic_timm_v2_val.py    --TEST_ID TEST_SCRATCH_v2_ID004    --dataset stl10   --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
sleep 60
# # FLOWERS 102

python  TEST_basic_timm_v2.py    --TEST_ID TEST_SCRATCH_v2_ID005    --dataset flowers102    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 102    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
python  TEST_basic_timm_v2_val.py    --TEST_ID TEST_SCRATCH_v2_ID005    --dataset flowers102    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 102    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
sleep 60
# # OXFORD PETS

python  TEST_basic_timm_v2.py    --TEST_ID TEST_SCRATCH_v2_ID006    --dataset oxford_pets    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 37    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
python  TEST_basic_timm_v2_val.py    --TEST_ID TEST_SCRATCH_v2_ID006    --dataset oxford_pets    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 37    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
sleep 60
# # CIFAR100

python  TEST_basic_timm_v2.py    --TEST_ID TEST_SCRATCH_v2_ID007    --dataset cifar100    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
python  TEST_basic_timm_v2_val.py    --TEST_ID TEST_SCRATCH_v2_ID007    --dataset cifar100    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 100    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
sleep 60
# # TINY IMAGENET 200

python  TEST_basic_timm_v2.py    --TEST_ID TEST_SCRATCH_v2_ID008    --dataset tinyimagenet    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 200    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
python  TEST_basic_timm_v2_val.py    --TEST_ID TEST_SCRATCH_v2_ID008    --dataset tinyimagenet    --batch_size 256 --n_epoch 600 --image_size 224  --patch_size 4  --num_classes 200    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
sleep 60