# ACE: Anatomically Consistency Embedding via Composition and Decomposition

# Pretrain ACE models:

Using DDP to pretrain ACE:
```
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port 28301 main.py --arch swin_base --batch_size_per_gpu 8 --data_path pretrain image path --output_dir 
```