# Lamps: Learning Anatomy from Multiple Perspectives via Self-supervision

# Pretrain Lamps models:

Using DDP to pretrain Lamps:
```
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port 28301 main.py --arch swin_base --batch_size_per_gpu 8 --data_path pretrain image path --output_dir 
```


Pretrained weights
| Backbone | Input Resolution | Pretrain dataset | model |
|----------|------------------|------------------|-------|
| Swin-B | 448x448 | ChestX-ray14 | [download](https://drive.google.com/file/d/18nHHlsffRqYpQ1c9YK0uJZjL2C77gWXy/view?usp=sharing)|
| Swin-B | 448x448 | Large-scale |[download](https://drive.google.com/file/d/1v-BkyFPprLjo3IkrAP4NVnkTgQyDbvAw/view?usp=sharing)