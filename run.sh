# python3 run_training.py --cfg configs/shape/nero/angel.yaml
# python3 run_training.py --cfg configs/shape/nero/bell.yaml
# python3 run_training.py --cfg configs/shape/nero/cat.yaml
# python3 run_training.py --cfg configs/shape/nero/horse.yaml
# python3 run_training.py --cfg configs/shape/nero/luyu.yaml
# python3 run_training.py --cfg configs/shape/nero/potion.yaml
# python3 run_training.py --cfg configs/shape/nero/teapot.yaml
# python3 run_training.py --cfg configs/shape/nero/tbell.yaml
# python3 eval_geo.py --cfg configs/shape/nero/angel.yaml
# CUDA_VISIBLE_DEVICES=1 python3 eval_geo.py --cfg configs/shape/nero/bell.yaml
# CUDA_VISIBLE_DEVICES=1 python3 eval_geo.py --cfg configs/shape/nero/cat.yaml
# CUDA_VISIBLE_DEVICES=1 python3 eval_geo.py --cfg configs/shape/nero/horse.yaml
# CUDA_VISIBLE_DEVICES=1 python3 eval_geo.py --cfg configs/shape/nero/luyu.yaml
# CUDA_VISIBLE_DEVICES=1 python3 eval_geo.py --cfg configs/shape/nero/potion.yaml
# CUDA_VISIBLE_DEVICES=1 python3 eval_geo.py --cfg configs/shape/nero/teapot.yaml
# CUDA_VISIBLE_DEVICES=1 python3 eval_geo.py --cfg configs/shape/nero/tbell.yaml
# CUDA_VISIBLE_DEVICES=2 python3 eval_geo.py --cfg configs/shape/nero-nerf/bell.yaml
# CUDA_VISIBLE_DEVICES=2 python3 eval_geo.py --cfg configs/shape/nero-nerf/cat.yaml
# CUDA_VISIBLE_DEVICES=2 python3 eval_geo.py --cfg configs/shape/nero-nerf/horse.yaml
# CUDA_VISIBLE_DEVICES=2 python3 extract_mesh.py --cfg configs/shape/nero-nerf/angel.yaml
# CUDA_VISIBLE_DEVICES=2 python3 extract_mesh.py --cfg configs/shape/nero-nerf/bell.yaml
# CUDA_VISIBLE_DEVICES=2 python3 extract_mesh.py --cfg configs/shape/nero-nerf/cat.yaml
# CUDA_VISIBLE_DEVICES=2 python3 extract_mesh.py --cfg configs/shape/nero-nerf/horse.yaml
# CUDA_VISIBLE_DEVICES=2 python3 extract_mesh.py --cfg configs/shape/nero-nerf/luyu.yaml
# CUDA_VISIBLE_DEVICES=2 python3 extract_mesh.py --cfg configs/shape/nero-nerf/potion.yaml
# CUDA_VISIBLE_DEVICES=2 python3 extract_mesh.py --cfg configs/shape/nero-nerf/teapot.yaml
# CUDA_VISIBLE_DEVICES=2 python3 extract_mesh.py --cfg configs/shape/nero-nerf/tbell.yaml

CUDA_VISIBLE_DEVICES=2 python3 run_training.py --cfg configs/shape/nero-nerf/teapot.yaml