# CUDA_VISIBLE_DEVICES=1 python3 run_training.py --cfg configs/shape/nero-nerf/luyu.yaml
# CUDA_VISIBLE_DEVICES=0 python3 run_training.py --cfg configs/shape/nero-nerf/potion.yaml
# CUDA_VISIBLE_DEVICES=1 python3 run_training.py --cfg configs/shape/nero-nerf/tbell.yaml
# CUDA_VISIBLE_DEVICES=2 python3 run_training.py --cfg configs/shape/nero-nerf/teapot.yaml

# CUDA_VISIBLE_DEVICES=3 python3 eval_geo.py --cfg configs/shape/nero-nerf/angel.yaml
# CUDA_VISIBLE_DEVICES=1 python3 eval_mat.py --cfg configs/mat/syn/bell-nerf.yaml
# CUDA_VISIBLE_DEVICES=2 python3 eval_geo.py --cfg configs/shape/nero-nerf/tbell.yaml
# CUDA_VISIBLE_DEVICES=1 python3 eval_geo.py --cfg configs/shape/nero-nerf/teapot.yaml


CUDA_VISIBLE_DEVICES=1 python3 extract_mesh.py --cfg configs/shape/nero-nerf/luyu.yaml
CUDA_VISIBLE_DEVICES=1 python3 extract_mesh.py --cfg configs/shape/nero-nerf/potion.yaml
CUDA_VISIBLE_DEVICES=1 python3 extract_mesh.py --cfg configs/shape/nero-nerf/teapot.yaml
CUDA_VISIBLE_DEVICES=1 python3 extract_mesh.py --cfg configs/shape/nero-nerf/tbell.yaml