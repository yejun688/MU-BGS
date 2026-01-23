import cv2
import sys
import raytracing
import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from network.gaussian_renderer import render_surfel, render_initial, render_volume
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from utils.loss_utils import calculate_loss, l1_loss
from dataset.database import parse_database_name, get_database_split, BaseDatabase
from network.other_field import SingleVarianceNetwork, NeRFNetwork, TVLoss
from network.fields import *
from utils.image_utils import visualize_depth
from utils.network_utils import get_intersection, sample_pdf, extract_geometry, safe_l2_normalize
from utils.base_utils import color_map_forward, downsample_gaussian_blur
from utils.raw_utils import linear_to_srgb
import time
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.mesh_utils import GaussianExtractor, post_process_mesh
import random
from datetime import datetime
from random import randint

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

@torch.no_grad()
def evaluate_psnr(scene, renderFunc, renderkwargs):
    psnr_test = 0.0
    torch.cuda.empty_cache()
    if len(scene.getTestCameras()):
        for viewpoint in scene.getTestCameras():
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(scene.getTestCameras())
        
    torch.cuda.empty_cache()
    return psnr_test

NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger(args): 
    
    if not args.model_path:
        # 获取当前时间并格式化为精确到分钟
        current_time = datetime.now().strftime('%m%d_%H%M')
        # 获取args.source_path的最后一个子目录名
        last_subdir = os.path.basename(os.path.normpath(args.source_path))

        
        # 生成带有时间戳和opt属性的简洁输出目录
        args.model_path = os.path.join(
            "./output/", f"{last_subdir}/",
            f"{last_subdir}-{current_time}"
        )
       
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def build_imgs_info(database:BaseDatabase, img_ids, apply_mask_loss=False):
    images = [database.get_image(img_id) for img_id in img_ids]
    poses = [database.get_pose(img_id) for img_id in img_ids]
    Ks = [database.get_K(img_id) for img_id in img_ids]

    images = np.stack(images, 0)
    images = color_map_forward(images).astype(np.float32)
    Ks = np.stack(Ks, 0).astype(np.float32)
    poses = np.stack(poses, 0).astype(np.float32)
    imgs_info = {
        'imgs': images,
        'Ks': Ks, 
        'poses': poses,
    }

    if apply_mask_loss:
        masks = [database.get_depth(img_id)[1] for img_id in img_ids]
        masks = np.stack(masks, 0)
        imgs_info['masks'] = masks
    
    return imgs_info

def imgs_info_to_torch(imgs_info, device='cpu'):
    for k, v in imgs_info.items():
        v = torch.from_numpy(v)
        if k.startswith('imgs'): v = v.permute(0,3,1,2)
        imgs_info[k] = v.to(device)
    return imgs_info

def imgs_info_slice(imgs_info, idxs):
    new_imgs_info={}
    for k, v in imgs_info.items():
        new_imgs_info[k]=v[idxs]
    return new_imgs_info

def imgs_info_to_cuda(imgs_info):
    for k, v in imgs_info.items():
        imgs_info[k]=v.cuda()
    return imgs_info

def imgs_info_downsample(imgs_info, ratio):
    b, _, h, w = imgs_info['imgs'].shape
    dh, dw = int(ratio*h), int(ratio*w)
    imgs_info_copy = {k:v for k,v in imgs_info.items()}
    imgs_info_copy['imgs'], imgs_info_copy['Ks'] = [], []
    for bi in range(b):
        img = imgs_info['imgs'][bi].cpu().numpy().transpose([1,2,0])
        img = downsample_gaussian_blur(img, ratio)
        img = cv2.resize(img, (dw,dh), interpolation=cv2.INTER_LINEAR)
        imgs_info_copy['imgs'].append(torch.from_numpy(img).permute(2,0,1))
        K = torch.from_numpy(np.diag([dw / w, dh / h, 1]).astype(np.float32)) @ imgs_info['Ks'][bi]
        imgs_info_copy['Ks'].append(K)

    imgs_info_copy['imgs'] = torch.stack(imgs_info_copy['imgs'], 0)
    imgs_info_copy['Ks'] = torch.stack(imgs_info_copy['Ks'], 0)
    return imgs_info_copy


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1

 
class ShapeRenderer(nn.Module):
    default_cfg = {
        # standard deviation for opacity density
        'std_net': 'default',
        'std_act': 'exp',
        'inv_s_init': 0.3,
        'freeze_inv_s_step': None,

        # geometry network
        'sdf_net': 'default',
        'sdf_activation': 'none',
        'sdf_bias': 0.5,
        'sdf_n_layers': 8,
        'sdf_freq': 6,
        'sdf_d_out': 129,
        'geometry_init': True,

        # shader network
        'shader_config': {},

        # sampling strategy
        'n_samples': 64,
        'n_bg_samples': 32,
        'inf_far': 1000.0,
        'n_importance': 64,
        'up_sample_steps': 4,  # 1 for simple coarse-to-fine sampling
        'perturb': 1.0,
        'anneal_end': 50000,
        'train_ray_num': 1024,
        'test_ray_num': 2048,
        'clip_sample_variance': True,

        # dataset
        'database_name': 'nerf_synthetic/lego/black_800',

        # validation
        'test_downsample_ratio': True,
        'downsample_ratio': 0.25,
        'val_geometry': False,

        # losses
        'rgb_loss': 'charbonier',
        'apply_occ_loss': True,
        'apply_tv_loss' : True,
        'apply_sparse_loss' : True,
        'apply_hessian_loss': True,
        'apply_gaussian_loss': False,
        'occ_loss_step': 20000,
        'occ_loss_max_pn': 2048,
        'occ_sdf_thresh': 0.01,
        'apply_gaussian_loss': False,
        'gaussianLoss_step': 20000,

        "fixed_camera": False,
        
        # Tenso
        'device' : 'cuda',
        'gridSize' : [512, 512, 512],
        'aabb' : [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        'step_ratio' : 0.5,
        'alphaMask_thres' : 0.0001,
        'marched_weights_thres' : 0.0001,
        'sdf_n_comp' : 16,
        'app_n_comp' : 36,
        'sdf_dim' : 128,
        'app_dim' : 128,  
        'max_levels': 1,
        'has_radiance_field': False,
        'radiance_field_step': 0,
        'predict_BG': True,
        'isBGWhite': True,

        # dataset
        'nerfDataType': False,
        'split_manul': False,
        'apply_mask_loss': False,

        # alphaMask multi Length
        'mul_length': 10,
        # 'using_pretrain': False
        
        # GS的相关参数
        'sh_degree': 4,  # Spherical Harmonics degree for GaussianModel
        'radius_reg': 1.0,  # Radius for GaussianModel
    }

    def __init__(self, cfg, training=True):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        
        self.geometry_awared_contral = False
        
        self.device = self.cfg['device']
        gridSize = torch.tensor(self.cfg['gridSize'])
        max_levels = self.cfg['max_levels']
        self.aabb = torch.tensor(self.cfg['aabb'], device=self.device)
        self.center = torch.mean(self.aabb, axis=0).float().view(1, 1, 3)
        self.radius = (self.aabb[1] - self.center).mean().float()
        self.alphaMask = None
        self.step_ratio = self.cfg['step_ratio']
        self.alphaMask_thres = self.cfg['alphaMask_thres']
        self.marched_weights_thres = self.cfg['marched_weights_thres']
        self.sdf_n_comp, self.app_n_comp = self.cfg['sdf_n_comp'], self.cfg['app_n_comp']
        self.sdf_dim, self.app_dim = self.cfg['sdf_dim'], self.cfg['app_dim']
        self.update_stepSize(gridSize, max_levels)
        
        self.sdf_network = TensoSDF(
            self.gridSize, self.aabb, device=self.device, init_n_levels=self.max_levels,
            sdf_n_comp=self.sdf_n_comp, sdf_dim=self.sdf_dim, app_dim=self.app_dim)

        self.tv_reg = TVLoss()
        self.deviation_network = SingleVarianceNetwork(init_val=self.cfg['inv_s_init'], activation=self.cfg['std_act'])

        # background nerf is a nerf++ model (this is outside the unit bounding sphere, so we call it outer nerf)
        if self.cfg['predict_BG']:
            self.outer_nerf = NeRFNetwork(D=8, d_in=4, d_in_view=3, W=256, multires=10, multires_view=4, output_ch=4, skips=[4], use_viewdirs=True)
            nn.init.constant_(self.outer_nerf.rgb_linear.bias, np.log(0.5))
        else:
            self.cfg['n_bg_samples'] = 0

        self.cfg['shader_config'] = {
            'occ_loss_step': self.cfg['occ_loss_step'],
            'has_radiance_field': self.cfg['has_radiance_field'],
            'radiance_field_step': self.cfg['radiance_field_step'],
        }
        self.training = training
        self.color_network = ShapeShadingNetwork(self.cfg['shader_config'])
        self.sdf_inter_fun = lambda x: self.sdf_network.sdf(x, None)

        if training:
            self._init_dataset()
            # pretrain ref-gs 如果不需要就修改迭代次数 不能直接注释这行代码 
            self.pretrain_gs()
    
    def pretrain_gs(self):
        
        # 预训练ref高斯——让高斯快速训练到一个较好的状态得到深度(暂时还未实现)和法线后快速监督SDF场
        # print("begin pretraining gs:")
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        # parser.source_path=os.path.join(self.cfg['dataset_dir'],self.cfg['database_name'])
        # parser.add_argument('--eval', type=bool, default=True)
        # parser.add_argument('--white_background', type=bool, default=True)
        parser.add_argument('--ip', type=str, default="127.0.0.1")
        parser.add_argument('--port', type=int, default=6009)
        parser.add_argument('--detect_anomaly', action='store_true', default=False)
        parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
        parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,20000,30000,40000,50000,60000])
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--start_checkpoint", type=str, default = None)
        # self.args = parser.parse_args(sys.argv[1:])
        self.args = parser.parse_args(args=[])
        self.args.save_iterations.append(self.args.iterations)
        self.args.test_iterations = self.args.test_iterations + [i for i in range(10000, self.args.iterations+1, 5000)]
        self.args.test_iterations.append(self.args.volume_render_until_iter)
        self.args.source_path = os.path.join(self.cfg['dataset_dir'], self.cfg['database_name'].split('/')[1]) 
        lp._source_path = os.path.join(self.cfg['dataset_dir'],self.cfg['database_name'])
        
        if not self.args.model_path:
            # 获取当前时间并格式化为精确到分钟
            current_time = datetime.now().strftime('%m%d_%H%M')
            # 获取args.source_path的最后一个子目录名
            last_subdir = os.path.basename(os.path.normpath(self.args.source_path))

            
            # 生成带有时间戳和opt属性的简洁输出目录
            self.args.model_path = os.path.join(
                "./output/", f"{last_subdir}/",
                f"{last_subdir}-{current_time}"
            )
            
        self.dataset = lp.extract(self.args)
        self.opt = op.extract(self.args)
        self.pp = pp.extract(self.args)
        
        tb_writer = prepare_output_and_logger(self.args)
        
        # 高斯模型初始化——此时全都注册为类中的成员!
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        set_gaussian_para(self.gaussians, self.opt, vol=(self.opt.volume_render_until_iter > self.opt.init_until_iter))
        self.scene = Scene(self.dataset, self.gaussians)  # init all parameters(pos, scale, rot...) from pcds
        self.gaussians.training_setup(self.opt)
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussExtractor = GaussianExtractor(self.gaussians, render_initial, self.pp, bg_color=bg_color)
                             
        self.gaussians.training_setup(self.opt)
        self.viewpoint_stack = self.scene.getTrainCameras().copy()
        self.viewpoint_candidate = self.scene.getTrainCameras().copy()
        self.pretrain_it = self.opt.iterations + 1
        self.pretrain_it = 20000  # 预训练40000次
        # test
        # if self.training is False:
        #     self.pretrain_it=0
        viewpoint_stack = None
        ema_loss_for_log = 0.0
        ema_dist_for_log = 0.0
        ema_normal_for_log = 0.0
        ema_normal_smooth_for_log = 0.0
        ema_depth_smooth_for_log = 0.0
        ema_psnr_for_log = 0.0
        psnr_test = 0
        
        first_iter = 0
        progress_bar = tqdm(range(first_iter, self.pretrain_it), desc="Pre-training progress")
        first_iter += 1
        iteration = first_iter
        
        initial_stage = self.opt.initial
        
        if not initial_stage:
            self.opt.init_until_iter = 0
        
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        
        self.TEST_INTERVAL = 1000
        self.MESH_EXTRACT_INTERVAL = 2000

        # For real scenes
        self.USE_ENV_SCOPE = self.opt.use_env_scope  # False
        if self.USE_ENV_SCOPE:
            center = [float(c) for c in self.opt.env_scope_center]
            self.ENV_CENTER = torch.tensor(center, device='cuda')
            self.ENV_RADIUS = self.opt.env_scope_radius
            self.REFL_MSK_LOSS_W = 0.4
        
        while iteration < self.pretrain_it:
            iter_start.record()

            self.gaussians.update_learning_rate(iteration)
            
            # Increase SH levels every 1000 iterations
            if iteration > self.opt.feature_rest_from_iter and iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # Control the init stage
            if iteration > self.opt.init_until_iter:
                initial_stage = False
            
            # Control the indirect stage
            if iteration == self.opt.indirect_from_iter + 1:
                self.opt.indirect = 1


            if iteration == (self.opt.volume_render_until_iter + 1) and self.opt.volume_render_until_iter > self.opt.init_until_iter:
                reset_gaussian_para(self.gaussians, self.opt)

            # Initialize envmap
            if not initial_stage:
                # 默认设置是18000次后开启第二个环境图
                if iteration <= self.opt.volume_render_until_iter:
                    envmap2 = self.gaussians.get_envmap_2 
                    envmap2.build_mips()
                else:
                    envmap = self.gaussians.get_envmap 
                    envmap.build_mips()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            
            from random import randint
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))


            # Set render
            render = select_render_method(iteration, self.opt, initial_stage)
            render_pkg = render(viewpoint_cam, self.gaussians, self.pp, background, srgb=self.opt.srgb, opt=self.opt)
            # print(render_pkg['rend_alpha'].shape) [1,800,800]
            # if iteration==3999: 
            #     print(render_pkg['rend_alpha'].max(), render_pkg['rend_alpha'].min())
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # if iteration >= 2000:
            #     gs_depth_hand = render_pkg["surf_depth"]
            #     gs_depth = gs_depth_hand.mean(dim=0, keepdim=True).permute(1, 2, 0)
            #     min_d, max_d = gs_depth.min(), gs_depth.max()
            #     print("min_d, max_d:", min_d, max_d)
            
            
            gt_image = viewpoint_cam.original_image.cuda()

            total_loss, tb_dict = calculate_loss(viewpoint_cam, self.gaussians, render_pkg, self.opt, iteration)
            dist_loss, normal_loss, loss, Ll1, normal_smooth_loss, depth_smooth_loss = tb_dict["loss_dist"], tb_dict["loss_normal_render_depth"], tb_dict["loss0"], tb_dict["loss_l1"], tb_dict["loss_normal_smooth"], tb_dict["loss_depth_smooth"] 

            def get_outside_msk():
                return None if not self.USE_ENV_SCOPE else torch.sum((self.gaussians.get_xyz - self.ENV_CENTER[None])**2, dim=-1) > self.ENV_RADIUS**2
            
            if self.USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
                refls = self.gaussians.get_refl
                refl_msk_loss = refls[get_outside_msk()].mean()
                total_loss += self.REFL_MSK_LOSS_W * refl_msk_loss
            
            # self.gaussians.check_gradients(prefix="联合训练开始时")
            total_loss.backward() 
            # self.gaussians.check_gradients(prefix="联合训练开始时")

            iter_end.record()


            # self.gaussians.check_gradients(prefix="联合训练开始时")

            with torch.no_grad():
                
                if iteration % self.TEST_INTERVAL == 0 or iteration == first_iter + 1 or iteration == self.opt.volume_render_until_iter + 1:
                    save_training_vis(viewpoint_cam, self.gaussians, background, render, self.pp, self.opt, iteration, initial_stage, self.args)
                # 指数移动平均平滑损失值
                ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
                ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
                ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log
                ema_normal_smooth_for_log = 0.4 * normal_smooth_loss + 0.6 * ema_normal_smooth_for_log
                ema_depth_smooth_for_log = 0.4 * depth_smooth_loss + 0.6 * ema_depth_smooth_for_log
                ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
                if iteration % self.TEST_INTERVAL == 0:
                    psnr_test = evaluate_psnr(self.scene, render, {"pipe": self.pp, "bg_color": background, "opt": self.opt})
                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "Distort": f"{ema_dist_for_log:.{5}f}",
                        "Normal": f"{ema_normal_for_log:.{5}f}",
                        "Points": f"{len(self.gaussians.get_xyz)}",
                        "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                        "PSNR-test": f"{psnr_test:.{4}f}"
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)
                if iteration == self.pretrain_it - 1:
                    progress_bar.close()

                if tb_writer:
                    tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                    tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                self.args.test_iterations, self.scene, render, {"pipe": self.pp, "bg_color": background, "opt":self.opt})

                if iteration in self.args.save_iterations:
                    print(f"\n[ITER {iteration}] Saving Gaussians")
                    self.scene.save(iteration)

                # Densification 25000次迭代前(非18000次迭代时) 持续进行增密
                if iteration < self.opt.densify_until_iter and iteration != self.opt.volume_render_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter],
                                                                        radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration <= self.opt.init_until_iter:
                        opacity_reset_intval = 3000
                        densification_interval = 100
                    elif iteration <= self.opt.normal_prop_until_iter :
                        opacity_reset_intval = 3000
                        densification_interval = self.opt.densification_interval_when_prop
                    else:
                        opacity_reset_intval = 3000
                        densification_interval = 100
                    # 500次迭代开始 每隔100次迭代进行增密
                    if iteration > self.opt.densify_from_iter and iteration % densification_interval == 0:
                        size_threshold = 10 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.prune_opacity_threshold, self.scene.cameras_extent,
                                                    size_threshold)

                    HAS_RESET0 = False
                    if iteration % opacity_reset_intval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        HAS_RESET0 = True
                        outside_msk = get_outside_msk()
                        self.gaussians.reset_opacity0()
                        self.gaussians.reset_refl(exclusive_msk=outside_msk)
                    if self.opt.opac_lr0_interval > 0 and (
                            self.opt.init_until_iter < iteration <= self.opt.normal_prop_until_iter ) and iteration % self.opt.opac_lr0_interval == 0:
                        self.gaussians.set_opacity_lr(self.opt.opacity_lr)
                    if (self.opt.init_until_iter < iteration <= self.opt.normal_prop_until_iter ) and iteration % self.opt.normal_prop_interval == 0:
                        if not HAS_RESET0:
                            outside_msk = get_outside_msk()
                            self.gaussians.reset_opacity1(exclusive_msk=outside_msk)
                            if iteration > self.opt.volume_render_until_iter and self.opt.volume_render_until_iter > self.opt.init_until_iter:
                                self.gaussians.dist_color(exclusive_msk=outside_msk)
                                # self.gaussians.dist_albedo(exclusive_msk=outside_msk)

                            self.gaussians.reset_scale(exclusive_msk=outside_msk)
                            if self.opt.opac_lr0_interval > 0 and iteration != self.opt.normal_prop_until_iter :
                                self.gaussians.set_opacity_lr(0.0)
                    
                if (iteration >= self.opt.indirect_from_iter and iteration % self.MESH_EXTRACT_INTERVAL == 0) or iteration == (self.opt.indirect_from_iter):
                    if not HAS_RESET0:
                        self.gaussExtractor.reconstruction(self.scene.getTrainCameras())
                        if 'ref_real' in self.dataset.source_path:
                            mesh = self.gaussExtractor.extract_mesh_unbounded(resolution=self.opt.mesh_res)
                        else:
                            depth_trunc = (self.gaussExtractor.radius * 2.0) if self.opt.depth_trunc < 0  else self.opt.depth_trunc
                            voxel_size = (depth_trunc / self.opt.mesh_res) if self.opt.voxel_size < 0 else self.opt.voxel_size
                            sdf_trunc = 5.0 * voxel_size if self.opt.sdf_trunc < 0 else self.opt.sdf_trunc
                            mesh = self.gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                        mesh = post_process_mesh(mesh, cluster_to_keep=self.opt.num_cluster)
                        ply_path = os.path.join(self.args.model_path,f'test_{iteration:06d}.ply')
                        o3d.io.write_triangle_mesh(ply_path, mesh)
                        self.gaussians.update_mesh(mesh)

                if iteration < self.pretrain_it:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    # self.gaussians.check_gradients(prefix="联合训练开始时")

                if iteration in self.args.checkpoint_iterations:
                    print(f"\n[ITER {iteration}] Saving Checkpoint")
                    torch.save((self.gaussians.capture(), iteration), self.scene.model_path + f"/chkpnt{iteration}.pth")

            iteration += 1
            
            # self.gaussians.check_gradients(prefix="联合训练开始时")
      
    def update_stepSize(self, gridSize, max_levels):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)        
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize = torch.tensor(gridSize.cpu(), dtype=torch.int32).to(self.device)
        self.max_levels = max_levels
        self.units = self.aabbSize / (self.gridSize-1)
        self.stepSize = torch.mean(self.units)*self.step_ratio
        print("sampling step size: ", self.stepSize)
    
    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(128,128,128)):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        alpha, grid_xyz = self.compute_gridAlpha(gridSize)
        grid_xyz = grid_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None] # (1,1,gridSize012,)
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = grid_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))

        torch.set_default_tensor_type('torch.FloatTensor')
        return new_aabb

    @torch.no_grad()    
    def compute_gridAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else torch.LongTensor(gridSize).to(self.device)
        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        grid_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples
        stepLength = torch.mean(self.aabbSize / (gridSize - 1))
        alpha = torch.zeros_like(grid_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_grid_alpha(grid_xyz[i].view(-1,3), stepLength).view((gridSize[1], gridSize[2]))
        return alpha, grid_xyz

    def compute_grid_alpha(self, xyz_locs, length):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
        
        alpha = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            xyz_sampled = xyz_locs[alpha_mask]
            sdfs = self.sdf_inter_fun(xyz_sampled)[..., 0]
            near_surf_mask = torch.abs(sdfs) < self.cfg['mul_length'] * length
            inv_s = self.deviation_network(xyz_sampled).clip(1e-6, 1e6)
            inv_s = inv_s[..., 0]
            estimated_next_sdf = sdfs - length * 0.5
            estimated_prev_sdf = sdfs + length * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)       # [N_rays, ]
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha_weights = ((p + 1e-5) / (c + 1e-5)).clip(min=0.0, max=1.0)
            alpha_weights[near_surf_mask] = 1
            alpha[alpha_mask] = alpha_weights
        return alpha      

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'sdf_n_comp': self.sdf_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'sdf_dim': self.sdf_dim,
            'app_dim': self.app_dim,

            'alphaMask_thres': self.alphaMask_thres,
            'marched_weights_thres' : self.marched_weights_thres,
            'step_ratio': self.step_ratio,
            'max_levels': self.max_levels,
        }

    def ckpt_to_save(self):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'network_state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        return ckpt

    def load_ckpt(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['network_state_dict'])

    def upsample_sdf_grid(self, res_target):
        res_target = torch.tensor(res_target, device=self.device)
        new_res, max_levels = self.sdf_network.upsample_volume_grid(res_target)
        self.update_stepSize(new_res, max_levels)

    def shrink_sdf_grid(self, new_aabb):
        raise NotImplementedError

    def get_train_opt_params(self, learning_rate_xyz, learning_rate_net):
        grad_vars = []
        get_grad_vars_from_net = lambda net : [{'params' : net.parameters(), 'lr' : learning_rate_net}]
        grad_vars += self.sdf_network.get_optparam_groups(learning_rate_xyz, learning_rate_net)
        grad_vars += get_grad_vars_from_net(self.deviation_network)
        grad_vars += get_grad_vars_from_net(self.color_network)
        if self.cfg['predict_BG']:
            grad_vars += get_grad_vars_from_net(self.outer_nerf)
        return grad_vars
   
    def _init_dataset(self,mug_reg=False):
        # train/test split
        self.database = parse_database_name(self.cfg['database_name'], self.cfg['dataset_dir'], isWhiteBG=self.cfg['isBGWhite'])
        self.train_ids, self.test_ids = get_database_split(self.database, split_manul=self.cfg['split_manul'])
        self.train_ids = np.asarray(self.train_ids)

        self.train_imgs_info = build_imgs_info(self.database, self.train_ids, apply_mask_loss=self.cfg['apply_mask_loss'])
        self.train_imgs_info = imgs_info_to_torch(self.train_imgs_info, 'cpu')
        b, _, h, w = self.train_imgs_info['imgs'].shape
        print(f'training size {h} {w} ...')
        self.train_num = len(self.train_ids)

        self.test_imgs_info = build_imgs_info(self.database, self.test_ids, apply_mask_loss=self.cfg['apply_mask_loss'])
        self.test_imgs_info = imgs_info_to_torch(self.test_imgs_info, 'cpu')
        self.test_num = len(self.test_ids)
        print(f'Acutal splits num: train -> {self.train_num}, val -> {self.test_num}')
        # clean the data if we already have
        if hasattr(self, 'train_batch'):
            del self.train_batch

        self.train_batch_mutual_reg, _, _, _ = self._construct_ray_batch_nerf_for_mutual_reg(self.train_imgs_info)
        print("begin filter per image:")
        for img_idx, img_batch in tqdm(self.train_batch_mutual_reg.items()):
            filtered_img_batch, valid_count = self.filtering_rays_per_image(img_batch)
            if valid_count > 0:
                self.train_batch_mutual_reg[img_idx] = filtered_img_batch
                print(f"Image {img_idx}: original ray {img_batch['rays_o'].shape[0]}->valid ray {valid_count}")
                
        if self.cfg['nerfDataType']:
            self.train_batch, _, _, _ = self._construct_ray_batch_nerf(self.train_imgs_info)
        else:    
            self.train_batch, _, _, _ = self._construct_ray_batch(self.train_imgs_info, apply_mask=self.cfg['apply_mask_loss'])
                
        self.filtering_train_rays()
        self._shuffle_train_batch()
        
    
    def _shuffle_train_batch(self):
        self.train_batch_i = 0
        shuffle_idxs = torch.randperm(self.tbn, device='cpu')  # shuffle
        for k, v in self.train_batch.items():
            self.train_batch[k] = v[shuffle_idxs]

    def _construct_ray_batch(self, imgs_info, device='cpu', apply_mask=False):
        imn, _, h, w = imgs_info['imgs'].shape
    
        # 1. 创建像素坐标网格
        y_coords = torch.arange(h, dtype=torch.float32, device=device)
        x_coords = torch.arange(w, dtype=torch.float32, device=device)
        
        # 创建网格 [h, w]
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # y在前符合图像坐标系
        
        # 2. 保存整数坐标用于对齐
        int_yy = yy.clone().int()  # 整数y坐标 [h, w]
        int_xx = xx.clone().int()  # 整数x坐标 [h, w]
        
        # 3. 转换为齐次坐标（像素中心）
        coords = torch.stack([xx, yy], dim=-1) + 0.5  # [h, w, 2] (x, y)
        coords = torch.cat([coords, torch.ones(h, w, 1, device=device)], dim=-1)  # [h, w, 3]
        
        # 4. 扩展到所有图像
        coords = coords[None, ...].repeat(imn, 1, 1, 1)  # [imn, h, w, 3]
        int_yy = int_yy[None, ...].repeat(imn, 1, 1)  # [imn, h, w]
        int_xx = int_xx[None, ...].repeat(imn, 1, 1)  # [imn, h, w]
        
        # 5. 展平为射线列表
        rn = imn * h * w
        coords_flat = coords.reshape(imn * h * w, 3)  # [rn, 3]
        int_yy_flat = int_yy.reshape(rn)  # [rn]
        int_xx_flat = int_xx.reshape(rn)  # [rn]
        
        # 6. 计算相机空间方向
        # 获取逆内参矩阵 [imn, 3, 3]
        inv_Ks = torch.inverse(imgs_info['Ks']).permute(0, 2, 1)  # [imn, 3, 3]
        
        # 批处理矩阵乘法计算方向 [imn, h*w, 3] @ [imn, 3, 3] = [imn, h*w, 3]
        dirs = (coords_flat.view(imn, h*w, 3) @ inv_Ks).view(rn, 3)  # [rn, 3]
        
        # 7. 获取图像颜色值
        imgs = imgs_info['imgs'].permute(0, 2, 3, 1)  # [imn, h, w, 3]
        rgbs = imgs.reshape(rn, 3)  # [rn, 3]
        
        # 8. 创建图像索引
        idxs = torch.arange(imn, device=device)[:, None, None]  # [imn, 1, 1]
        idxs = idxs.repeat(1, h, w).reshape(rn)  # [rn]
        
        # 9. 获取相机位姿并计算世界坐标系中的光线
        poses = imgs_info['poses']  # [imn, 3, 4]
        rays_o = torch.zeros(rn, 3, device=device)  # [rn, 3]
        world_dirs = torch.zeros(rn, 3, device=device)  # [rn, 3]
        
        # 对每个图像单独处理（避免大矩阵操作）
        for i in range(imn):
            mask = (idxs == i)
            pts = coords_flat[mask].view(-1, 3, 1)  # [N, 3, 1]
            
            # 计算光线原点: -R^T t
            t = poses[i, :, 3]  # [3]
            R = poses[i, :, :3]  # [3, 3]
            rays_o_i = -R.t() @ t  # [3]
            rays_o[mask] = rays_o_i
            
            # 计算世界坐标系中的方向: R * (K_inv @ [x,y,1])
            # 更准确的方法：dirs在相机空间，用R旋转到世界空间
            dirs_i = dirs[mask]  # [N, 3]
            world_dirs_i = dirs_i @ R.t()  # [N, 3]
            world_dirs[mask] = F.normalize(world_dirs_i, dim=-1)
        
        # 10. 整合为射线向量 (原点 + 方向)
        rays = torch.cat([rays_o, world_dirs], dim=-1)  # [rn, 6]
        
        # 11. 获取前景蒙版（如果可用）
        if apply_mask and 'masks' in imgs_info:
            masks = imgs_info['masks'].reshape(rn)  # [rn]
        else:
            masks = torch.ones(rn, device=device, dtype=torch.float32)
        
        # 12. 返回对齐的批处理数据
        ray_batch = {
            'rays': rays,         # [rn, 6] 原点+方向
            'rgbs': rgbs,         # [rn, 3] RGB颜色
            'fg_mask': masks,     # [rn] 前景蒙版
            'used_index': idxs,   # [rn] 图像索引
            'used_y': int_yy_flat,  # [rn] 整数y坐标
            'used_x': int_xx_flat,  # [rn] 整数x坐标
        }
    
        return ray_batch, rn, h, w

    def _construct_ray_batch_nerf(self, imgs_info, device='cpu', is_train=True):
        imn, _, h, w = imgs_info['imgs'].shape
        i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        K = imgs_info['Ks'][0]
        dirs = torch.stack([(i-K[0][2]+0.5)/K[0][0], -(j-K[1][2]+0.5)/K[1][1], -torch.ones_like(i)], -1) # h, w, 3
        dirs = dirs[None, ...].repeat(imn, 1, 1, 1) # imn, h, w, 3
        
        imgs = imgs_info['imgs'].permute(0, 2, 3, 1).reshape(imn, h * w, 3)  # imn,h*w,3
        idxs = torch.arange(imn, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # imn,h*w,1
        poses = imgs_info['poses']  # imn, 4, 4
        # 相机位置为射线起点
        # [...,:3,-1] ...表示维度保留 :3表示取矩阵的前三行 -1表示取最后一列 (imn,3)
        # [:,None,:] 表示插入一个新的维度 (imn,1,3)
        # repeat(1, h*w, 1) 表示在第二个维度上重复 h*w 次 (imn, h*w, 3)
        rays_o = poses[..., :3, -1][:, None, :].repeat(1, h*w, 1) # imn, h*w, 3
        
        rn = imn * h * w
        dirs = dirs.float().reshape(rn, 3).to(device) # rn,3
        idxs = idxs.long().reshape(rn, 1).to(device)  # rn,1 
        rays_o = rays_o.float().reshape(rn, 3).to(device) # rn,3
        imgs = imgs.float().reshape(rn, 3).to(device) # rn,3
        # 转换到世界坐标系
        # dirs[..., None, :] -> (rn,1,3)
        # poses[idxs[..., 0], :3, :3] -> (rn,3,3)
        # dirs扩展为(rn,3,3)与poses矩阵逐元素相乘 后续在维度上进行累加达到矩阵乘法的效果
        dirs = torch.sum(dirs[..., None, :] * poses[idxs[..., 0], :3, :3], -1)  # rn,3
        dirs = F.normalize(dirs, dim=-1)
        
        ray_batch = {
            'dirs': dirs,
            'rays_o': rays_o,
            'rgbs': imgs,
            'human_poses': poses[idxs[..., 0], :3, :],
        }

        if is_train:
            masks = imgs_info['masks'].reshape(imn, h * w).float().reshape(rn, 1).to(device)
            ray_batch['masks'] = masks
        return ray_batch, rn, h, w

    def _construct_ray_batch_nerf_for_mutual_reg(self, imgs_info, device='cpu', is_train=True):
        
        # 获取图像信息
        imn, _, h, w = imgs_info['imgs'].shape
        # 使用整数网格确保精确对齐
        int_i = torch.arange(w, dtype=torch.int64, device=device)
        int_j = torch.arange(h, dtype=torch.int64, device=device)
        int_i_grid, int_j_grid = torch.meshgrid(int_i, int_j, indexing='xy')  # w, h (x, y)
        int_i_grid = int_i_grid.permute(1, 0)  # h, w (转为图像坐标系)
        int_j_grid = int_j_grid.permute(1, 0)  # h, w
        K = imgs_info['Ks'][0]  # 假设所有图像共享内参
        poses = imgs_info['poses']  # imn, 4, 4
        ray_batch_per_image = {}
        # 遍历每张图像
        for img_idx in range(imn):
            # 当前图像的坐标网格
            img_i_grid = int_i_grid.clone()  # h, w
            img_j_grid = int_j_grid.clone()  # h, w
            
            # 展平坐标
            rn_img = h * w
            used_x = img_i_grid.reshape(rn_img)  # [rn_img]
            used_y = img_j_grid.reshape(rn_img)  # [rn_img]
            
            # 创建图像索引张量
            idxs = torch.full((rn_img,), img_idx, dtype=torch.int64, device=device)  # [rn_img]
            
            i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            dirs = torch.stack([(i-K[0][2]+0.5)/K[0][0], -(j-K[1][2]+0.5)/K[1][1], -torch.ones_like(i)], -1)
            dirs = dirs.float().reshape(rn_img, 3).to(device) # rn,3
            rays_o = poses[img_idx][:3, -1].repeat(rn_img, 1)  # [rn_img, 3] 光线原点
            
            # 转换到世界坐标系
            # dirs = dirs.float().to(device)  # rn_img, 3
            rays_o = rays_o.float().to(device)  # rn_img, 3
            
            # 使用当前图像的位姿进行变换
            pose = poses[img_idx]  # 4,4
            dirs = torch.sum(dirs[..., None, :] * pose[None, :3, :3], -1)  # rn_img,3
            dirs = F.normalize(dirs, dim=-1)
            
            # 获取当前图像的颜色值
            rgbs = imgs_info['imgs'][img_idx].permute(1, 2, 0).reshape(rn_img, 3)  # [rn_img, 3]
            
            # 创建当前图像的光线数据字典
            img_ray_batch = {
                'rays_o': rays_o,       # [rn_img, 3] 光线原点
                'dirs': dirs,           # [rn_img, 3] 光线方向
                'rgbs': rgbs,           # [rn_img, 3] 颜色值
                'used_index': idxs,     # [rn_img] 图像索引
                'used_y': used_y,       # [rn_img] 整数y坐标
                'used_x': used_x,       # [rn_img] 整数x坐标
            }
            
            if is_train and 'masks' in imgs_info:
                masks = imgs_info['masks'][img_idx].reshape(rn_img,1).float().to(device)  # [rn_img]
                img_ray_batch['masks'] = masks
            
            ray_batch_per_image[img_idx] = img_ray_batch
        
        rn = imn * h * w
        return ray_batch_per_image, rn, h, w

    def get_human_coordinate_poses(self, poses):
        pn = poses.shape[0]
        cam_cen = (-poses[:, :, :3].permute(0, 2, 1) @ poses[:, :, 3:])[..., 0]  # pn,3
        if self.cfg['fixed_camera']:
            pass
        else:
            cam_cen[..., 2] = 0

        Y = torch.zeros([1, 3], device=poses.device).expand(pn, 3)
        Y[:, 2] = -1.0
        Z = torch.clone(poses[:, 2, :3]).to(poses.device)  # pn, 3
        Z[:, 2] = 0
        Z = F.normalize(Z, dim=-1)
        X = torch.cross(Y, Z)  # pn, 3
        R = torch.stack([X, Y, Z], 1)  # pn,3,3
        t = -R @ cam_cen[:, :, None]  # pn,3,1
        return torch.cat([R, t], -1)

    # 随着包围盒的缩小 所有光线不一定都和包围盒有交点 过滤掉这些光线
    @torch.no_grad()
    def filtering_train_rays(self, device='cuda', chunk=10240*5):
        print('========> filtering rays ...')
        tt = time.time()
        rays_o, rays_d = self.train_batch['rays_o'], self.train_batch['dirs']  
        N = torch.tensor(rays_o.shape[:-1]).cpu().prod()
        aabb = self.aabb.to(device)

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)        
        for idx_chunk in idx_chunks:
            rays_o_chunk, rays_d_chunk = rays_o[idx_chunk].to(device), rays_d[idx_chunk].to(device)
            # 将方向为0的地方替换为一个小值，避免除零错误
            vec = torch.where(rays_d_chunk == 0, torch.full_like(rays_d_chunk, 1e-6), rays_d_chunk)
            rate_a = (aabb[1] - rays_o_chunk) / vec
            rate_b = (aabb[0] - rays_o_chunk) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
            t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
            mask_inbbox = t_max > t_min
            
            mask_filtered.append(mask_inbbox.cpu())
            
        mask_filtered = torch.cat(mask_filtered).view(rays_o.shape[:-1])
        valid_rn = torch.sum(mask_filtered)
        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {valid_rn / N}')
        
        for k, v in self.train_batch.items():
            self.train_batch[k] = v[mask_filtered]
        self.tbn = valid_rn

    @torch.no_grad()
    def filtering_rays_per_image(self, img_ray_batch, device='cuda', chunk=10240*5):
        # 从光线数据中分离出原点和方向
        rays_o = img_ray_batch['rays_o']  # [rn_img, 3] 光线原点
        rays_d = img_ray_batch['dirs']  # [rn_img, 3] 光线方向
        
        N = rays_o.shape[0]  # 当前图像的光线总数
        aabb = self.aabb.to(device)  # 场景边界框
        
        # 如果没有光线需要处理，直接返回
        if N == 0:
            return img_ray_batch, 0
        
        # 初始化有效掩码
        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)  # 分块索引
        
        # 分块处理光线
        for idx_chunk in idx_chunks:
            # 将当前块数据移到GPU
            rays_o_chunk = rays_o[idx_chunk].to(device)
            rays_d_chunk = rays_d[idx_chunk].to(device)
            
            # 避免除零错误
            vec = torch.where(rays_d_chunk == 0, torch.full_like(rays_d_chunk, 1e-6), rays_d_chunk)
            
            # 计算与AABB的交点参数
            rate_a = (aabb[1] - rays_o_chunk) / vec  # 与最大边界交点
            rate_b = (aabb[0] - rays_o_chunk) / vec  # 与最小边界交点
            
            # 计算每条光线的进入(t_min)和退出(t_max)参数
            t_min = torch.minimum(rate_a, rate_b).amax(-1)
            t_max = torch.maximum(rate_a, rate_b).amin(-1)
            
            # 判断光线是否与AABB相交
            mask_inbbox = t_max > t_min
            mask_filtered.append(mask_inbbox.cpu())
        
        # 合并所有块的过滤结果
        mask_filtered = torch.cat(mask_filtered)
        valid_rn = torch.sum(mask_filtered).item()  # 有效光线数量
        
        # 创建过滤后的光线数据字典
        filtered_batch = {}
        for k, v in img_ray_batch.items():
            filtered_batch[k] = v[mask_filtered]
        
        return filtered_batch, valid_rn

    @torch.no_grad()     
    def nvs(self, pose, K, h, w):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = 'cuda'
        K = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device)
        pose = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0).to(device)
        rn = h * w

        def construct_ray_dirs():
            coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
            coords = coords.to(device)
            coords = coords.float()[None, :, :, :].repeat(1, 1, 1, 1)  # 1,h,w,2
            coords = coords.reshape(1, h * w, 2)
            coords = torch.cat([coords + 0.5, torch.ones(1, h * w, 1, dtype=torch.float32, device=device)], 2)  # 1,h*w,3
            # 1,h*w,3 @ imn,3,3 => 1,h*w,3
            dirs = coords @ torch.inverse(K).permute(0, 2, 1)
            dirs = dirs.reshape(-1, 3)
            dirs = F.normalize(dirs, dim=-1)         
            ray_batch = {
                'dirs': dirs.float().to(device),
            }
            return ray_batch
        
        def construct_ray_dirs_nerf():
            i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
            i = i.t().to(device)
            j = j.t().to(device)
            k = K[0]
            dirs = torch.stack([(i-k[0][2])/k[0][0], -(j-k[1][2])/k[1][1], -torch.ones_like(i).to(device)], -1) # h, w, 3
            dirs = F.normalize(dirs, dim=-1)
            dirs = dirs.reshape(-1, 3)

            idxs = torch.arange(1, dtype=torch.int64, device=device)[:, None, None].repeat(1, rn, 1)  # 1,rn,1
            idxs = idxs.long().reshape(rn, 1).to(device)
            
            ray_batch = {
                'dirs': dirs.float().to(device),
                'human_poses': pose[idxs[..., 0], :3, :],
                'idxs': idxs,
            }
            return ray_batch
        
        if self.cfg['nerfDataType']:
            ray_batch = construct_ray_dirs_nerf()
            process_ray_batch = self._process_ray_batch_nerf
        else:
            ray_batch = construct_ray_dirs()
            process_ray_batch = self._process_ray_batch            

        trn = 2048
        output = {
                  'color' : [], 'albedo' : [], 'roughness' : [], 'normal' : [], 'normal_vis': [],
                  'occ_predict' : [], 'occ_trace' : [], 
                  'diff_color' : [], 'spec_color' : [], 
                  'diff_light' : [], 'spec_light' : [],
                  'indirect_light' : [], 
                  }
        if self.cfg['has_radiance_field']:
            output['radiance'] = []
        for ri in range(0, rn, trn):
            cur_ray_batch = {}
            for k, v in ray_batch.items(): cur_ray_batch[k] = v[ri:ri + trn]

            with torch.no_grad():
                cur_ray_batch, near, far = process_ray_batch(cur_ray_batch, pose)
                human_poses = cur_ray_batch['human_poses']
                cur_outputs = self.render(cur_ray_batch, near, far, human_poses, is_train=False, step=300000)
                output['color'].append(cur_outputs['ray_rgb'].detach().cpu().numpy())
                output['albedo'].append(cur_outputs['albedo'].detach().cpu().numpy())
                output['roughness'].append(cur_outputs['roughness'].detach().cpu().numpy())
                output['normal'].append(cur_outputs['normal'].detach().cpu().numpy())
                output['normal_vis'].append(cur_outputs['normal_vis'].detach().cpu().numpy())

                output['occ_predict'].append(cur_outputs['occ_prob'].detach().cpu().numpy())
                output['occ_trace'].append(cur_outputs['occ_prob_gt'].detach().cpu().numpy())
                
                output['diff_color'].append(cur_outputs['diffuse_color'].detach().cpu().numpy())
                output['spec_color'].append(cur_outputs['specular_color'].detach().cpu().numpy())
                
                output['diff_light'].append(cur_outputs['diffuse_light'].detach().cpu().numpy())
                output['spec_light'].append(cur_outputs['specular_light'].detach().cpu().numpy())
                output['indirect_light'].append(cur_outputs['indirect_light'].detach().cpu().numpy())
                if self.cfg['has_radiance_field']:
                    output['radiance'].append(cur_outputs['radiance'].detach().cpu().numpy())
        for k in output:        
            val = np.concatenate(output[k], 0)
            output[k] = np.reshape(val, [h, w, val.shape[-1]])
        torch.set_default_tensor_type('torch.FloatTensor')
        return output

    def get_anneal_val(self, step):
        if self.cfg['anneal_end'] < 0:
            return 1.0
        else:
            return np.min([1.0, step / self.cfg['anneal_end']])

    def near_far_from_sphere(self, rays_o, rays_d):
        radius = self.radius if self.radius is not None else 1.0
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - radius
        far = mid + radius
        near = torch.clamp(near, min=1e-3)
        return near, far

    def compute_sample_level(self, pts):
        level = torch.zeros(pts.shape[:-1] + (1, ))
        return level

    def _process_ray_batch(self, ray_batch, poses):
        rays_d = ray_batch['dirs']  # rn,3
        idxs = ray_batch['idxs'][..., 0]  # rn

        rays_o = poses[:, :, :3].permute(0, 2, 1) @ -poses[:, :, 3:]  # trn,3,1
        rays_o = rays_o[idxs, :, 0]  # rn,3
        rays_d = poses[idxs, :, :3].permute(0, 2, 1) @ rays_d.unsqueeze(-1)
        rays_d = rays_d[..., 0]  # rn,3

        rays_o = rays_o
        rays_d = F.normalize(rays_d, dim=-1)
        near, far = self.near_far_from_sphere(rays_o, rays_d)

        ray_batch['rays_o'] = rays_o
        ray_batch['dirs'] = rays_d
        return ray_batch, near, far  # rn, 3, 4
    
    def _process_ray_batch_nerf(self, ray_batch, poses):
        rays_d = ray_batch['dirs']  # rn,3
        idxs = ray_batch['idxs'][..., 0]  # rn

        rays_o = poses[idxs, :3, -1] # rn,3
        rays_d = torch.sum(rays_d[..., None, :] * poses[idxs, :3, :3], -1)  # rn,3
        rays_d = F.normalize(rays_d, dim=-1)
        near, far = self.near_far_from_sphere(rays_o, rays_d)

        ray_batch['rays_o'] = rays_o
        ray_batch['dirs'] = rays_d
        return ray_batch, near, far # rn, 3, 4

    def test_step(self, index, step,):
        target_imgs_info, target_img_ids = self.test_imgs_info, self.test_ids
        imgs_info = imgs_info_slice(target_imgs_info, torch.from_numpy(np.asarray([index], np.int64)))
        gt_depth, gt_mask = self.database.get_depth(target_img_ids[index])  # used in evaluation
        if self.cfg['test_downsample_ratio']:
            imgs_info = imgs_info_downsample(imgs_info, self.cfg['downsample_ratio'])
            h, w = gt_depth.shape
            dh, dw = int(self.cfg['downsample_ratio'] * h), int(self.cfg['downsample_ratio'] * w)
            gt_depth, gt_mask = cv2.resize(gt_depth, (dw, dh), interpolation=cv2.INTER_NEAREST), \
                cv2.resize(gt_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
        gt_depth, gt_mask = torch.from_numpy(gt_depth), torch.from_numpy(gt_mask.astype(np.int32))
        if self.cfg['nerfDataType']:
            ray_batch, rn, h, w = self._construct_ray_batch_nerf(imgs_info, is_train=False)        
        else:        
            ray_batch, rn, h, w = self._construct_ray_batch(imgs_info)

        for k, v in ray_batch.items(): ray_batch[k] = v.cuda()

        trn = self.cfg['test_ray_num']
        outputs_keys = ['ray_rgb', 'gradient_error', 'depth', 'acc', 'normal_vis']
        outputs_keys += [
                'diffuse_albedo', 'diffuse_light', 'diffuse_color',
                'specular_albedo', 'specular_light', 'specular_color', 'specular_ref', 'specular_direct_light',
                'metallic', 'roughness', 'occ_prob', 'indirect_light', 'occ_prob_gt',
            ]
        if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
            outputs_keys.append('radiance')
            outputs_keys.append('roughness_weights')
        if self.color_network.cfg['human_light']:
            outputs_keys += ['human_light']
        outputs = {k: [] for k in outputs_keys}
        
        for ri in range(0, rn, trn):
            cur_ray_batch= {k:v[ri:ri+trn] for k, v in ray_batch.items()}
            rays_o, rays_d, human_poses = cur_ray_batch['rays_o'], cur_ray_batch['dirs'], cur_ray_batch['human_poses']
            near, far = self.near_far_from_sphere(rays_o, rays_d)
            
            cur_outputs = self.render(cur_ray_batch, near, far, human_poses, 0, 0, is_train=False, step=step)
            for k in outputs_keys: outputs[k].append(cur_outputs[k].detach())

        for k in outputs_keys: outputs[k] = torch.cat(outputs[k], 0)
        outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb'], ray_batch['rgbs'])
        outputs['gt_rgb'] = ray_batch['rgbs'].reshape(h, w, 3)
        outputs['ray_rgb'] = outputs['ray_rgb'].reshape(h, w, 3)
        if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
            outputs['loss_radiance'] = self.compute_rgb_loss(outputs['radiance'], ray_batch['rgbs']) * outputs['roughness_weights']
            outputs['loss_rgb'] = outputs['loss_rgb'] * (1.0 - outputs['roughness_weights'])
            outputs['radiance'] = outputs['radiance'].reshape(h, w, 3)

        # used in evaluation
        outputs['gt_depth'] = gt_depth.unsqueeze(-1)
        outputs['gt_mask'] = gt_mask.unsqueeze(-1)

        self.zero_grad()
        return outputs

    def train_step(self, step):
        
        mutual_reg = False
        self.sdf_pretrain = 50000
        # self.sdf_pretrain = 0
        # 消融设置为0
        
        if step > self.sdf_pretrain:
            mutual_reg = True
        # 默认为单独训练
        if mutual_reg is False:
            rn = self.cfg['train_ray_num']
            train_ray_batch = {k: v[self.train_batch_i:self.train_batch_i + rn].cuda() for k, v in self.train_batch.items()}
            self.train_batch_i += rn
            if self.train_batch_i + rn >= self.tbn: self._shuffle_train_batch()
            rays_o, rays_d = train_ray_batch['rays_o'], train_ray_batch['dirs']
            human_poses = None
            near, far = self.near_far_from_sphere(rays_o, rays_d)
            outputs = self.render(train_ray_batch, near, far, human_poses, -1, self.get_anneal_val(step), is_train=True, step=step)
            
            # 单独训练TensoSDF的渲染网络
            outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb'], train_ray_batch['rgbs'])  # ray_loss
            if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
                outputs['loss_radiance'] = self.compute_rgb_loss(outputs['radiance'], train_ray_batch['rgbs']) * outputs['roughness_weights']  # ray_loss
                outputs['loss_rgb'] = outputs['loss_rgb'] * (1.0 - outputs['roughness_weights'])
            if self.cfg['apply_mask_loss']:
                outputs['loss_mask'] = F.binary_cross_entropy(outputs['acc'].clip(1e-3, 1.0 - 1e-3), (train_ray_batch['masks'] > 0.5).float())

            return outputs
        
        # 联合训练分支
        
        else:

            def cos_similarity_loss(a, b, mask = None):
                if mask is None:
                    return 1.0-((a*b).sum(dim=-1) / (a.norm(dim=-1)*b.norm(dim=-1)+1e-8)).abs().mean()
                else:
                    return 1.0-((a*b).sum(dim=-1) / (a.norm(dim=-1)*b.norm(dim=-1)+1e-8))[mask].abs().mean()
                
            rn = self.cfg['train_ray_num']
            img_indices = list(self.train_batch_mutual_reg.keys())
            img_idx = randint(0, len(img_indices) - 1)  # 随机选择一张图像进行训练
            # print("img_idx", img_idx)
            img_data = self.train_batch_mutual_reg[img_idx]
            
            train_ray_batch = {}
            total_rays = img_data['rays_o'].shape[0]
            indices = random.sample(range(total_rays), rn)
            train_ray_batch = {k: v[indices].cuda() for k, v in img_data.items()}
            rays_o, rays_d = train_ray_batch['rays_o'], train_ray_batch['dirs']
            human_poses = None
            near, far = self.near_far_from_sphere(rays_o, rays_d)
            
            current_gs_step = step - self.sdf_pretrain + self.pretrain_it
            initial_stage = self.opt.initial
            # 使用深度引导采样的SDF
            outputs = self.render(train_ray_batch, near, far, human_poses, -1, self.get_anneal_val(step), is_train=True, step=step)
            # GS-branch training begin
            # --------------------------------------------------------------------------------
            # Control the init stage
            if current_gs_step > self.opt.init_until_iter:
                initial_stage = False
            
            self.gaussians.update_learning_rate(current_gs_step)
            # Initialize envmap
            if not initial_stage:
                if current_gs_step <= self.opt.volume_render_until_iter:
                    envmap2 = self.gaussians.get_envmap_2 
                    envmap2.build_mips()
                else:
                    envmap = self.gaussians.get_envmap 
                    envmap.build_mips()
            
            # GS-branch training end
            # --------------------------------------------------------------------------------
            
            viewpoint_cam = self.viewpoint_stack[train_ray_batch['used_index'][0]]
            # Get the same pixel indexes as sdf representation
            yy = train_ray_batch['used_y']
            xx = train_ray_batch['used_x']       
            debug_dir = './debug_outputs'
            # os.makedirs(debug_dir, exist_ok=True)
            # retain_grad = (current_gs_step < self.opt.update_until and current_gs_step >= 0)
            render = select_render_method(current_gs_step, self.opt, initial_stage=False)
            
            bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            render_pkg = render(viewpoint_cam, self.gaussians, self.pp, background, srgb=self.opt.srgb, opt=self.opt)
            image, viewspace_point_tensor, visibility_filter, radii, gs_depth_hand, gs_normal = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["surf_depth"], render_pkg["rend_normal"]
    
            # masks_info = self.train_imgs_info['masks'][img_idx].cuda()  # [640000,1]
            masks_info = None
            total_loss, tb_dict = calculate_loss(viewpoint_cam, self.gaussians, render_pkg, self.opt, current_gs_step, masks_info)
            
            # bidirectional normal supervision
            # debug_dir = './debug_outputs'
            # gs_normal_detach_for_sdf = (gs_normal.permute(1, 2, 0))* 0.5 + 0.5 # [800,800,3]
            gs_normal_detach_for_sdf = (gs_normal.permute(1, 2, 0))
            # save_image(gs_normal.permute(2,0,1), os.path.join(debug_dir, "gs_normal.png"))
            
            
            picked_gs_normal_detach_for_sdf = gs_normal_detach_for_sdf[yy, xx]
            
            
            # gs_normal = gs_normal.permute(1, 2, 0)* 0.5 + 0.5 # [800,800,3]
            # picked_gs_normal = gs_normal[yy, xx]
            # sdf_normal = np.clip(((outputs['normal'] + 1.0) * 0.5),a_min=0,a_max=255).astype(np.uint8)
            sdf_normal = outputs['normal'].clone().detach()
            total_loss += cos_similarity_loss(sdf_normal, picked_gs_normal_detach_for_sdf) * 0.5
            
            def get_outside_msk():
                return None if not self.USE_ENV_SCOPE else torch.sum((self.gaussians.get_xyz - self.ENV_CENTER[None])**2, dim=-1) > self.ENV_RADIUS**2
        
            if self.USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
                refls = self.gaussians.get_refl
                refl_msk_loss = refls[get_outside_msk()].mean()
                total_loss += self.REFL_MSK_LOSS_W * refl_msk_loss
            
            total_loss.backward()
            
            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
            
            # gs_alpha = render_pkg['render_alpha']
            # save_image(gs_alpha.permute(2,0,1), os.path.join(debug_dir, "gs_alpha.png"))
            with torch.no_grad():
                if current_gs_step % 1000 == 0:
                    save_training_vis(viewpoint_cam, self.gaussians, background, render, self.pp, self.opt, current_gs_step, initial_stage, self.args)
                    # save_training_vis_split(viewpoint_cam, self.gaussians, background, render, self.pp, self.opt, current_gs_step, initial_stage, self.args)
            
            # gs_depth = gs_depth_hand.mean(dim=0, keepdim=True).permute(1, 2, 0)
            # picked_gs_depth = gs_depth[yy, xx]
            # gs_normal = (gs_normal.permute(1, 2, 0).detach())* 0.5 + 0.5 # [800,800,3]
            # gs_normal_f = F.normalize(gs_normal, dim=-1)* 0.5 + 0.5 # [800,800,3]
            # gs_normal = (gs_normal.permute(1, 2, 0).detach()) 
            # picked_gs_normal = gs_normal[yy, xx]
            # picked_gs_depth_dt = picked_gs_depth.detach()

            # diff_depth = torch.abs(outputs['depth'] - picked_gs_depth_dt)
            # outputs['loss_depth_mutual'] = diff_depth.mean()  # 深度损失
            # save_image(gs_normal.permute(2,0,1), os.path.join(debug_dir, "gs_normal.png"))
            # save_image(gs_normal.permute(2,0,1), os.path.join(debug_dir, f"gs_normal{step}.png"))
            # save_image(gs_normal_f.permute(2,0,1), os.path.join(debug_dir, f"gs_normal_f{step}.png"))
            
            if step > 100000:
                diff_normal = cos_similarity_loss(outputs['normal'], picked_gs_normal_detach_for_sdf.detach(), outputs['entropy_mask'].squeeze())
            else:
                diff_normal = cos_similarity_loss(outputs['normal'], picked_gs_normal_detach_for_sdf.detach())
                                                                                                                              
            outputs['loss_normal_mutual'] = diff_normal.mean()
            
            outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb'], train_ray_batch['rgbs'])  # ray_loss
            if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
                outputs['loss_radiance'] = self.compute_rgb_loss(outputs['radiance'], train_ray_batch['rgbs']) * outputs['roughness_weights']  # ray_loss
                outputs['loss_rgb'] = outputs['loss_rgb'] * (1.0 - outputs['roughness_weights'])
            if self.cfg['apply_mask_loss']:
                outputs['loss_mask'] = F.binary_cross_entropy(outputs['acc'].clip(1e-3, 1.0 - 1e-3), (train_ray_batch['masks'] > 0.5).float())
            
            # # GS-branch training loss to logs
            # # ------------------------------------------------------------------------------------------
            
            with torch.no_grad():
                
                if current_gs_step % self.TEST_INTERVAL == 0 or current_gs_step == self.opt.volume_render_until_iter + 1:
                    save_training_vis(viewpoint_cam, self.gaussians, background, render, self.pp, self.opt, current_gs_step, initial_stage, self.args)
                
                if current_gs_step in self.args.save_iterations:
                    print(f"\n[ITER {current_gs_step}] Saving Gaussians")
                    self.scene.save(current_gs_step)

                self.HAS_RESET0 = False
                    
                # Densification
                # 默认配置是25000次迭代才停止致密和减枝 18000次迭代时体渲染模式截止
                if current_gs_step < self.opt.densify_until_iter and current_gs_step != self.opt.volume_render_until_iter:
                    
                    # 可见性掩码为visibility_filter标记哪些高斯点在当前视图可见 更新最大半径
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter],
                                                                        radii[visibility_filter])
                    # 收集致密化统计信息 
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # 初始化阶段(不一定有) 
                    if current_gs_step <= self.opt.init_until_iter:
                        opacity_reset_intval = 3000 # 不透明度重置间隔
                        densification_interval = 100 # 致密化间隔
                    elif current_gs_step <= self.opt.normal_prop_until_iter :
                        opacity_reset_intval = 3000 
                        densification_interval = self.opt.densification_interval_when_prop
                    else:
                        opacity_reset_intval = 3000
                        densification_interval = 100

                    if current_gs_step > self.opt.densify_from_iter and current_gs_step % densification_interval == 0:
                        # 记录加密前的点数
                        pre_count = self.gaussians.get_xyz.shape[0]
                        
                        with torch.no_grad():
                            gs_center = self.gaussians.get_xyz
                            gs_level = self.compute_sample_level(gs_center)
                            # 获取 SDF 并切断梯度
                            center_sdf = self.sdf_network.sdf_for_gs(gs_center, gs_level).detach()
                            
                        size_threshold = 20 if current_gs_step > self.opt.opacity_reset_interval else None
                        
                        # 调用内部优化后的函数
                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, 
                            self.opt.prune_opacity_threshold, 
                            self.scene.cameras_extent,
                            size_threshold, 
                            sdf_values=center_sdf
                        )
                        
                        # 打印监控信息
                        post_count = self.gaussians.get_xyz.shape[0]
                        print(f"[Step {current_gs_step}] Densification: {pre_count} -> {post_count} points.")
                        
                        torch.cuda.empty_cache()
                        # self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.prune_opacity_threshold, self.scene.cameras_extent,
                        #                             size_threshold, sdf_values=None)
                    self.HAS_RESET0 = False
                    if current_gs_step % opacity_reset_intval == 0 or (self.dataset.white_background and current_gs_step == self.opt.densify_from_iter):
                        self.HAS_RESET0 = True
                        outside_msk = get_outside_msk()
                        self.gaussians.reset_opacity0()
                        self.gaussians.reset_refl(exclusive_msk=outside_msk)
                    if self.opt.opac_lr0_interval > 0 and (
                            self.opt.init_until_iter < current_gs_step <= self.opt.normal_prop_until_iter ) and current_gs_step % self.opt.opac_lr0_interval == 0:
                        self.gaussians.set_opacity_lr(self.opt.opacity_lr)
                    if (self.opt.init_until_iter < current_gs_step <= self.opt.normal_prop_until_iter ) and current_gs_step % self.opt.normal_prop_interval == 0:
                        if not self.HAS_RESET0:
                            outside_msk = get_outside_msk()
                            self.gaussians.reset_opacity1(exclusive_msk=outside_msk)
                            if current_gs_step > self.opt.volume_render_until_iter and self.opt.volume_render_until_iter > self.opt.init_until_iter:
                                self.gaussians.dist_color(exclusive_msk=outside_msk)
                                # self.gaussians.dist_albedo(exclusive_msk=outside_msk)

                            self.gaussians.reset_scale(exclusive_msk=outside_msk)
                            if self.opt.opac_lr0_interval > 0 and current_gs_step != self.opt.normal_prop_until_iter :
                                self.gaussians.set_opacity_lr(0.0)
                
                if (current_gs_step >= self.opt.indirect_from_iter and current_gs_step % self.MESH_EXTRACT_INTERVAL == 0) or current_gs_step == (self.opt.indirect_from_iter):
                    if not self.HAS_RESET0:
                        self.gaussExtractor.reconstruction(self.scene.getTrainCameras())
                        if 'ref_real' in self.dataset.source_path:
                            mesh = self.gaussExtractor.extract_mesh_unbounded(resolution=self.opt.mesh_res)
                        else:
                            depth_trunc = (self.gaussExtractor.radius * 2.0) if self.opt.depth_trunc < 0  else self.opt.depth_trunc
                            voxel_size = (depth_trunc / self.opt.mesh_res) if self.opt.voxel_size < 0 else self.opt.voxel_size
                            sdf_trunc = 5.0 * voxel_size if self.opt.sdf_trunc < 0 else self.opt.sdf_trunc
                            mesh = self.gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                        mesh = post_process_mesh(mesh, cluster_to_keep=self.opt.num_cluster)
                        ply_path = os.path.join(self.args.model_path,f'test_{current_gs_step:06d}.ply')
                        o3d.io.write_triangle_mesh(ply_path, mesh)
                        self.gaussians.update_mesh(mesh)

                if current_gs_step < self.pretrain_it:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

                if current_gs_step in self.args.checkpoint_iterations:
                    print(f"\n[ITER {current_gs_step}] Saving Checkpoint")
                    torch.save((self.gaussians.capture(), current_gs_step), self.scene.model_path + f"/chkpnt{current_gs_step}.pth")
                    
            # ------------------------------------------------------------------------------------------
            
            return outputs


    def compute_rgb_loss(self, rgb_pr, rgb_gt):
        if self.cfg['rgb_loss'] == 'l2':
            rgb_loss = torch.sum((rgb_pr - rgb_gt) ** 2, -1)
        elif self.cfg['rgb_loss'] == 'l1':
            rgb_loss = torch.sum(F.l1_loss(rgb_pr, rgb_gt, reduction='none'), -1)
        elif self.cfg['rgb_loss'] == 'smooth_l1':
            rgb_loss = torch.sum(F.smooth_l1_loss(rgb_pr, rgb_gt, reduction='none', beta=0.25), -1)
        elif self.cfg['rgb_loss'] == 'charbonier':
            epsilon = 0.001
            rgb_loss = torch.sqrt(torch.sum((rgb_gt - rgb_pr) ** 2, dim=-1) + epsilon)
        else:
            raise NotImplementedError
        return rgb_loss

    def density_activation(self, density, dists):
        return 1.0 - torch.exp(-F.softplus(density) * dists)

    def compute_density(self, points):
        points_norm = torch.norm(points, dim=-1, keepdim=True)
        points_norm = torch.clamp(points_norm, min=1e-3)
        sigma = self.outer_nerf.density(torch.cat([points / points_norm, 1.0 / points_norm], -1))[..., 0]
        return sigma

    @staticmethod
    def upsample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        # 计算SDF变化率
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        # z_vals.shape:[1024, 64], weights.shape:[1024, 63],n_importance=16
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        level = self.compute_sample_level(pts) # [rn, sn, 1]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3), level).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def sample_ray(self, rays_o, rays_d, near, far, perturb):
        n_samples = self.cfg['n_samples'] 
        n_bg_samples = self.cfg['n_bg_samples']
        n_importance = self.cfg['n_importance']
        up_sample_steps = self.cfg['up_sample_steps']

        # sample points
        batch_size = len(rays_o)
        z_vals = torch.linspace(0.0, 1.0, n_samples)  # sn
        
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near[..., 0], max=far[..., 0]).unsqueeze(-1)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near[..., 0], max=far[..., 0]).unsqueeze(-1)
        
        z_vals = t_min + (t_max - t_min) * z_vals[None, :]  # rn,sn
        if n_bg_samples > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (n_bg_samples + 1.0), n_bg_samples)

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / n_samples

            if n_bg_samples > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if n_bg_samples > 0:
            z_vals_outside = t_max / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / n_bg_samples

        # Up sample
        with torch.no_grad():
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None] # [rn, sn, 3]
            level = self.compute_sample_level(pts) # [rn, sn, 1]
            sdf = self.sdf_network.sdf(pts, level).reshape(batch_size, n_samples)

            for i in range(up_sample_steps):
                rn, sn = z_vals.shape
                if self.cfg['clip_sample_variance']:
                    inv_s = self.deviation_network(torch.empty([1, 3])).expand(rn, sn - 1)
                    inv_s = torch.clamp(inv_s, max=64 * 2 ** i)  # prevent too large inv_s
                else:
                    inv_s = torch.ones(rn, sn - 1) * 64 * 2 ** i
                new_z_vals = self.upsample(rays_o, rays_d, z_vals, sdf, n_importance // up_sample_steps, inv_s)
                z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == up_sample_steps))

        if n_bg_samples > 0:
            z_vals = torch.cat([z_vals, z_vals_outside], -1)
        # [1024,128] 
        return z_vals


    def _compute_depth_weights(self, z_vals, guided_depth, sigma=10.0):
        """计算深度引导权重"""
        mid_z_vals = (z_vals[..., :-1] + z_vals[..., 1:]) * 0.5
        depth_diff = torch.abs(mid_z_vals - guided_depth)
        return torch.exp(-depth_diff * sigma)

    def _compute_base_weights(self, sdf, z_vals,rays_o, rays_d, inv_s):
        """计算基础的SDF权重"""
        
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        # 计算SDF变化率
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        
        return weights

    def _sample_far_background(self, far, n_samples):
        """深度引导远景背景采样"""
        far_start = far
        far_end = far * 5.0  # 5倍远平面距离
        return torch.linspace(0, 1, n_samples, device=far.device) * (far_end - far_start) + far_start

    
    def render(self, ray_batch, near, far, human_poses, perturb_overwrite=-1, cos_anneal_ratio=0.0, is_train=True, step=None):
        """
        :param ray_batch: rn,x
        :param near:   rn,1
        :param far:    rn,1
        :param human_poses:     rn,3,4
        :param perturb_overwrite: set 0 for inference
        :param cos_anneal_ratio:
        :param is_train:
        :param step:
        :return:
        """
        perturb = self.cfg['perturb']
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        rays_o, rays_d = ray_batch['rays_o'], ray_batch['dirs']
        z_vals = self.sample_ray(rays_o, rays_d, near, far, perturb)
        ret = self.render_core(rays_o, rays_d, z_vals, human_poses, cos_anneal_ratio=cos_anneal_ratio, step=step, is_train=is_train)
        return ret

    def compute_validation_info(self, z_vals, rays_o, rays_d, weights, human_poses, step):
        depth = torch.sum(weights * z_vals, -1, keepdim=True)  # rn, 1
        points = depth * rays_d + rays_o  # rn,3
        level = self.compute_sample_level(points) # [rn, 1]
        gradients, _ = self.sdf_network.gradient(points, level, training=False)  # rn,3
        outer_mask = ((self.aabb[0]>points) | (points>self.aabb[1])).any(dim=-1)
        inner_mask = ~outer_mask[..., None]

        outputs = {
            'depth': depth,  # rn,1
        }

        if not self.cfg['nerfDataType']:
            outputs['normal_vis'] = ((F.normalize(gradients, dim=-1) + 1.0) * 0.5) * inner_mask

        feature_vector = self.sdf_network(points, level)[..., 1:]  # rn,f
        _, occ_info, inter_results = self.color_network(points, gradients, -F.normalize(rays_d, dim=-1), feature_vector, human_poses, inter_results=True, step=step)
        _, occ_prob, _ = get_intersection(self.sdf_inter_fun, self.deviation_network, points, occ_info['reflective'], sn0=128, sn1=9)  # pn,sn-1
        occ_prob_gt = torch.sum(occ_prob, dim=-1, keepdim=True)
        outputs['occ_prob_gt'] = occ_prob_gt
        for k, v in inter_results.items(): inter_results[k] = v * inner_mask
        outputs.update(inter_results)
        return outputs 
        
    def compute_sdf_alpha(self, points, level, dists, dirs, cos_anneal_ratio, step, is_train):
        # points [...,3] dists [...] dirs[...,3]
        sdf_nn_output = self.sdf_network(points, level)
        sdf = sdf_nn_output[..., 0]
        feature_vector = sdf_nn_output[..., 1:]

        gradients, hessian = self.sdf_network.gradient(points, level, training=is_train, sdf=sdf[..., None])  # ...,3
        inv_s = self.deviation_network(points).clip(1e-6, 1e6)  # ...,1
        inv_s = inv_s[..., 0]

        if self.cfg['freeze_inv_s_step'] is not None and step < self.cfg['freeze_inv_s_step']:
            inv_s = inv_s.detach()

        true_cos = (dirs * gradients).sum(-1)  # [...]
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)  # [...]
        return alpha, gradients, feature_vector, inv_s, sdf, hessian

    def compute_density_alpha(self, points, dists, dirs, nerf):
        norm = torch.norm(points, dim=-1, keepdim=True)
        points = torch.cat([points / norm, 1.0 / norm], -1)
        density, color = nerf(points, dirs)  # [...,1] [...,3]
        alpha = self.density_activation(density[..., 0], dists)
        color = linear_to_srgb(torch.exp(torch.clamp(color, max=5.0)))
        return alpha, color

    def compute_occ_loss(self, occ_info, points, sdf, gradients, dirs, step):
        if step < self.cfg['occ_loss_step']: return torch.zeros(1)

        occ_prob = occ_info['occ_prob']
        reflective = occ_info['reflective']

        # select a subset for occ loss
        # note we only apply occ loss on the surface
        outer_mask = ((self.aabb[0]>points) | (points>self.aabb[1])).any(dim=-1)
        inner_mask = ~outer_mask
        
        sdf_mask = torch.abs(sdf) < self.cfg['occ_sdf_thresh']
        normal_mask = torch.sum(gradients * dirs, -1) < 0  # pn
        mask = (inner_mask & normal_mask & sdf_mask)

        if torch.sum(mask) > self.cfg['occ_loss_max_pn']:
            indices = torch.nonzero(mask)[:, 0]  # npn
            idx = torch.randperm(indices.shape[0], device='cuda')  # npn
            indices = indices[idx[:self.cfg['occ_loss_max_pn']]]  # max_pn
            mask_new = torch.zeros_like(mask)
            mask_new[indices] = 1
            mask = mask_new

        if torch.sum(mask) > 0:
            inter_dist, inter_prob, inter_sdf = get_intersection(self.sdf_inter_fun, self.deviation_network, points[mask], reflective[mask], sn0=64, sn1=16)  # pn,sn-1
            occ_prob_gt = torch.sum(inter_prob, -1, keepdim=True)
            return F.l1_loss(occ_prob[mask], occ_prob_gt)
        else:
            return torch.zeros(1)

    def render_core(self, rays_o, rays_d, z_vals, human_poses, cos_anneal_ratio=0.0, step=None, is_train=True):
        batch_size, n_samples = z_vals.shape

        # section length in original space
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # rn,sn-1
        dists = torch.cat([dists, dists[..., -1:]], -1)  # rn,sn
        mid_z_vals = z_vals + dists * 0.5
        
        points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * mid_z_vals.unsqueeze(-1) # [rn, sn, 3]
        level = self.compute_sample_level(points) # [rn, sn, 1]
        derived_normals = torch.zeros(batch_size, n_samples, 3)
        entropy_weight = torch.zeros(batch_size, n_samples, 1)

        outer_mask = ((self.aabb[0]>points) | (points>self.aabb[1])).any(dim=-1)
        inner_mask = ~outer_mask      

        dirs = rays_d.unsqueeze(-2).expand(batch_size, n_samples, 3)
        # human_poses_pt = human_poses.unsqueeze(-3).expand(batch_size, n_samples, 3, 4)
        dirs = F.normalize(dirs, dim=-1)
        alpha, sampled_color = torch.zeros(batch_size, n_samples), torch.zeros(batch_size, n_samples, 3)
        if self.cfg['predict_BG'] and torch.sum(outer_mask) > 0:
            alpha[outer_mask], sampled_color[outer_mask] = self.compute_density_alpha(points[outer_mask], dists[outer_mask], -dirs[outer_mask], self.outer_nerf)
        
        if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
            sampled_radiance = torch.zeros(batch_size, n_samples, 3)
            roughness = torch.zeros(batch_size, n_samples, 1)

        alpha_rest_ratio = 1.0
        if self.alphaMask is not None:
            alpha_mask = self.alphaMask.sample_alpha(points[inner_mask]) > 0
            alpha_rest_ratio = 1.0 - torch.sum(~alpha_mask) / torch.sum(inner_mask)
            inner_mask_invalid = ~inner_mask
            inner_mask_invalid[inner_mask] |= (~alpha_mask)
            inner_mask = ~inner_mask_invalid 
                
        if torch.sum(inner_mask) > 0:                
            alpha[inner_mask], gradients, feature_vector, inv_s, sdf, hessian = self.compute_sdf_alpha(points[inner_mask], level[inner_mask], dists[inner_mask], dirs[inner_mask], cos_anneal_ratio, step, is_train)
            valid_normals = gradients
            derived_normals[inner_mask] = gradients
            if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
                sampled_color[inner_mask], sampled_radiance[inner_mask], occ_info, entropy_weight_points = self.color_network(points[inner_mask], valid_normals, -dirs[inner_mask], feature_vector, #human_poses_pt[inner_mask], 
                                                                                                       None ,step=step)      
                roughness[inner_mask] = occ_info['roughness']      
            else:
                sampled_color[inner_mask], _, occ_info, entropy_weight_points = self.color_network(points[inner_mask], valid_normals, -dirs[inner_mask], feature_vector, #human_poses_pt[inner_mask], 
                                                                            step=step)
            
            entropy_weight[inner_mask] = entropy_weight_points.unsqueeze(-1)
            # Eikonal loss
            gradient_error = (torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2
            
            if self.cfg['apply_sparse_loss']:
                gamma = 20.
                reg_loss = torch.exp(-gamma * sdf.abs())        # [..., ]
                reg_loss = reg_loss.sum() / (inner_mask.sum() + 1e-5) * alpha_rest_ratio
                
            if self.cfg['apply_hessian_loss'] and hessian is not None:
                hessian_loss = hessian.abs().sum() / (inner_mask.sum() + 1e-5) * alpha_rest_ratio
            else:
                hessian_loss = torch.zeros(1)
        else:
            gradient_error = torch.zeros(1)
            if self.cfg['apply_sparse_loss']:
                reg_loss = torch.zeros(1)
            if self.cfg['apply_hessian_loss']:
                hessian_loss = torch.zeros(1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[..., :-1]  # rn,sn
        
        depth_map = torch.sum(weights * mid_z_vals, dim=-1, keepdim=True)  # rn,1
        
        acc_map = torch.sum(weights, dim=-1, keepdim=True) # rn, 1          
        if not self.cfg['predict_BG'] and self.cfg['isBGWhite']:
            color = (sampled_color * weights[..., None]).sum(dim=1) + (1 - acc_map)
        else:
            color = (sampled_color * weights[..., None]).sum(dim=1)
            
        entropy_mask = (entropy_weight * weights[..., None]).sum(dim=1)
        entropy_mask = (entropy_mask < 0.1)
        outputs = {
            'ray_rgb': color,  # rn,3
            'gradient_error': gradient_error,  # rn
            'acc' : acc_map,
            'depth': depth_map,  # rn,1
            'entropy_mask': entropy_mask, #rn,1
        }

        acc_sampled_normal = (derived_normals * weights[..., None]).sum(dim=1)
        # outputs['normal'] = F.normalize(acc_sampled_normal * acc_map + (1. - acc_map) * torch.tensor([0.0, 0.0, 1.0], device=acc_sampled_normal.device), dim=-1)
        
        normal_normalized = F.normalize(acc_sampled_normal, dim=-1, eps=1e-6)
        outputs['normal'] = normal_normalized * acc_map
        
        if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
            if not self.cfg['predict_BG'] and self.cfg['isBGWhite']:
                radiance = (sampled_radiance * weights[..., None]).sum(dim=1) + (1 - acc_map)
            else:
                radiance = (sampled_radiance * weights[..., None]).sum(dim=1)
            roughness_weights = (roughness * weights[..., None]).sum(dim=1) # rn, 1
            outputs['radiance'] = radiance
            outputs['roughness_weights'] = roughness_weights.squeeze(-1).clone().detach() # rn

        if torch.sum(inner_mask) > 0:
            outputs['std'] = torch.mean(1 / inv_s)
        else:
            outputs['std'] = torch.zeros(1)

        if step < 1000:
            if torch.sum(inner_mask) > 0:
                outputs['sdf_pts'] = points[inner_mask]
                outputs['sdf_vals'] = self.sdf_network.sdf(points[inner_mask], level[inner_mask])[..., 0]
            else:
                outputs['sdf_pts'] = torch.zeros(1)
                outputs['sdf_vals'] = torch.zeros(1)

        if self.cfg['apply_occ_loss']:
            # occlusion loss
            if torch.sum(inner_mask) > 0:
                outputs['loss_occ'] = self.compute_occ_loss(occ_info, points[inner_mask], sdf, valid_normals, dirs[inner_mask], step)
            else:
                outputs['loss_occ'] = torch.zeros(1)

        if self.cfg['apply_gaussian_loss'] and step > self.cfg['gaussianLoss_step']:
            # gaussian loss
            if torch.sum(inner_mask) > 0:
                outputs['loss_gaussian'] = self.sdf_network.grid_gaussian_loss()
            else:
                outputs['loss_gaussian'] = torch.zeros(1)

        if self.cfg['apply_tv_loss']:
            outputs['loss_tv_sdf'] = self.sdf_network.TV_loss_sdf(self.tv_reg)          

        if self.cfg['apply_sparse_loss']:
            outputs['loss_sparse'] = reg_loss

        if self.cfg['apply_hessian_loss']:
            outputs['loss_hessian'] = hessian_loss

        if not is_train:
            outputs['normal_vis'] = ((outputs['normal'] + 1.0) * 0.5) * acc_map + (1. - acc_map)
            outputs.update(self.compute_validation_info(z_vals, rays_o, rays_d, weights, human_poses, step))

        return outputs

    def forward(self, data):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        is_train = 'eval' not in data
        step = data['step']

        if is_train:
            outputs = self.train_step(step)
        else:
            index = data['index']
            outputs = self.test_step(index, step=step)

            if index == 0 and self.cfg['val_geometry']:
                bbox_min = -torch.ones(3)
                bbox_max = torch.ones(3)
                vertices, triangles = extract_geometry(bbox_min, bbox_max, 128, 0, lambda x: self.sdf_network.sdf(x))
                outputs['vertices'] = vertices
                outputs['triangles'] = triangles

        torch.set_default_tensor_type('torch.FloatTensor')
        return outputs

    def predict_materials(self):
        name = self.cfg['name']
        mesh = open3d.io.read_triangle_mesh(f'data/meshes/{name}-300000.ply')
        xyz = np.asarray(mesh.vertices)
        xyz = torch.from_numpy(xyz.astype(np.float32)).cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        metallic, roughness, albedo = [], [], []
        batch_size = 8192
        for vi in range(0, xyz.shape[0], batch_size):
            feature_vectors = self.sdf_network(xyz[vi:vi + batch_size])[:, 1:]
            m, r, a = self.color_network.predict_materials(xyz[vi:vi + batch_size], feature_vectors)
            metallic.append(m.cpu().numpy())
            roughness.append(r.cpu().numpy())
            albedo.append(a.cpu().numpy())

        return {'metallic': np.concatenate(metallic, 0),
                'roughness': np.concatenate(roughness, 0),
                'albedo': np.concatenate(albedo, 0)}
        
def select_render_method(iteration, opt, initial_stage):

    if initial_stage:
        render = render_initial
    # 18000次前使用体渲染
    elif iteration <= opt.volume_render_until_iter:
        render = render_volume
    # 18000次后使用表面渲染
    else:   
        render = render_surfel

    return render


def set_gaussian_para(gaussians, opt, vol=False):
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr 
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_refl_value = opt.init_refl_value
    gaussians.refl_msk_thr = opt.refl_msk_thr

def reset_gaussian_para(gaussians, opt):
    gaussians.reset_ori_color()
    gaussians.reset_refl_strength(opt.init_refl_value)
    gaussians.reset_roughness(opt.init_roughness_value)
    gaussians.refl_msk_thr = opt.refl_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr
    

def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration, initial_stage, args):
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)

        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])

        if initial_stage:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),
                render_pkg["render"], 
                render_pkg["rend_alpha"].repeat(3, 1, 1),
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5, 
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                error_map 
            ]

        elif iteration <= opt.volume_render_until_iter:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"], 
                render_pkg["base_color_map"], 
                render_pkg["diffuse_map"],      
                render_pkg["specular_map"],  
                render_pkg["refl_strength_map"].repeat(3, 1, 1),  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]), 
                render_pkg["rend_normal"] * 0.5 + 0.5,  
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                error_map
            ]
            if opt.indirect:
                visualization_list += [
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["direct_light"],
                    render_pkg["indirect_light"],
                ]

        else:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"],  
                render_pkg["base_color_map"],  
                render_pkg["diffuse_map"],
                render_pkg["specular_map"],
                render_pkg["refl_strength_map"].repeat(3, 1, 1),  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5,  
                render_pkg["surf_normal"] * 0.5 + 0.5,  
                error_map, 
            ]
  

        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 800
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        if not initial_stage:
            if opt.volume_render_until_iter > opt.init_until_iter and iteration <= opt.volume_render_until_iter:
                env_dict = gaussians.render_env_map_2() 
            else:
                env_dict = gaussians.render_env_map()

            grid = [
                env_dict["env1"].permute(2, 0, 1),
                env_dict["env2"].permute(2, 0, 1),
            ]
            grid = make_grid(grid, nrow=1, padding=10)
            save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderkwargs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()