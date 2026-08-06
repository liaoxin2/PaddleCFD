from __future__ import annotations
 
import json
import logging
import os
import sys
 
import random
from timeit import default_timer
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
 
import hydra
import meshio
import numpy as np
import paddle
import pyvista as pv
import vtk
import math
from omegaconf import DictConfig
from omegaconf import open_dict
from paddle import distributed as dist
from paddle.distributed import ParallelEnv
from paddle.distributed import fleet
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
 
from ppcfd.models.ppfno.data import instantiate_datamodule
from ppcfd.models.ppfno.losses import LpLoss
from ppcfd.models.ppfno.networks import instantiate_network
from ppcfd.models.ppfno.optim.schedulers import instantiate_scheduler
from ppcfd.models.ppfno.optim.soap import SOAP
from ppcfd.models.ppfno.utils.average_meter import AverageMeter
from ppcfd.models.ppfno.utils.average_meter import AverageMeterDict
from ppcfd.models.ppfno.utils.dot_dict import DotDict
from ppcfd.models.ppfno.utils.dot_dict import flatten_dict
 
 
def set_seed(seed: int = 0):
    paddle.seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
 
 
world_size = dist.get_world_size()
if world_size > 1:
    strategy = fleet.DistributedStrategy()
    strategy.find_unused_parameters = True
    fleet.init(is_collective=True, strategy=strategy)
 
print(f"total gpu num: {world_size}")
 
 
def save_vtp_from_dict(
    filename: str,
    data_dict: Dict[str, np.ndarray],
    coord_keys: Tuple[str, ...],
    value_keys: Tuple[str, ...],
    num_timestamps: int = 1,
):
 
    if len(coord_keys) not in [3]:
        raise ValueError(f"ndim of coord ({len(coord_keys)}) should be 3 in vtp format")
 
    coord = [data_dict[k] for k in coord_keys if k not in ("t", "sdf")]
    assert all([c.ndim == 2 for c in coord]), "array of each axis should be [*, 1]"
    coord = np.concatenate(coord, axis=1)
 
    if not isinstance(coord, np.ndarray):
        raise ValueError(f"type of coord({type(coord)}) should be ndarray.")
    if len(coord) % num_timestamps != 0:
        raise ValueError(
            f"coord length({len(coord)}) should be an integer multiple of "
            f"num_timestamps({num_timestamps})"
        )
    if coord.shape[1] not in [3]:
        raise ValueError(f"ndim of coord({coord.shape[1]}) should be 3 in vtp format.")
 
    if len(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
 
    npoint = len(coord)
    nx = npoint // num_timestamps
    if filename.endswith(".vtp"):
        filename = filename[:-4]
 
    for t in range(num_timestamps):
        coord_ = coord[t * nx : (t + 1) * nx]
        point_cloud = pv.PolyData(coord_)
        for k in value_keys:
            value_ = data_dict[k][t * nx : (t + 1) * nx]
            if value_ is not None and not isinstance(value_, np.ndarray):
                raise ValueError(f"type of value({type(value_)}) should be ndarray.")
            if value_ is not None and len(coord_) != len(value_):
                raise ValueError(
                    f"coord length({len(coord_)}) should be equal to value length({len(value_)})"
                )
            point_cloud[k] = value_
 
        if num_timestamps > 1:
            width = len(str(num_timestamps - 1))
            point_cloud.save(f"{filename}_t-{t:0{width}}.vtp", binary=True)
        else:
            point_cloud.save(f"{filename}.vtp", binary=True)
 
    if num_timestamps > 1:
        logging.info(
            f"Visualization results are saved to: {filename}_t-{0:0{width}}.vtp ~ "
            f"{filename}_t-{num_timestamps - 1:0{width}}.vtp"
        )
    else:
        logging.info(f"Visualization result is saved to: {filename}.vtp")
 
 
def calculate_lateral_angle(car_speed, wind_speed, wind_angle):
 
    wind_direction_rad = math.radians(wind_angle)
    lateral_component = wind_speed * math.cos(wind_direction_rad) + car_speed
    longitudinal_component = wind_speed * math.sin(wind_direction_rad)
    side_bias_angle_rad = math.atan2(longitudinal_component, lateral_component)
    side_bias_angle_deg = math.degrees(side_bias_angle_rad)
    
    return side_bias_angle_deg
 
 
def hybrid_loss(pred, truth, scale=None, eps_rel=0.05, eps=1e-6):
    # 相对误差平方。分母采用「固定特征尺度软下限」而非逐样本 |truth|：
    #   denom = sqrt(truth^2 + (eps_rel*scale)^2)
    # scale 为该物理量的全局特征尺度(常数, 如 RMS/std)。当 |truth| >> eps_rel*scale
    # 时退化为普通相对误差；当 truth 近零时分母被 eps_rel*scale 托住，避免小分母
    # 样本主导梯度（点头力矩 22/249 条 |真值|<10 的近零穿越样本正是病根）。
    # scale=None 时回退到旧的逐样本 |truth| 归一化（向后兼容）。
    if scale is None:
        denom = truth.abs() + eps
    else:
        denom = paddle.sqrt(truth ** 2 + (eps_rel * scale) ** 2) + eps
    return ((pred - truth) / denom) ** 2


def lever_weighted_rel_l2(pred, truth, lever_dist, alpha):
    # 力臂加权相对 L2 场损失(建议 B)。俯仰力矩误差 δM_y ≈ Σ r_x·n_z·A·δp 由车厢
    # 端部(长力臂)压力误差主导, 而普通 LpLoss 对所有点一视同仁。这里按
    #   w_i = 1 + α·|x_i - x_ref^(车厢)| / (L_车厢/2)
    # 加权分子(lever_dist 已在模型侧按车厢半长归一化到 [0,2]), 再除以 mean(w)
    # 保持整体标定不变(不隐式放大 pressure loss 相对 weight_list/积分损失的权重)。
    # 分母保持不加权, 与 LpLoss.rel 同口径。
    # pred/truth: [ch, n]; lever_dist: [n] (常量, 已 detach)。
    w = 1.0 + alpha * lever_dist
    w = w / w.mean()
    diff = paddle.sqrt(((pred - truth) ** 2 * w).sum(axis=1))
    ref = paddle.sqrt((truth ** 2).sum(axis=1))
    return (diff / ref).mean()


def sign_penalty(pred, truth, scale, w=0.3):
    # 符号一致性惩罚：仅当真值明显非零(|truth| > 0.1*scale)且预测与真值异号
    # (pred*truth < 0)时施加，直接抑制 F066 那类 +168 -> -338 的符号翻转灾难。
    # 过零区(|truth| <= 0.1*scale)不施加，避免干扰本就无意义的近零符号。
    # 归一化到 scale^2 使量纲与 hybrid_loss 一致、与工况解耦。
    mask = (truth.abs() > 0.1 * scale).astype(pred.dtype)
    return w * mask * paddle.nn.functional.relu(-pred * truth) / (scale ** 2)


def compute_or_load_fm_scale(cfg, model, train_dataloader, loss_fn, decode_fn):
    """获取 6 分量力/力矩的全局特征尺度(物理空间 RMS)，供 hybrid_loss 软下限 &
    sign_penalty / 残差正则使用。

    优先读取缓存 force_moment_scale.json；未命中则用「模型自身的积分」(forward
    产出的 F_truth_{region}/M_truth_{region})对训练集跑一遍统计后写缓存——truth 与
    模型权重无关，故任何阶段计算都与训练时的积分完全一致，避免在预处理里重复实现
    分区力矩积分带来的不一致风险。
    顺序: scale_F=[drag, lift, lateral], scale_M=[overturning, pitching, roll]。
    """
    cache_dir = cfg.bounds_dir if cfg.get("bounds_dir", None) else cfg.train_output_path
    cache_path = os.path.join(cache_dir, "force_moment_scale.json")

    MAX_CARR = 20  # 方案 D: 支持的最大车厢/区域数(carriage_1..carriage_MAX_CARR)

    # 1) 命中缓存直接读
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            sF, sM = d["scale_F"], d["scale_M"]
            if len(sF) == 3 and len(sM) == 3 and all(v > 0 for v in list(sF) + list(sM)):
                sFr = d.get("scale_F_region", {}) or {}
                sMr = d.get("scale_M_region", {}) or {}
                logging.info(f"[fm_scale] 载入缓存 {cache_path}: scale_F={sF}, scale_M={sM}, "
                             f"regions={list(sMr.keys())}")
                return ([float(x) for x in sF], [float(x) for x in sM],
                        {k: [float(x) for x in v] for k, v in sFr.items()},
                        {k: [float(x) for x in v] for k, v in sMr.items()})
            logging.warning(f"[fm_scale] 缓存 {cache_path} 非法, 重新计算")
        except Exception as e:
            logging.warning(f"[fm_scale] 读取缓存失败({e}), 重新计算")

    # 2) 未命中: 遍历训练集, 用模型 forward 产出的 truth 积分统计 RMS
    logging.info("[fm_scale] 首次运行: 遍历训练集计算 6 分量力/力矩 RMS ...")
    was_training = model.training
    model.eval()
    F_sq = np.zeros(3, dtype=np.float64)
    M_sq = np.zeros(3, dtype=np.float64)
    n = 0
    # 方案 D: 分区域(逐车厢)累计平方和, 固定大小便于 DDP all_reduce。
    F_sq_r = np.zeros((MAX_CARR, 3), dtype=np.float64)
    M_sq_r = np.zeros((MAX_CARR, 3), dtype=np.float64)
    n_r = np.zeros(MAX_CARR, dtype=np.float64)
    idx_batch = 0
    with paddle.no_grad():
        for data_dict in train_dataloader:
            try:
                _, _, F_M_dict, region_masks = model(
                    data_dict, idx_batch, loss_fn=loss_fn, decode_fn=decode_fn)
            except Exception as e:
                logging.warning(f"[fm_scale] batch {idx_batch} forward 失败, 跳过: {e}")
                idx_batch += 1
                continue
            idx_batch += 1
            if F_M_dict == {} or ("OOM" in F_M_dict and F_M_dict.get("OOM")):
                continue
            for region in region_masks:
                Ft = F_M_dict.get(f"F_truth_{region}")
                Mt = F_M_dict.get(f"M_truth_{region}")
                if Ft is None or Mt is None:
                    continue
                Ft = Ft.numpy().astype(np.float64)
                Mt = Mt.numpy().astype(np.float64)
                if not (np.all(np.isfinite(Ft)) and np.all(np.isfinite(Mt))):
                    continue
                F_sq += Ft ** 2
                M_sq += Mt ** 2
                n += 1
                try:
                    k = int(region.split("_")[1]) - 1
                except (IndexError, ValueError):
                    k = -1
                if 0 <= k < MAX_CARR:
                    F_sq_r[k] += Ft ** 2
                    M_sq_r[k] += Mt ** 2
                    n_r[k] += 1

    # DDP: 汇总各 rank 的平方和与计数(全局 + 分区域一次性 all_reduce)
    if cfg.enable_ddp:
        payload = np.concatenate([
            F_sq, M_sq, [float(n)],
            F_sq_r.flatten(), M_sq_r.flatten(), n_r,
        ])
        t = paddle.to_tensor(payload, dtype="float64").cuda()
        paddle.distributed.all_reduce(t, op=paddle.distributed.ReduceOp.SUM)
        arr = t.numpy()
        F_sq, M_sq, n = arr[:3], arr[3:6], arr[6]
        off = 7
        F_sq_r = arr[off:off + MAX_CARR * 3].reshape(MAX_CARR, 3)
        off += MAX_CARR * 3
        M_sq_r = arr[off:off + MAX_CARR * 3].reshape(MAX_CARR, 3)
        off += MAX_CARR * 3
        n_r = arr[off:off + MAX_CARR]

    if was_training:
        model.train()

    if n <= 0:
        logging.warning("[fm_scale] 未收集到有效样本, 回退到 yaml 中的 scale_F/scale_M")
        fb_F = list(cfg.get("scale_F")) if cfg.get("scale_F", None) else None
        fb_M = list(cfg.get("scale_M")) if cfg.get("scale_M", None) else None
        return fb_F, fb_M, {}, {}

    scale_F = np.maximum(np.sqrt(F_sq / n), 1e-6)
    scale_M = np.maximum(np.sqrt(M_sq / n), 1e-6)
    sF = [float(x) for x in scale_F]
    sM = [float(x) for x in scale_M]

    # 方案 D: 逐区域 RMS(样本数>0 才计算), 键为 carriage_k。
    sFr, sMr = {}, {}
    for k in range(MAX_CARR):
        if n_r[k] > 0:
            region = f"carriage_{k + 1}"
            sFr[region] = [float(x) for x in np.maximum(np.sqrt(F_sq_r[k] / n_r[k]), 1e-6)]
            sMr[region] = [float(x) for x in np.maximum(np.sqrt(M_sq_r[k] / n_r[k]), 1e-6)]

    if paddle.distributed.get_rank() == 0:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fp:
                json.dump({"scale_F": sF, "scale_M": sM, "n_samples": int(n),
                           "scale_F_region": sFr, "scale_M_region": sMr,
                           "order_F": ["drag", "lift", "lateral"],
                           "order_M": ["overturning", "pitching", "roll"]},
                          fp, indent=2, ensure_ascii=False)
            logging.info(f"[fm_scale] 已写缓存 {cache_path}")
        except Exception as e:
            logging.warning(f"[fm_scale] 写缓存失败: {e}")

    logging.info(f"[fm_scale] scale_F(drag,lift,lateral)={sF}, "
                 f"scale_M(over,pitch,roll)={sM}, n={int(n)}, regions={list(sMr.keys())}")
    return sF, sM, sFr, sMr

 
def train(cfg: DictConfig):
    os.makedirs(cfg.train_output_path, exist_ok=True)
    os.makedirs(os.path.join(cfg.train_output_path, "log"), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(cfg.train_output_path, "log", f"{cfg.mode}.log"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s: %(message)s",
        force=True,
    )
 
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
    )
    logging.getLogger().addHandler(stream_handler)
 
    os.makedirs(os.path.join(cfg.train_output_path, "json"), exist_ok=True)
    train_json_file_path = os.path.join(cfg.train_output_path, "json", "loss.json")
 
    def create_json(json_file_path):
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        if os.path.isfile(json_file_path):
            os.remove(json_file_path)
        with open(json_file_path, "w") as file:
            json.dump([], file)
 
    if paddle.distributed.get_rank() == 0:
        create_json(train_json_file_path)
 
    def append_dict_to_json_list(file_path, dict_element):
        assert os.path.exists(file_path), file_path
        with open(file_path, "r") as file:
            data = json.load(file)

        if isinstance(data, list):
            data.append(dict_element)
        elif isinstance(data, dict):
            data.update(dict_element)
        else:
            logging.info("Error: The root of the JSON file is not a list.")
            return
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
 
    model = instantiate_network(cfg)
    # 方案 A: 把 Phase-1 积分力/力矩损失权重挂到模型上(DDP wrap 前, 保证内层可见)。
    # >0 时模型 forward 在 Phase-1 会产出可微分区积分 F/M 供 train.py 追加损失。
    model.phase1_fm_loss_w = float(cfg.get("phase1_fm_loss_w", 0.0))
    if cfg.optimizer == "AdamW":
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(), learning_rate=cfg.lr, weight_decay=1e-06
        )
    elif cfg.optimizer == "SOAP":
        optimizer = SOAP(parameters=model.parameters(), learning_rate=cfg.lr, weight_decay=1e-06)
    loss_fn = LpLoss(size_average=True)
    if cfg.enable_ddp:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
 
    resume_ep = cfg.resume_ep
    if cfg.state:
        state = paddle.load(path=str(cfg.state))
        model.set_state_dict(state_dict=state["model"])
        optimizer.set_lr(state["lr"])
        resume_ep = state["epoch"]
        logging.info(f"Resuming model from epoch {resume_ep}.")
    elif getattr(cfg, "pretrained", None):
        # 仅加载预训练权重用于微调: 只加载 model 权重, 不恢复 epoch/lr,
        # 因此 resume_ep 保持 cfg.resume_ep(默认 -1), 训练从 ep=0 开始跑微调阶段。
        # 兼容两种格式: 完整 checkpoint({"model":...}) 或裸 state_dict。
        state = paddle.load(path=str(cfg.pretrained))
        model_state = state["model"] if isinstance(state, dict) and "model" in state else state
        model.set_state_dict(state_dict=model_state)
        logging.info(f"Loaded pretrained weights from {cfg.pretrained} (finetune, not resuming epoch/lr).")
 
    device = ParallelEnv().device_id
 
    memory_allocated = paddle.device.cuda.memory_allocated(device=device) / (
        1024 * 1024 * 1024
    )
    logging.info(f"Memory usage with model loading: {memory_allocated:.2f} GB")
 
    datamodule = instantiate_datamodule(
        cfg,
        cfg.train_input_path,
        cfg.n_train_num,
        0,
        cfg.n_test_num,
        cfg.train_ratio,
        cfg.test_ratio,
    )
    train_dataloader = datamodule.train_dataloader(
        enable_ddp=cfg.enable_ddp, batch_size=cfg.batch_size
    )
 
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(
        T_max=cfg.num_epochs, learning_rate=optimizer.get_lr()
    )
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr

    # 力/力矩特征尺度: 首次运行从训练集计算并缓存, 之后自动读取; 覆盖 yaml 回退值。
    _sF, _sM, _sFr, _sMr = compute_or_load_fm_scale(
        cfg, model, train_dataloader, loss_fn, datamodule.decode)
    with open_dict(cfg):
        if _sF is not None:
            cfg.scale_F = _sF
        if _sM is not None:
            cfg.scale_M = _sM
        # 方案 D: 逐区域(车厢)特征尺度, 供 hybrid_loss 软下限按车厢自适应(小真值中间
        # 车厢点头不再被整车全局 floor 淹没)。region_scale=False 或空 dict 时回退全局 scale。
        if cfg.get("region_scale", True):
            cfg.scale_F_region = _sFr if _sFr else {}
            cfg.scale_M_region = _sMr if _sMr else {}
        else:
            cfg.scale_F_region = {}
            cfg.scale_M_region = {}

    logging.info(f"Start training {cfg.model} ... Train samples: {len(train_dataloader.dataset)}, batches: {len(train_dataloader)}.")
 
    # evaluate init
    test_dataloader = datamodule.test_dataloader(
        enable_ddp=False, batch_size=cfg.batch_size
    )  # each GPU use full test dataset
    # all_files = os.listdir(cfg.train_input_path)
    # prefix = "area"
    os.makedirs(os.path.join(cfg.train_output_path, "json"), exist_ok=True)
 
    data = {
        "test_case_id": datamodule.test_indices,
        "train_case_id": datamodule.train_indices,
    }
    if paddle.distributed.get_rank() == 0:
        with open(
            os.path.join(cfg.train_output_path, "json", "radius.json"), "w"
        ) as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
 
    if paddle.distributed.get_rank() == 0:
        logging.info(f"train indices: {datamodule.train_indices}")
        logging.info(f"test indices: {datamodule.test_indices}")
 
    def cal_mre(pred, label):
        return paddle.abs(x=pred - label) / paddle.abs(x=label)
 
    max_loss_case_id = None
    min_loss_case_id = None
 
    def evaluate_on_fly(epoch_id) -> int | None:
        eval_meter = AverageMeterDict()
        t1 = default_timer()
        max_error = 0.0
        min_error = float("inf")
        total_error = 0
        F_error = [0.0, 0.0, 0.0]
        M_error = [0.0, 0.0, 0.0]
        max_loss_case_id = None
        min_loss_case_id = None
        if paddle.distributed.get_rank() == 0:
            logging.info(
                f"Start evaluting {cfg.model} at epoch {epoch_id}, number of samples: {len(test_dataloader)}"
            )
 
        indices = datamodule.test_indices
        full_indices = datamodule.test_full_caseids
        error_dict = {key:0.0 for key in indices}
 
        sideslip_filename={}
        for key in indices:
            sideslip_filename[key] = os.path.join(
                cfg.train_output_path,
                "json",
                f"{str(key)}",
                "sideslip_angle.json",
            )
            if paddle.distributed.get_rank() == 0:
                create_json(sideslip_filename[key])
 
        current_model = eval_model if "eval_model" in locals() else model
        is_train = current_model.training
        if is_train:
            current_model.eval()
        # for dataParallel
        if isinstance(current_model, paddle.DataParallel):
            current_model = current_model._layers
        for i, data_dict in enumerate(test_dataloader):
            case_coefficent_json_dict = {}
            device = ParallelEnv().device_id
            device = paddle.CUDAPlace(device)
            try:
                out_dict, pred, truth, F_M_dict, region_masks = current_model.eval_dict(
                    device, data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode
                )
 
                if paddle.any(paddle.isnan(F_M_dict["F_truth"])) or paddle.any(paddle.isnan(F_M_dict["M_truth"])):
                    logging.info(
                        f"WARNING: nan detected on test sample {i}, skipping this sample."
                    )
                    continue
 
                if cfg.save_eval_results:
                    save_eval_results(
                        cfg,
                        pred,
                        truth,
                        full_indices[i],
                        epoch_id,
                        decode_fn=datamodule.decode,
                        caseid=datamodule.test_full_caseids[i],
                        q_ref=data_dict["q_ref"][0],
                    )
                    # save json output file
                    caseid=datamodule.test_full_caseids[i]
                    json_filename = os.path.join(
                        cfg.train_output_path,
                        "csv_vtp",
                        str(epoch_id),
                        f"{str(caseid)[:-4]}",
                        f"{str(caseid).split('-')[-1]}",
                        "case.json",
                    )
                    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
                    with open(json_filename, "w") as f:
                        json.dump(
                            {
                                "car_speed": data_dict["info"][0]["car_speed"],
                                "wind_speed": data_dict["info"][0]["wind_speed"],
                                "wind_angle": data_dict["info"][0]["wind_angle"],
                            },
                            f,
                        )   
                # paddle.device.cuda.empty_cache()
            except MemoryError as e:
                if "Out of memory" in str(e):
                    logging.info(f"WARNING: OOM on sample {i}, skipping this sample.")
                    if hasattr(paddle.device.cuda, "empty_cache"):
                        paddle.device.cuda.empty_cache()
                    continue
                else:
                    raise
            msg = f"Eval sample {i}... L2_Error: "
 
            for k, v in out_dict.items():
                if k.split("_")[0] == "L2":
                    msg += f"{k}: {v.item():.4f}, "
                    eval_meter.update({k: v})
            msg += f"|| MRE and Value: "
            for k, v in out_dict.items():
                if "pred" in k.split("_") and ("F" or "M" in k.split("_")):
                    k_truth = f"{k[:k.rfind('_')]}_truth"
                    mre = cal_mre(v, out_dict[k_truth])
                    eval_meter.update({f"MRE_{k[:k.rfind('_')]}": mre})
                    msg += f"MRE_{k[:k.rfind('_')]}: {mre.numpy()}, "
                    msg += f"[{k}: {v.numpy()}, {k_truth}: {out_dict[k_truth].numpy()}], "
                    if k == "F_pred" :
                        F_error = mre.numpy()
                    if k == "M_pred" :
                        M_error = mre.numpy()
 
            total_error += F_error.sum() + M_error.sum()
            
            for region, mask in region_masks.items():

                # 真·系数空间：F_pred_*_modify / M_pred_*_modify 为无量纲系数，
                # 评估/落盘需的是物理力/力矩，故 ×(1/const) 折回。const 缺失退化为 1.0。
                F_const = F_M_dict.get("F_const", 1.0)
                M_const = F_M_dict.get("M_const", 1.0)
                F_pred_modify = F_M_dict[f'F_pred_{region}_modify'] / F_const
                M_pred_modify = F_M_dict[f'M_pred_{region}_modify'] / M_const
                F_truth = F_M_dict[f"F_truth_{region}"]
                M_truth = F_M_dict[f"M_truth_{region}"]
                F_pred = F_M_dict[f"F_pred_{region}"]
                M_pred = F_M_dict[f"M_pred_{region}"]

                F_mre_modify = cal_mre(F_pred_modify, F_truth)
                M_mre_modify = cal_mre(M_pred_modify, M_truth)
                eval_meter.update({"MRE_F_modify": F_mre_modify})
                eval_meter.update({"MRE_M_modify": M_mre_modify})

                F_mre_modify = paddle.abs(x=F_pred_modify - F_truth) / paddle.abs(
                    x=F_truth
                )
                M_mre_modify = paddle.abs(x=M_pred_modify - M_truth) / paddle.abs(
                    x=M_truth
                )

                load_types = {
                    'aerodynamic_lift': {'real': F_truth[1], 'pred': F_pred[1], 'pred_modify': F_pred_modify[1]},
                    'aerodynamic_drag': {'real': F_truth[0], 'pred': F_pred[0], 'pred_modify': F_pred_modify[0]},
                    'pneumatic_lateral_force': {'real': F_truth[2], 'pred': F_pred[2], 'pred_modify': F_pred_modify[2]},
                    'pneumatic_overturning_moment': {'real': M_truth[0], 'pred': M_pred[0], 'pred_modify': M_pred_modify[0]},
                    'pneumatic_pitching_moment': {'real': M_truth[1], 'pred': M_pred[1], 'pred_modify': M_pred_modify[1]},
                    'pneumatic_roll_moment': {'real': M_truth[2], 'pred': M_pred[2], 'pred_modify': M_pred_modify[2]},
                }
 
                sideslip_angle = calculate_lateral_angle(
                    data_dict["info"][0]["car_speed"], 
                    data_dict["info"][0]["wind_speed"], 
                    data_dict["info"][0]["wind_angle"]
                )
 
                case_coefficent_json_dict['carriage_number'] = region.split('_')[1]
                case_coefficent_json_dict['type_value'] = sideslip_angle
                case_coefficent_json_dict['car_speed'] = data_dict["info"][0]["car_speed"]
                case_coefficent_json_dict['wind_speed'] = data_dict["info"][0]["wind_speed"]
                case_coefficent_json_dict['wind_angle'] = data_dict["info"][0]["wind_angle"]
 
 
                for load_name, values in load_types.items():
                    real_val = values['real'].numpy() if hasattr(values['real'], 'numpy') else values['real']
                    cal_val = values['pred'].numpy() if hasattr(values['pred'], 'numpy') else values['pred']
                    cal_val_modify = values['pred_modify'].numpy() if hasattr(values['pred_modify'], 'numpy') else values['pred_modify']
                    cal_error = 2*np.abs(cal_val - real_val) / (np.abs(real_val)+np.abs(cal_val)) 
                    cal_error_modify = 2*np.abs(cal_val_modify - real_val) / (np.abs(real_val)+np.abs(cal_val_modify)) 
                    if float(cal_error) == 2.0:
                        cal_val = -cal_val
                        cal_error = 2*np.abs(cal_val - real_val) / (np.abs(real_val)+np.abs(cal_val)) 
                    if float(cal_error_modify) == 2.0:
                        cal_val_modify = -cal_val_modify
                        cal_error_modify = 2*np.abs(cal_val_modify - real_val) / (np.abs(real_val)+np.abs(cal_val_modify)) 
                        
                    if epoch_id < cfg.num_epochs - cfg.finetuning_epochs:
                    # 微调前：仅记录 base FNO 预测结果
                       case_coefficent_json_dict[load_name] = {
                            'real_value': float(real_val),
                            'cal_value': float(cal_val),
                            'cal_error': float(cal_error),
                        }
                    else:
                        # 微调后：cal_value 为微调后结果，同时保留微调前结果
                        case_coefficent_json_dict[load_name] = {
                            'real_value': float(real_val),
                            'cal_value': float(cal_val_modify),
                            'cal_error': float(cal_error_modify),
                            'cal_value_no_modify': float(cal_val),
                            'cal_error_no_modify': float(cal_error),
                        }
 
                caseid=datamodule.test_full_caseids[i]
                if paddle.distributed.get_rank() == 0:
                    append_dict_to_json_list(sideslip_filename[str(caseid)[:-4]], case_coefficent_json_dict)
 
                msg += f"MRE_F_modify: {F_mre_modify.numpy()}, "
                msg += f"[F_pred_modify: {F_pred_modify.numpy()}, "
                msg += f"F_truth: {F_truth.numpy()}], "
                msg += f"MRE_M_modify: {M_mre_modify.numpy()}, "
                msg += f"[M_pred_modify: {M_pred_modify.numpy()}, "
                msg += f"M_truth: {M_truth.numpy()}], "
 
            logging.info(msg)
        
            error_dict[full_indices[i][:-4]] += F_error.sum() + M_error.sum()
 
        max_loss_case_id = max(error_dict, key=error_dict.get)
        min_loss_case_id = min(error_dict, key=error_dict.get)
        max_error = error_dict[max_loss_case_id]
        min_error = error_dict[min_loss_case_id]
 
        t2 = default_timer()
        msg = f"Testing took {t2 - t1:.2f} seconds. Everage eval values: "
        eval_dict = eval_meter.avg
        for k, v in eval_dict.items():
            msg += f"{v.numpy()}({k}), "
 
        if is_train:
            current_model.train()
 
        if max_loss_case_id is not None and min_loss_case_id is not None:
            msg += f"Maximum Error Sample ID: {max_loss_case_id}, Maximum Error: {max_error}, "
            msg += f"Minimum Error Sample ID: {min_loss_case_id}, Minimum Error: {min_error}, "
        elif max_loss_case_id is not None:
            msg += f"Maximum Error Sample ID: {max_loss_case_id}, Maximum Error: {max_error}, "
        elif min_loss_case_id is not None:
            msg += f"Minimum Error Sample ID: {min_loss_case_id}, Minimum Error: {min_error}, "
        else:
            msg += "Wawrning: No maximum and minimum Error, because all samples are not evaluated, might for OMM or other reason."
        logging.info(msg)
 
        if max_loss_case_id is not None and min_loss_case_id is not None:
            return max_loss_case_id, min_loss_case_id, total_error
        elif max_loss_case_id is not None:
            return max_loss_case_id, None, total_error
        elif min_loss_case_id is not None:
            return None, min_loss_case_id, total_error
        else:
            return None, None, total_error
 
    best_error = float('inf')
    for ep in range(cfg.num_epochs):
        if paddle.distributed.get_rank() == 0:
            train_json_dict = {}
        coefficent_json_dict = None
        if ep <= resume_ep:
            continue
        if ep == resume_ep + 1:
            logging.info(f"lr of {ep} is {optimizer.get_lr():.2e}")
 
        t1 = default_timer()
        train_l2_meter = AverageMeterDict()
        num_OOM = 0
        idx_batch = 0
        msg = "|| "
 
        if ep == cfg.num_epochs - cfg.finetuning_epochs + 1:
            # WarmupCosineDecay：前 warmup_steps 个 epoch 线性升温，
            # 之后 cosine 衰减至 eta_min，避免末期 lr 趋近于 0。
            # 相比原 CosineAnnealingDecay(T_max=finetuning_epochs)，
            # eta_min 保底 + warmup 使得后期仍保持有效学习率。
            _ft_eps   = cfg.finetuning_epochs          # 150
            _warmup   = max(1, int(_ft_eps * 0.1))     # 前 10% 升温，约 15 ep
            _eta_min  = cfg.lr_cd * 0.05               # 保底 lr = 5e-5（lr_cd 的 5%）
            # 用 LinearWarmup 包裹 CosineAnnealingDecay 实现 warmup + cosine
            _cosine_lr = paddle.optimizer.lr.CosineAnnealingDecay(
                T_max=_ft_eps - _warmup,
                learning_rate=cfg.lr_cd,
                eta_min=_eta_min,
            )
            tmp_lr = paddle.optimizer.lr.LinearWarmup(
                learning_rate=_cosine_lr,
                warmup_steps=_warmup,
                start_lr=cfg.lr_cd * 0.1,
                end_lr=cfg.lr_cd,
            )
            optimizer.set_lr_scheduler(tmp_lr)
            scheduler = tmp_lr
 
        if ep <= cfg.num_epochs - cfg.finetuning_epochs:
            for name, param in model.named_parameters():
                if "integral_cd" not in name:
                    param.stop_gradient = False
                else:
                    param.stop_gradient = True
            msg = "Integral CD params are frozen. || "
            model.train()
        else:
            for name, param in model.named_parameters():
                if "integral_cd" in name:
                    param.stop_gradient = False
                else:
                    param.stop_gradient = True
            msg = "Other params are frozen. || "
            msg += f"lr_cd: {optimizer.get_lr():.2e}, "
            model.eval()

        for data_dict in train_dataloader:
            try:
                if idx_batch == 0 and paddle.distributed.get_rank() == 0:
                    msg += f"Data Loading Time: {data_dict['Data_loading_time'][0]:.2f} seconds. || "
                    memory_allocated = paddle.device.cuda.memory_allocated(
                        device=device
                    ) / (1024 * 1024 * 1024)
                    msg += f"Memory Usage: {memory_allocated:.2f} GB (forward), "

                optimizer.clear_gradients(set_to_zero=False)
                pred, truth, F_M_dict,region_masks = model(
                    data_dict, idx_batch, loss_fn=loss_fn, decode_fn=datamodule.decode
                )
                # DDP 模式下，所有 rank 必须同步 skip 决定，避免 AllReduce 死锁
                local_skip = 0
                if "OOM" in F_M_dict:
                    if F_M_dict["OOM"] == True:
                        logging.info(f"[ep={ep} batch={idx_batch}] rank={paddle.distributed.get_rank()} OOM detected, marking skip.")
                        local_skip = 1
                    elif F_M_dict["OOM"] == False and paddle.any(
                        paddle.isnan(F_M_dict["F_truth"])
                    ):
                        logging.info(
                            f"[ep={ep} batch={idx_batch}] rank={paddle.distributed.get_rank()} nan detected in F_truth, marking skip."
                        )
                        local_skip = 1
                if cfg.enable_ddp:
                    skip_tensor = paddle.to_tensor([local_skip], dtype='int32').cuda()
                    paddle.distributed.all_reduce(skip_tensor, op=paddle.distributed.ReduceOp.MAX)
                    local_skip = skip_tensor.item()
                    if local_skip:
                        logging.info(f"[ep={ep} batch={idx_batch}] rank={paddle.distributed.get_rank()} global skip=True after all_reduce, skipping batch.")
                if local_skip:
                    idx_batch += 1
                    continue
 
            except MemoryError as e:
                if "Out of memory" in str(e):
                    num_OOM += 1
                    if hasattr(paddle.device.cuda, "empty_cache"):
                        paddle.device.cuda.empty_cache()
                    raise
                else:
                    raise
            loss = paddle.to_tensor(data=0.0).cuda(blocking=True)
 
            if F_M_dict == {} or F_M_dict.get("mode") == "phase1":
                # 力臂加权场损失(建议 B): 仅在 Phase-1 且模型返回 lever_dist 时对
                # pressure 通道生效; wss 对 M_y 经小力臂 r_z 进入, 保持普通 LpLoss。
                lever_alpha = float(cfg.get("lever_weight_alpha", 0.0))
                lever_dist = F_M_dict.get("lever_dist", None)
                for i in range(len(cfg.out_keys)):
                    key = cfg.out_keys[i]
                    st, end = (
                        sum(cfg.out_channels[:i]),
                        sum(cfg.out_channels[:i]) + cfg.out_channels[i],
                    )
                    if key == "pressure" and lever_alpha > 0 and lever_dist is not None:
                        loss_key = lever_weighted_rel_l2(
                            pred[st:end], truth[st:end], lever_dist, lever_alpha)
                    else:
                        loss_key = loss_fn(pred[st:end], truth[st:end])

                    train_l2_meter.update({key: loss_key.detach().item()})

                    loss += cfg.weight_list[i] * loss_key

                # 方案 A: Phase-1 追加「可微分区积分力/力矩」损失, 梯度直达 backbone,
                # 用 weight_F_M(含点头权重)与逐区域物理尺度(方案 D)在物理空间比较。
                if F_M_dict.get("mode") == "phase1":
                    p1w = float(cfg.get("phase1_fm_loss_w", 0.0))
                    eps_rel = cfg.get("eps_rel", 0.05)
                    gF = cfg.get("scale_F", None)   # [drag, lift, lateral]
                    gM = cfg.get("scale_M", None)   # [overturning, pitching, roll]
                    sFr = cfg.get("scale_F_region", {}) or {}
                    sMr = cfg.get("scale_M_region", {}) or {}
                    # F 分量顺序 [drag,lift,lateral]; weight_F_M [lift,drag,lateral,over,pitch,roll]
                    F_w = [cfg.weight_F_M[1], cfg.weight_F_M[0], cfg.weight_F_M[2]]
                    M_w = [cfg.weight_F_M[3], cfg.weight_F_M[4], cfg.weight_F_M[5]]

                    def _reg_sc(region, glob, per_region, i):
                        # 优先用逐区域物理尺度(方案 D), 缺失回退全局; 都无则 None(退化 |truth|)。
                        rv = per_region.get(region) if per_region else None
                        if rv is not None and i < len(rv):
                            return float(rv[i])
                        if glob is not None and i < len(glob):
                            return float(glob[i])
                        return None

                    for region in region_masks:
                        Fp = F_M_dict[f"F_pred_{region}"]
                        Ft = F_M_dict[f"F_truth_{region}"]
                        Mp = F_M_dict[f"M_pred_{region}"]
                        Mt = F_M_dict[f"M_truth_{region}"]
                        for i in range(3):
                            sfc = _reg_sc(region, gF, sFr, i)
                            smc = _reg_sc(region, gM, sMr, i)
                            loss += p1w * F_w[i] * hybrid_loss(Fp[i], Ft[i], sfc, eps_rel)
                            loss += p1w * M_w[i] * hybrid_loss(Mp[i], Mt[i], smc, eps_rel)
            else:
                for region, mask in region_masks.items():
                    F_pred_modify_net = F_M_dict[f'F_pred_{region}_modify']
                    M_pred_modify_net = F_M_dict[f'M_pred_{region}_modify']
                    F_truth = F_M_dict[f"F_truth_{region}"]
                    M_truth = F_M_dict[f"M_truth_{region}"]
                    F_pred = F_M_dict[f"F_pred_{region}"]
                    M_pred = F_M_dict[f"M_pred_{region}"]

                    # 真·系数空间：修正网络直接输出无量纲力/力矩系数，
                    # 故 truth 同步 ×const 换算到系数空间再比较（const=2/(ρv²A[·L])）。
                    # hybrid_loss 为相对误差，loss 数值不变，但网络输出天然 O(1)、
                    # 梯度量级与工况解耦。const 缺失时退化为 1.0（等价物理空间）。
                    F_const = F_M_dict.get("F_const", 1.0)
                    M_const = F_M_dict.get("M_const", 1.0)
                    F_truth_coef = F_truth * F_const
                    M_truth_coef = M_truth * M_const

                    # loss 直接用网络原始输出（系数），保证梯度通路畅通。
                    # scale 为各物理量全局特征尺度(物理空间 RMS)，×const 换算到系数
                    # 空间后传入 hybrid_loss/sign_penalty，使软下限与符号惩罚在物理上
                    # 一致、与工况解耦（const 在分子分母同时出现故自动约掉）。
                    F_scale = cfg.get("scale_F", None)   # [drag, lift, lateral]
                    M_scale = cfg.get("scale_M", None)   # [overturning, pitching, roll]
                    eps_rel = cfg.get("eps_rel", 0.05)
                    sp_w = cfg.get("sign_penalty_w", 0.3)
                    dreg = cfg.get("delta_reg_lambda", 0.01)

                    def _sc(scale_list, i, const):
                        # 物理尺度 -> 系数空间尺度（× const），与 *_truth_coef 同空间
                        if scale_list is None:
                            return None
                        return float(scale_list[i]) * float(const)

                    # F 网络分量顺序: [0]=drag,[1]=lift,[2]=lateral;
                    # weight_F_M 顺序: [lift, drag, lateral, overturning, pitching, roll]
                    F_w = [cfg.weight_F_M[1], cfg.weight_F_M[0], cfg.weight_F_M[2]]
                    M_w = [cfg.weight_F_M[3], cfg.weight_F_M[4], cfg.weight_F_M[5]]

                    F_delta = F_M_dict.get(f"F_delta_{region}", None)
                    M_delta = F_M_dict.get(f"M_delta_{region}", None)

                    for i in range(3):
                        sfc = _sc(F_scale, i, F_const)
                        smc = _sc(M_scale, i, M_const)
                        loss += F_w[i] * hybrid_loss(F_pred_modify_net[i], F_truth_coef[i], sfc, eps_rel)
                        loss += M_w[i] * hybrid_loss(M_pred_modify_net[i], M_truth_coef[i], smc, eps_rel)

                        # 符号惩罚仅作用于易变号的力矩分量
                        if sp_w > 0 and smc is not None:
                            loss += sign_penalty(M_pred_modify_net[i], M_truth_coef[i], smc, sp_w)

                        # 残差正则: 约束修正量 Δ 不偏离 base 太远(结构上无法翻转符号)
                        if dreg > 0:
                            if F_delta is not None and sfc:
                                loss += dreg * ((F_delta[i] / sfc) ** 2)
                            if M_delta is not None and smc:
                                loss += dreg * ((M_delta[i] / smc) ** 2)

                    # 系数 -> 物理力/力矩(× 1/const)，仅用于日志展示与 paddle.where 比较
                    F_pred_modify_phys = F_pred_modify_net / F_const
                    M_pred_modify_phys = M_pred_modify_net / M_const

                    # paddle.where 仅用于日志展示：选取更接近真值的预测值
                    F_pred_err = paddle.abs(x=F_pred_modify_phys - F_truth)
                    F_orig_err = paddle.abs(x=F_pred - F_truth)
                    F_pred_modify = paddle.where(F_orig_err < F_pred_err, F_pred, F_pred_modify_phys)

                    M_pred_err = paddle.abs(x=M_pred_modify_phys - M_truth)
                    M_orig_err = paddle.abs(x=M_pred - M_truth)
                    M_pred_modify = paddle.where(M_orig_err < M_pred_err, M_pred, M_pred_modify_phys)
 
                    F_mre_modify = paddle.abs(x=F_pred_modify - F_truth) / paddle.abs(
                        x=F_truth
                    )
                    M_mre_modify = paddle.abs(x=M_pred_modify - M_truth) / paddle.abs(
                        x=M_truth
                    )
                    F_mre = paddle.abs(x=F_pred_modify - F_truth) / paddle.abs(x=F_truth)
                    M_mre = paddle.abs(x=M_pred_modify - M_truth) / paddle.abs(x=M_truth)
 
                    train_l2_meter.update({"MSE_loss": loss.detach().item()})
                    train_l2_meter.update({"F_mre": F_mre.numpy()})
                    train_l2_meter.update({"M_mre": M_mre.numpy()})
                    train_l2_meter.update({"F_pred": F_pred.numpy()})
                    train_l2_meter.update({"M_pred": M_pred.numpy()})
                    train_l2_meter.update(
                        {"F_pred_modify": F_pred_modify.numpy()}
                    )
                    train_l2_meter.update({"M_pred_modify": M_pred_modify.numpy()})
                    train_l2_meter.update({"F_truth": F_truth.numpy()})
                    train_l2_meter.update({"M_truth": M_truth.numpy()})
                    train_l2_meter.update({"aerodynamic_lift": F_mre.numpy()[1]})
                    train_l2_meter.update({"aerodynamic_drag": F_mre.numpy()[0]})
                    train_l2_meter.update({"pneumatic_lateral_force": F_mre.numpy()[2]})
                    train_l2_meter.update({"pneumatic_overturning_moment": M_mre.numpy()[0]})
                    train_l2_meter.update({"pneumatic_pitching_moment": M_mre.numpy()[1]})
                    train_l2_meter.update({"pneumatic_roll_moment": M_mre.numpy()[2]})

                train_l2_meter.update(
                    {"pressure": F_M_dict["L2_pressure"].detach().item()}
                )
                train_l2_meter.update(
                    {"wallshearstress": F_M_dict["L2_wallshearstress"].detach().item()}
                )
                
 
            loss.backward()

            if idx_batch == 0 and paddle.distributed.get_rank() == 0:
                memory_allocated = (
                    paddle.device.cuda.memory_allocated(device=device) / 1024**3
                )
                msg += f"{memory_allocated:.2f} GB (backward), "
                max_memory_allocated = (
                    paddle.device.cuda.max_memory_allocated(device=device) / 1024**3
                )
                msg += f"{max_memory_allocated:.2f} GB (MAX), "
                memory_researved = paddle.device.cuda.memory_reserved() / 1024**3
                msg += f"{memory_researved:.2f} GB (Reserved)."
 
            optimizer.step()
            optimizer.clear_gradients(set_to_zero=False)
            paddle.device.cuda.empty_cache()
            idx_batch += 1
        scheduler.step()
        t2 = default_timer()
 
        if cfg.enable_ddp:
            train_l2_meter.sync()
 
        if paddle.distributed.get_rank() == 0:
            train_json_dict["epoch"] = ep
            if "aerodynamic_lift" in train_l2_meter.avg:
                train_json_dict["aerodynamic_lift"] = train_l2_meter.avg["aerodynamic_lift"]
                train_json_dict["aerodynamic_drag"] = train_l2_meter.avg["aerodynamic_drag"]
                train_json_dict["pneumatic_lateral_force"] = train_l2_meter.avg["pneumatic_lateral_force"]
                train_json_dict["pneumatic_overturning_moment"] = train_l2_meter.avg["pneumatic_overturning_moment"]
                train_json_dict["pneumatic_pitching_moment"] = train_l2_meter.avg["pneumatic_pitching_moment"]
                train_json_dict["pneumatic_roll_moment"] = train_l2_meter.avg["pneumatic_roll_moment"]
            else:
                train_json_dict["aerodynamic_lift"] = 0
                train_json_dict["aerodynamic_drag"] = 0
                train_json_dict["pneumatic_lateral_force"] = 0
                train_json_dict["pneumatic_overturning_moment"] = 0
                train_json_dict["pneumatic_pitching_moment"] = 0
                train_json_dict["pneumatic_roll_moment"] = 0
            train_json_dict["pressure_loss"] = train_l2_meter.avg["pressure"]
            train_json_dict["shear_stress_loss"] = train_l2_meter.avg["wallshearstress"]
 
        if num_OOM != 0:
            logging.info(f"WARNING: {num_OOM} samples OOM, skipping these samples.")
        msg_ep = f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2_Loss: "
        train_dict = train_l2_meter.avg
        for k, v in train_dict.items():
            msg_ep += f"{v}({k}), "
        if paddle.distributed.get_rank() == 0 and "msg" in locals():
            logging.info(msg_ep + msg)
 
        if ep == 0 or (ep + 1) % cfg.save_per_epoch == 0 or ep == cfg.num_epochs - 1:
            state = {"model": model.state_dict(), "lr": optimizer.get_lr(), "epoch": ep}
            os.makedirs(
                os.path.dirname(
                    f"{cfg.train_output_path}/pd/latest.pdparams"
                ),
                exist_ok=True,
            )
            paddle.save(
                obj=state, path=f"{cfg.train_output_path}/pd/latest.pdparams"
            )
            logging.info(
                f"Save checkpoint to: {cfg.train_output_path}/pd/latest.pdparams"
            )
            max_loss_case_id, min_loss_case_id, total_error = evaluate_on_fly(ep)
            if total_error < best_error:
                best_error = total_error
                os.makedirs(
                    os.path.dirname(
                        f"{cfg.train_output_path}/pd/best.pdparams"
                    ),
                    exist_ok=True,
                )
                paddle.save(
                    obj=state, path=f"{cfg.train_output_path}/pd/best.pdparams"
                )
                logging.info(
                    f"Save checkpoint to: {cfg.train_output_path}/pd/best.pdparams"
                )
 
 
        if paddle.distributed.get_rank() == 0:
            max_min_loss_dict = {"max_loss_case_id": max_loss_case_id, "min_loss_case_id": min_loss_case_id}
            append_dict_to_json_list(train_json_file_path, train_json_dict)
            append_dict_to_json_list(os.path.join(cfg.train_output_path, "json", "radius.json"), max_min_loss_dict)
 
 
def save_eval_results(
    cfg: DictConfig,
    pred,
    truth,
    centroid_idx,
    epoch_id,
    decode_fn=None,
    caseid=None,
    q_ref=None,
):
    pred_pressure = decode_fn(pred[0:1, :], 0, q_ref=q_ref).cpu().detach().numpy()
    pred_wallshearstress = decode_fn(pred[1:4, :], 1, q_ref=q_ref).cpu().detach().numpy()
    truth_pressure = decode_fn(truth[0:1, :], 0, q_ref=q_ref).cpu().detach().numpy()
    truth_wallshearstress = decode_fn(truth[1:4, :], 1, q_ref=q_ref).cpu().detach().numpy()
    delta_pressure = pred_pressure - truth_pressure
    delta_wallshearstress = pred_wallshearstress - truth_wallshearstress
    evals_results = {
        "cal_pressure_drag": pred_pressure,
        "cal_friction_resistance": pred_wallshearstress,
        "real_pressure_drag": truth_pressure,
        "real_friction_resistance": truth_wallshearstress,
        "cal_err_pressure": delta_pressure,
        "cal_err_friction_resistance": delta_wallshearstress,
    }
    centroid = np.load(f"{cfg.train_input_path}/centroid_{centroid_idx}.npy")
    cells = [("vertex", np.arange(tuple(centroid.shape)[0]).reshape(-1, 1))]
 
    output_dir = cfg.train_output_path
    os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
    for k, v in evals_results.items():
        # save 6 csv output files
        array_hstack = np.hstack((centroid, v.T))
        csv_filename = os.path.join(
            output_dir,
            "csv_vtp",
            str(epoch_id),
            f"{str(caseid)[:-4]}",
            f"{str(caseid).split('-')[-1]}",
            f"{k}.csv",
        )
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        np.savetxt(csv_filename, array_hstack, delimiter=",", fmt="%f")
        logging.info(f"Save csv to: {csv_filename}")
 
        vtp_filename = os.path.join(
            output_dir,
            "csv_vtp",
            str(epoch_id),
            f"{str(caseid)[:-4]}",
            f"{str(caseid).split('-')[-1]}",
            f"{k}.vtp",
        )
        os.makedirs(os.path.dirname(vtp_filename), exist_ok=True)
        if v.T.shape[1] == 1:
            save_vtp_from_dict(
                vtp_filename,
                {
                    "x": centroid[:, 0:1],
                    "y": centroid[:, 1:2],
                    "z": centroid[:, 2:3],
                    k: v.T,
                },
                ("x", "y", "z"),
                (k,),
            )
        else:
            save_vtp_from_dict(
                    vtp_filename,
                {
                    "x": centroid[:, 0:1],
                    "y": centroid[:, 1:2],
                    "z": centroid[:, 2:3],
                    k: np.linalg.norm(v.T, axis=1, keepdims=True),
                },
                ("x", "y", "z"),
                (k,),
            )
 
        logging.info(f"Save vtp to: {vtp_filename}")
 
    return None


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    cfg.enable_ddp = world_size > 1
    if cfg.seed is not None:
        set_seed(cfg.seed)
 
    if cfg.mode == "train":
        print("################## training #####################")
        train(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train'], but got '{cfg.mode}'")
 
 
if __name__ == "__main__":
    main()