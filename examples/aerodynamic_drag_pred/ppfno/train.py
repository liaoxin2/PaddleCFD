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
            point_cloud.save(f"{filename}_t-{t:0{width}}.vtp")
        else:
            point_cloud.save(f"{filename}.vtp")

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
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(), learning_rate=cfg.lr, weight_decay=1e-06
    )
    # optimizer = SOAP(parameters=model.parameters(), learning_rate=cfg.lr, weight_decay=1e-06)
    loss_fn = LpLoss(size_average=True)
    if cfg.enable_ddp:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

    resume_ep = cfg.resume_ep
    if cfg.state:
        state = paddle.load(path=str(cfg.state))
        model.set_state_dict(state_dict=state["model"])
        optimizer.set_lr(state["lr"])
        #resume_ep = state["epoch"]
        logging.info(f"Resuming model from epoch {resume_ep}.")

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

    logging.info(f"Start training {cfg.model} ...")

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

    eval_meter = AverageMeterDict()

    if paddle.distributed.get_rank() == 0:
        logging.info(f"train indices: {datamodule.train_indices}")
        logging.info(f"test indices: {datamodule.test_indices}")

    def cal_mre(pred, label):
        return paddle.abs(x=pred - label) / paddle.abs(x=label)

    max_loss_case_id = None
    min_loss_case_id = None

    def evaluate_on_fly(epoch_id) -> int | None:
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
                out_dict, pred, truth, F_M_dict = current_model.eval_dict(
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
                    )
                    # save json output file
                    caseid=datamodule.test_full_caseids[i]
                    json_filename = os.path.join(
                        cfg.train_output_path,
                        "csv_vtp",
                        str(epoch_id),
                        f"{str(caseid)[:20]}",
                        f"{str(caseid)[21:]}",
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

            F_mre_modify = cal_mre(F_M_dict["F_pred_modify"], out_dict["F_truth"])
            M_mre_modify = cal_mre(F_M_dict["M_pred_modify"], out_dict["M_truth"])
            eval_meter.update({"MRE_F_modify": F_mre_modify})
            eval_meter.update({"MRE_M_modify": M_mre_modify})

            # if F_error.sum() + M_error.sum() > max_error:
            #     max_error = F_error.sum() + M_error.sum()
            #     max_loss_case_id = i
            # if F_error.sum() + M_error.sum() < min_error:
            #     min_error = F_error.sum() + M_error.sum()
            #     min_loss_case_id = i
            total_error += F_error.sum() + M_error.sum()

            F_pred_modify = F_M_dict["F_pred_modify"]
            M_pred_modify = F_M_dict["M_pred_modify"]
            F_truth = out_dict["F_truth"]
            M_truth = out_dict["M_truth"]
            F_pred = out_dict["F_pred"]
            M_pred = out_dict["M_pred"]
            F_mre_modify = paddle.abs(x=F_pred_modify - F_truth) / paddle.abs(
                x=F_truth
            )
            M_mre_modify = paddle.abs(x=M_pred_modify - M_truth) / paddle.abs(
                x=M_truth
            )

            load_types = {
                'aerodynamic_lift': {'real': F_M_dict['F_truth'][1], 'pred': F_M_dict['F_pred'][1], 'pred_modify': F_M_dict['F_pred_modify'][1]},
                'aerodynamic_drag': {'real': F_M_dict['F_truth'][0], 'pred': F_M_dict['F_pred'][0], 'pred_modify': F_M_dict['F_pred_modify'][0]},
                'pneumatic_lateral_force': {'real': F_M_dict['F_truth'][2], 'pred': F_M_dict['F_pred'][2], 'pred_modify': F_M_dict['F_pred_modify'][2]},
                'pneumatic_overturning_moment': {'real': F_M_dict['M_truth'][0], 'pred': F_M_dict['M_pred'][0], 'pred_modify': F_M_dict['M_pred_modify'][0]},
                'pneumatic_pitching_moment': {'real': F_M_dict['M_truth'][1], 'pred': F_M_dict['M_pred'][1], 'pred_modify': F_M_dict['M_pred_modify'][1]},
                'pneumatic_roll_moment': {'real': F_M_dict['M_truth'][2], 'pred': F_M_dict['M_pred'][2], 'pred_modify': F_M_dict['M_pred_modify'][2]},
            }

            sideslip_angle = calculate_lateral_angle(
                data_dict["info"][0]["car_speed"], 
                data_dict["info"][0]["wind_speed"], 
                data_dict["info"][0]["wind_angle"]
            )

            case_coefficent_json_dict['type_value'] = sideslip_angle
            case_coefficent_json_dict['car_speed'] = data_dict["info"][0]["car_speed"]
            case_coefficent_json_dict['wind_speed'] = data_dict["info"][0]["wind_speed"]
            case_coefficent_json_dict['wind_angle'] = data_dict["info"][0]["wind_angle"]


            for load_name, values in load_types.items():
                real_val = values['real'].numpy() if hasattr(values['real'], 'numpy') else values['real']
                cal_val = values['pred'].numpy() if hasattr(values['pred'], 'numpy') else values['pred']
                cal_val_modify = values['pred_modify'].numpy() if hasattr(values['pred_modify'], 'numpy') else values['pred_modify']
                cal_error = cal_val - real_val
                cal_error_modify = cal_val_modify - real_val
                case_coefficent_json_dict[load_name] = {
                    'real_value': float(real_val),
                    'cal_value': float(cal_val),
                    #'cal_value_modify': float(cal_val_modify),
                    'cal_error': float(cal_error),
                    #'cal_error_modify': float(cal_error_modify),
                }

            caseid=datamodule.test_full_caseids[i]
            append_dict_to_json_list(sideslip_filename[str(caseid)[:20]], case_coefficent_json_dict)

            msg += f"MRE_F_modify: {F_mre_modify.numpy()}, "
            msg += f"[F_pred_modify: {F_pred_modify.numpy()}, "
            msg += f"F_truth: {F_truth.numpy()}], "
            msg += f"MRE_M_modify: {M_mre_modify.numpy()}, "
            msg += f"[M_pred_modify: {M_pred_modify.numpy()}, "
            msg += f"M_truth: {M_truth.numpy()}], "

            logging.info(msg)
        
            error_dict[full_indices[i][:-6]] += F_error.sum() + M_error.sum()

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
            tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(
                T_max=cfg.finetuning_epochs, learning_rate=cfg.lr_cd
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
                pred, truth, F_M_dict = model(
                    data_dict, idx_batch, loss_fn=loss_fn, decode_fn=datamodule.decode
                )
                if "OOM" in F_M_dict:
                    if F_M_dict["OOM"] == True:
                        idx_batch += 1
                        continue
                    elif F_M_dict["OOM"] == False and paddle.any(
                        paddle.isnan(F_M_dict["F_truth"])
                    ):
                        logging.info(
                            f"WARNING: nan detected on sample {idx_batch}, skipping this sample."
                        )
                        idx_batch += 1

                        continue

            except MemoryError as e:
                raise
                if "Out of memory" in str(e):
                    num_OOM += 1
                    if hasattr(paddle.device.cuda, "empty_cache"):
                        paddle.device.cuda.empty_cache()
                    continue
                else:
                    raise
            loss = paddle.to_tensor(data=0.0).cuda(blocking=True)

            if F_M_dict == {}:
                for i in range(len(cfg.out_keys)):
                    key = cfg.out_keys[i]
                    st, end = (
                        sum(cfg.out_channels[:i]),
                        sum(cfg.out_channels[:i]) + cfg.out_channels[i],
                    )
                    loss_key = loss_fn(pred[st:end], truth[st:end])

                    train_l2_meter.update({key: loss_key.detach().item()})

                    loss += cfg.weight_list[i] * loss_key
            else:
                F_pred_modify = F_M_dict["F_pred_modify"]
                M_pred_modify = F_M_dict["M_pred_modify"]
                F_truth = F_M_dict["F_truth"]
                M_truth = F_M_dict["M_truth"]
                F_pred = F_M_dict["F_pred"]
                M_pred = F_M_dict["M_pred"]
                F_mre_modify = paddle.abs(x=F_pred_modify - F_truth) / paddle.abs(
                    x=F_truth
                )
                M_mre_modify = paddle.abs(x=M_pred_modify - M_truth) / paddle.abs(
                    x=M_truth
                )
                F_mre = paddle.abs(x=F_pred_modify - F_truth) / paddle.abs(x=F_truth)
                M_mre = paddle.abs(x=M_pred_modify - M_truth) / paddle.abs(x=M_truth)

                
                loss += 1.2*paddle.nn.functional.mse_loss(F_pred_modify[1], F_truth[1])
                loss += 1.2*paddle.nn.functional.mse_loss(M_pred_modify[1], M_truth[1])
                loss += 50*paddle.nn.functional.mse_loss(F_pred_modify[0], F_truth[0])
                loss += 3.1*paddle.nn.functional.mse_loss(F_pred_modify[2], F_truth[2])
                loss += 0.7*paddle.nn.functional.mse_loss(M_pred_modify[0], M_truth[0])
                loss += 1.3*paddle.nn.functional.mse_loss(M_pred_modify[2], M_truth[2])


                train_l2_meter.update(
                    {"pressure": F_M_dict["L2_pressure"].detach().item()}
                )
                train_l2_meter.update(
                    {"wallshearstress": F_M_dict["L2_wallshearstress"].detach().item()}
                )

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
                

            loss.backward(grad_tensor=loss)

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
            # train_json_dict["pressure_loss"] = train_l2_meter.avg["pressure"]
            # train_json_dict["shear_stress_loss"] = train_l2_meter.avg["wallshearstress"]

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
):
    pred_pressure = decode_fn(pred[0:1, :], 0).cpu().detach().numpy()
    pred_wallshearstress = decode_fn(pred[1:4, :], 1).cpu().detach().numpy()
    truth_pressure = decode_fn(truth[0:1, :], 0).cpu().detach().numpy()
    truth_wallshearstress = decode_fn(truth[1:4, :], 1).cpu().detach().numpy()
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
            f"{str(caseid)[:20]}",
            f"{str(caseid)[21:]}",
            f"{k}.csv",
        )
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        np.savetxt(csv_filename, array_hstack, delimiter=",", fmt="%f")
        logging.info(f"Save csv to: {csv_filename}")

        vtp_filename = os.path.join(
            output_dir,
            "csv_vtp",
            str(epoch_id),
            f"{str(caseid)[:20]}",
            f"{str(caseid)[21:]}",
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
