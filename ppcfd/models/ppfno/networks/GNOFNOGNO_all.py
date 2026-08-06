import copy
import sys

import paddle
import paddle.nn as nn
import math
import json

from ..neuralop.models import FNO
from .base_model import BaseModel
from .integral_finetuning import Integral_Cd
from .neighbor_ops import NeighborMLPConvLayer
from .neighbor_ops import NeighborMLPConvLayerLinear
from .neighbor_ops import NeighborMLPConvLayerWeighted
from .neighbor_ops import NeighborSearchLayer
from .net_utils import MLP
from .net_utils import AdaIN
from .net_utils import PositionalEmbedding
from .net_utils import Projection
from .utilities3 import count_params
from .utilities3 import memory_usage
from .utilities3 import num_of_nans
from .utilities3 import paddle_memory_usage
from .utilities3 import show_tensor_range


class GNOFNOGNO(BaseModel):
    def __init__(
        self,
        radius_in=0.05,
        radius_out=0.05,
        embed_dim=64,
        hidden_channels=(32, 32),
        in_channels=1,
        out_channels=1,
        fno_modes=(16, 16, 16),
        fno_hidden_channels=32,
        fno_out_channels=32,
        fno_domain_padding=0.125,
        fno_norm="group_norm",
        fno_factorization="tucker",
        fno_rank=0.4,
        linear_kernel=True,
        weighted_kernel=True,
    ):
        super().__init__()
        self.weighted_kernel = weighted_kernel
        self.nb_search_in = NeighborSearchLayer(radius_in)
        self.nb_search_out = NeighborSearchLayer(radius_out)
        self.pos_embed = PositionalEmbedding(embed_dim)
        self.df_embed = MLP([in_channels, embed_dim, 3 * embed_dim], paddle.nn.GELU)
        self.linear_kernel = linear_kernel
        kernel1 = MLP([10 * embed_dim, 512, 256, hidden_channels[0]], paddle.nn.GELU)
        self.gno1 = NeighborMLPConvLayerWeighted(mlp=kernel1)
        if linear_kernel == False:
            kernel2 = MLP(
                [fno_out_channels + 4 * embed_dim, 512, 256, hidden_channels[1]],
                paddle.nn.GELU,
            )
            self.gno2 = NeighborMLPConvLayer(mlp=kernel2)
        else:
            kernel2 = MLP([7 * embed_dim, 512, 256, hidden_channels[1]], paddle.nn.GELU)
            self.gno2 = NeighborMLPConvLayerLinear(mlp=kernel2)
        self.fno = FNO(
            fno_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=hidden_channels[0] + 3 + in_channels,
            out_channels=fno_out_channels,
            use_mlp=True,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=fno_domain_padding,
            factorization=fno_factorization,
            norm=fno_norm,
            rank=fno_rank,
        )
        self.projection = Projection(
            in_channels=hidden_channels[1],
            out_channels=out_channels,
            hidden_channels=256,
            non_linearity=paddle.nn.functional.gelu,
            n_dim=1,
        )
        self.print_model_size()

    def forward(self, x_in, x_out, df, x_eval=None, area_in=None, area_eval=None):
        in_to_out_nb = self.nb_search_in(x_in, x_out.reshape((-1, 3)))
        if x_eval is not None:
            out_to_in_nb = self.nb_search_out(x_out.reshape((-1, 3)), x_eval)
        else:
            out_to_in_nb = self.nb_search_out(x_out.reshape((-1, 3)), x_in)
        resolution = tuple(df.shape)[-1]
        n_in = tuple(x_in.shape)[0]
        if area_in is None or self.weighted_kernel is False:
            area_in = paddle.ones(shape=(n_in,))
        x_in = paddle.concat(x=[x_in, area_in.unsqueeze(axis=-1)], axis=-1)
        x_in_embed = self.pos_embed(x_in.reshape((-1,))).reshape((n_in, -1))
        if x_eval is not None:
            n_eval = tuple(x_eval.shape)[0]
            if area_eval is None or self.weighted_kernel is False:
                area_eval = paddle.ones(shape=(n_eval,))
            x_eval = paddle.concat(x=[x_eval, area_eval.unsqueeze(axis=-1)], axis=-1)
            x_eval_embed = self.pos_embed(x_eval.reshape((-1,))).reshape((n_eval, -1))
        x_out_embed = self.pos_embed(x_out.reshape((-1,))).reshape(
            (resolution**3, -1)
        )
        df_embed = self.df_embed(df.transpose(perm=[1, 2, 3, 0])).reshape(
            (resolution**3, -1)
        )
        grid_embed = paddle.concat(x=[x_out_embed, df_embed], axis=-1)
        u = self.gno1(x_in_embed, in_to_out_nb, grid_embed, area_in)
        u = (
            u.reshape((resolution, resolution, resolution, -1))
            .transpose(perm=[3, 0, 1, 2])
            .unsqueeze(axis=0)
        )
        u = paddle.concat(
            x=(
                x_out.transpose(perm=[3, 0, 1, 2]).unsqueeze(axis=0),
                df.unsqueeze(axis=0),
                u,
            ),
            axis=1,
        )
        u = self.fno(u)
        u = u.squeeze().transpose(perm=[1, 2, 3, 0]).reshape((resolution**3, -1))
        if self.linear_kernel == False:
            if x_eval is not None:
                u = self.gno2(u, out_to_in_nb, x_eval_embed)
            else:
                u = self.gno2(u, out_to_in_nb, x_in_embed)
        elif x_eval is not None:
            u = self.gno2(
                x_in=x_out_embed,
                neighbors=out_to_in_nb,
                in_features=u,
                x_out=x_eval_embed,
            )
        else:
            u = self.gno2(
                x_in=x_out_embed,
                neighbors=out_to_in_nb,
                in_features=u,
                x_out=x_in_embed,
            )
        u = u.unsqueeze(axis=0).transpose(perm=[0, 2, 1])
        u = self.projection(u).squeeze(axis=0).transpose(perm=[1, 0])
        return u

    def print_model_size(self):
        print("--------------------------------")
        print("The MLP_1 size is ", count_params(self.df_embed))
        print("The gno1 size is ", count_params(self.gno1))
        print("The gno2 size is ", count_params(self.gno2))
        print("The fno size is ", count_params(self.fno))
        print("The projection size is ", count_params(self.projection))
        print("The nb_search_in size is ", count_params(self.nb_search_in))
        print("The nb_search_out size is ", count_params(self.nb_search_out))
        print("The pos_embed size is ", count_params(self.pos_embed))
        return None


class GNOFNOGNO_all(GNOFNOGNO):
    def __init__(
        self,
        radius_in=0.05,
        radius_out=0.05,
        embed_dim=16,
        hidden_channels=(16, 16),
        in_channels=2,
        out_channels=[1, 3],
        fno_modes=(16, 16, 16),
        fno_hidden_channels=16,
        fno_out_channels=16,
        fno_domain_padding=0.125,
        fno_norm="ada_in",
        adain_embed_dim=64,
        fno_factorization="tucker",
        fno_rank=0.4,
        linear_kernel=True,
        weighted_kernel=True,
        max_in_points=5000,
        subsample_train=1,
        subsample_eval=1,
        reference_point=[-3.20625,0,0],
        layers=[2,64,128,64,1],
        out_keys=["pressure"],
    ):
        if fno_norm == "ada_in":
            init_norm = "group_norm"
        else:
            init_norm = fno_norm
        self.max_in_points = max_in_points
        self.subsample_train = subsample_train
        self.subsample_eval = subsample_eval
        self.out_keys = out_keys
        self.out_channels = out_channels
        self.reference_point = reference_point
        self.layers = layers
        super().__init__(
            radius_in=radius_in,
            radius_out=radius_out,
            embed_dim=embed_dim,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=sum(out_channels),
            fno_modes=fno_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_out_channels=fno_out_channels,
            fno_domain_padding=fno_domain_padding,
            fno_norm=init_norm,
            fno_factorization=fno_factorization,
            fno_rank=fno_rank,
            linear_kernel=linear_kernel,
            weighted_kernel=weighted_kernel,
        )
        self.integral_cd = Integral_Cd(layers=self.layers)
        if fno_norm == "ada_in":
            self.adain_pos_embed = PositionalEmbedding(adain_embed_dim)
            self.fno.fno_blocks.norm = paddle.nn.LayerList(
                sublayers=(
                    AdaIN(adain_embed_dim, fno_hidden_channels)
                    for _ in range(
                        self.fno.fno_blocks.n_norms * self.fno.fno_blocks.convs.n_layers
                    )
                )
            )
            self.use_adain = True
        else:
            self.use_adain = False
        print("The fno + adain size is ", count_params(self.fno))
        print("--------------------------------")

    def data_dict_to_input(self, data_dict, data_device=None):
        x_in = data_dict["centroids"][0]
        x_out = (
            data_dict["df_query_points"].squeeze(axis=0).transpose(perm=[1, 2, 3, 0])
        )
        df = data_dict["df"]
        area = data_dict["areas"][0]

        wind_angle = float(data_dict["info"][0]["wind_angle"])
        car_speed = float(data_dict["info"][0]["car_speed"])  # already m/s
        wind_speed = float(data_dict["info"][0]["wind_speed"])

        angle_field = wind_angle * paddle.ones_like(x=df).astype("float32")
        car_field = car_speed * paddle.ones_like(x=df).astype("float32")
        wind_field = wind_speed * paddle.ones_like(x=df).astype("float32")
    
        df = paddle.concat(x=(df, angle_field, car_field, wind_field), axis=0)
        if self.use_adain:
            vel = paddle.to_tensor(data=[[wind_angle, car_speed, wind_speed]], dtype="float32")
            vel_embed = self.adain_pos_embed(vel)
            vel_embed = vel_embed.squeeze(0)  
            for norm in self.fno.fno_blocks.norm:
                norm.update_embeddding(vel_embed)
        return x_in, x_out, df, area

    def build_region_masks(self, points, boundaries):
        x = points[:, 0]
        boundaries = paddle.to_tensor(boundaries)

        x_min = x.min().item()
        x_max = x.max().item()
        b_min = boundaries.min().item()
        b_max = boundaries.max().item()

        if not (x_min <= b_min and b_max <= x_max):
            boundaries = boundaries * 0.001
            b_min = boundaries.min().item()
            b_max = boundaries.max().item()
            if not (x_min <= b_min and b_max <= x_max):
                raise ValueError(
                    f"boundaries [{b_min:.6f}, {b_max:.6f}] (after *0.001) still out of "
                    f"x range [{x_min:.6f}, {x_max:.6f}]"
                )

        masks = {}

        masks["carriage_1"] = x < boundaries[0]

        for i in range(len(boundaries) - 1):
            masks[f"carriage_{i+2}"] = (x >= boundaries[i]) & (x < boundaries[i + 1])

        masks[f"carriage_{len(boundaries)+1}"] = x >= boundaries[-1]

        return masks

    def cal_F_M(self, data_dict, pred_decode, truth_decode, region_masks, key="pressure"):
        r0 = data_dict["info"][0]["reference_point"]
        r0 = json.loads(r0)

        triangle_normals = data_dict["triangle_normals"][0]        # (N,3)
        areas = data_dict["areas"][0].reshape([-1, 1])             # (N,1)
        centroids = data_dict["centroids_no_norms"][0]             # (N,3)

        # ======================
        # 单元力计算（不区分区域）
        # ======================
        if key == "pressure":
            traction_truth = -truth_decode.reshape([-1, 1]) * triangle_normals
            traction_pred  = -pred_decode.reshape([-1, 1])  * triangle_normals
        elif key == "wallshearstress":
            traction_truth = -truth_decode.T
            traction_pred  = -pred_decode.T
        else:
            raise ValueError(f"Unknown key: {key}")

        F_per_truth = traction_truth * areas      # (N,3)
        F_per_pred  = traction_pred  * areas

        # ======================
        # 分区域积分
        # ======================
        results = {}

        for region, mask in region_masks.items():
            r_rel = centroids - 0.001*paddle.to_tensor(r0[int(region.split('_')[1])-1], dtype="float32")
            
            F_r_truth = F_per_truth[mask]
            F_r_pred  = F_per_pred[mask]

            r_r = r_rel[mask]

            F_total_truth = F_r_truth.sum(axis=0)
            F_total_pred  = F_r_pred.sum(axis=0)

            M_total_truth = paddle.cross(r_r, F_r_truth).sum(axis=0)
            M_total_pred  = paddle.cross(r_r, F_r_pred).sum(axis=0)

            results[region] = {
                "F_truth": F_total_truth,
                "F_pred":  F_total_pred,
                "M_truth": M_total_truth,
                "M_pred":  M_total_pred,
            }

        return results

    def phase1_fm_terms(self, data_dict, pred, indices, decode_fn):
        """方案 A：Phase-1 阶段的「可微分区积分力/力矩」估计器。

        与 eval_dict/cal_F_M(Phase-2, @no_grad, 全网格积分) 不同，本方法在
        Phase-1 只有子采样点(x_in[::subsample_train][indices])可用，且 pred 仍带梯度、
        直连 backbone。因此在子网格上做分区积分，再用「区域面积比」把部分积分放大为
        整车物理积分的无偏估计，使 F_pred/M_pred 对 backbone 可导，从而让点头力矩等
        积分量的误差梯度直接重塑场预测(而非只靠 Phase-2 的冻结-backbone 修正头)。

        参数:
          pred    : [n_sub, total_ch] 网络原始(归一化/系数)输出, 带梯度。
          indices : 选中的子采样点在 strided 网格中的下标(与 forward 内一致)。
          decode_fn: datamodule.decode, 期望 [ch, n] 输入并返回 [ch, n]。
        返回:
          (terms, sub_masks)。terms[region] = {F_pred,F_truth,M_pred,M_truth}(物理空间,
          F_pred/M_pred 可导, *_truth 为常量)。
        """
        r0 = json.loads(data_dict["info"][0]["reference_point"])
        q_ref = data_dict["q_ref"][0]

        normals_full = data_dict["triangle_normals"][0]            # (Nf,3)
        areas_full = data_dict["areas"][0].reshape([-1, 1])        # (Nf,1)
        cent_full = data_dict["centroids_no_norms"][0]             # (Nf,3)
        n_full = cent_full.shape[0]

        # 子采样点在整网格中的绝对行号(与 forward 的 x_in[::subsample_train][indices] 对齐)
        strided = paddle.arange(0, n_full, self.subsample_train)
        abs_idx = strided[indices]
        normals_s = normals_full[abs_idx]                          # (n_sub,3)
        areas_s = areas_full[abs_idx]                              # (n_sub,1)
        cent_s = cent_full[abs_idx]                                # (n_sub,3)

        # 区域划分：在整网格上算 mask 再按 abs_idx 取子集，保证与 Phase-2 边界处理一致
        coordinate = data_dict["info"][0]["carriage_offset"]
        if coordinate is not None and "," in coordinate:
            value_list = [float(v) for v in coordinate.split(",")]
            full_masks = self.build_region_masks(cent_full, value_list)
        elif coordinate is not None:
            value_list = [float(coordinate)]
            full_masks = self.build_region_masks(cent_full, value_list)
        else:
            full_masks = {
                "carriage_1": paddle.ones([n_full], dtype=paddle.bool)
            }
        sub_masks = {r: m[abs_idx] for r, m in full_masks.items()}

        pred_t = pred.transpose(perm=[1, 0])                       # [total_ch, n_sub]

        # 预解码各通道的物理单元力(traction * area)，pred 带梯度、truth 为常量
        Fper_pred_keys = []
        Fper_truth_keys = []
        for i in range(len(self.out_keys)):
            key = self.out_keys[i]
            st = sum(self.out_channels[:i])
            end = st + self.out_channels[i]
            pred_key = pred_t[st:end, :]                           # [ch, n_sub]
            truth_key = data_dict[key][0][abs_idx]
            if len(tuple(truth_key.shape)) == 1:
                truth_key = truth_key.reshape([-1, 1])
            truth_key = truth_key[:, : self.out_channels[i]].transpose(perm=[1, 0])
            pred_dec = decode_fn(pred_key, i, q_ref=q_ref)         # [ch, n_sub]
            truth_dec = decode_fn(truth_key, i, q_ref=q_ref)
            if key == "pressure":
                tr_pred = -pred_dec.reshape([-1, 1]) * normals_s   # (n_sub,3)
                tr_truth = -truth_dec.reshape([-1, 1]) * normals_s
            elif key == "wallshearstress":
                tr_pred = -pred_dec.T
                tr_truth = -truth_dec.T
            else:
                raise ValueError(f"Unknown key: {key}")
            Fper_pred_keys.append(tr_pred * areas_s)               # (n_sub,3)
            Fper_truth_keys.append(tr_truth * areas_s)

        Fper_pred = sum(Fper_pred_keys)                            # (n_sub,3) 可导
        Fper_truth = sum(Fper_truth_keys)

        terms = {}
        kept_masks = {}
        # 力臂加权场损失(建议 B): 每个子采样点到其所属车厢力矩参考点的纵向距离,
        # 按该车厢半长归一化到 O(1)。俯仰力矩误差 δM_y ≈ Σ r_x·n_z·A·δp 被长力臂
        # 放大, 故上层可用 w=1+α·lever_dist 加重车厢端部点的场损失。
        lever_dist = paddle.zeros([cent_s.shape[0]], dtype="float32")
        for region in sub_masks:
            m_s = sub_masks[region]
            # 该区域在子采样网格上无点时跳过, 否则 a_sub≈0 会放大出虚假的 0 预测损失。
            if int(m_s.astype("int64").sum()) == 0:
                continue
            ridx = int(region.split("_")[1]) - 1
            r0_r = 0.001 * paddle.to_tensor(r0[ridx], dtype="float32")
            rrel = cent_s - r0_r                                   # (n_sub,3)

            # 该车厢纵向半长(整网格上统计, 与子采样无关), 用于力臂归一化
            x_region = cent_full[full_masks[region]][:, 0]
            half_len = (x_region.max() - x_region.min()) / 2.0
            d_all = (paddle.abs(rrel[:, 0]) / (half_len + 1e-12)).clip(0.0, 2.0)
            lever_dist = paddle.where(m_s, d_all, lever_dist)

            # 面积比: 用整网格该区域面积 / 子采样该区域面积, 把部分积分放大为全量无偏估计
            a_full = areas_full[full_masks[region]].sum()
            a_sub = areas_s[m_s].sum()
            factor = a_full / (a_sub + 1e-12)

            Fp = Fper_pred[m_s]
            Ft = Fper_truth[m_s]
            rr = rrel[m_s]

            F_pred = Fp.sum(axis=0) * factor
            F_truth = Ft.sum(axis=0) * factor
            # 显式分量叉乘 r×F 求合力矩(再对点求和), 规避本环境 paddle.cross 反向
            # 在 CrossGradKernel 处段错误(SIGSEGV)的问题; 数值与 paddle.cross 完全一致。
            rx, ry, rz = rr[:, 0], rr[:, 1], rr[:, 2]
            fpx, fpy, fpz = Fp[:, 0], Fp[:, 1], Fp[:, 2]
            ftx, fty, ftz = Ft[:, 0], Ft[:, 1], Ft[:, 2]
            M_pred = paddle.stack([
                (ry * fpz - rz * fpy).sum(),
                (rz * fpx - rx * fpz).sum(),
                (rx * fpy - ry * fpx).sum(),
            ]) * factor
            M_truth = paddle.stack([
                (ry * ftz - rz * fty).sum(),
                (rz * ftx - rx * ftz).sum(),
                (rx * fty - ry * ftx).sum(),
            ]) * factor

            terms[region] = {
                "F_pred": F_pred,
                "F_truth": F_truth,
                "M_pred": M_pred,
                "M_truth": M_truth,
            }
            kept_masks[region] = m_s

        return terms, kept_masks, lever_dist


    @paddle.no_grad()
    def eval_dict(self, device, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df, area = self.data_dict_to_input(data_dict, device)
        x_in = x_in[:: self.subsample_eval, ...]
        area = area[:: self.subsample_eval]
        if self.max_in_points is not None:
            r = min(self.max_in_points, tuple(x_in.shape)[0])
            pred_chunks = []
            x_in_sections = [r] * (x_in.shape[0] // r)
            if x_in.shape[0] % r != 0:
                x_in_sections.append(-1)
            area_sections = [r] * (area.shape[0] // r)
            if area.shape[0] % r != 0:
                area_sections.append(-1)
            x_in_chunks = paddle.split(x=x_in, num_or_sections=x_in_sections, axis=0)
            area_chunks = paddle.split(x=area, num_or_sections=area_sections, axis=0)
            for j in range(len(x_in_chunks)):
                pred_index_j = super().forward(
                    x_in,
                    x_out,
                    df,
                    x_in_chunks[j],
                    area_in=area,
                    area_eval=area_chunks[j],
                )
                pred_chunks.append(pred_index_j)
                # paddle.device.cuda.empty_cache()  # clear GPU memory
            pred = paddle.concat(x=tuple(pred_chunks), axis=0)
        else:
            pred = self(x_in, x_out, df, area=area)
        pred = pred.transpose(perm=[1, 0])
        if loss_fn is None:
            loss_fn = self.loss
        out_dict = {
            "F_pred": paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True),
            "F_truth": paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True),
            "M_pred": paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True),
            "M_truth": paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True),
        }
        truth = []

        coordinate = data_dict['info'][0]["carriage_offset"]
        if coordinate is not None and ',' in coordinate:
            value_list = [float(i) for i in coordinate.split(',')]
            region_masks = self.build_region_masks(
                data_dict["centroids_no_norms"][0],
                value_list,
            )
        elif coordinate is not None and ',' not in coordinate:
            value_list = [float(coordinate)]
            region_masks = self.build_region_masks(
                data_dict["centroids_no_norms"][0],
                value_list,
            )
        else:
            region_masks = {'carriage_1':paddle.ones(data_dict["centroids_no_norms"][0].shape[0],dtype=paddle.bool)}
        for i in range(len(self.out_keys)):
            key = self.out_keys[i]
            truth_key = data_dict[key][0].to(device)[:: self.subsample_eval, ...]
            # assert not paddle.any(paddle.isnan(truth_key)), "truth_key 存在无效值！"
            if len(tuple(truth_key.shape)) == 1:
                truth_key = truth_key.reshape((-1, 1))
            truth_key = truth_key[:, : self.out_channels[i]].transpose(perm=[1, 0])
            truth.append(truth_key)
            st, end = (
                sum(self.out_channels[:i]),
                sum(self.out_channels[:i]) + self.out_channels[i],
            )
            pred_key = pred[st:end, :]
            out_dict[f"L2_{key}"] = loss_fn(pred_key, truth_key)
            if decode_fn is not None:
                # 网络输出/标签均在「z-score(系数)」空间，decode 需传入 q_ref
                # 先反 z-score 得到 Cp/Cf，再乘 q_ref 还原物理量(p/τ)，方可积分气动力。
                q_ref = data_dict["q_ref"][0]
                pred_decode = decode_fn(pred_key, i, q_ref=q_ref)
                truth_decode = decode_fn(truth_key, i, q_ref=q_ref)
                
                
                if region_masks is not None:
                    region_results = self.cal_F_M(
                        data_dict,
                        pred_decode,
                        truth_decode,
                        region_masks,
                        key=key,
                    )

                    for region, vals in region_results.items():
                        out_dict[f"F_{key}_{region}_pred"] = vals["F_pred"]
                        out_dict[f"F_{key}_{region}_truth"] = vals["F_truth"]
                        out_dict[f"M_{key}_{region}_pred"] = vals["M_pred"]
                        out_dict[f"M_{key}_{region}_truth"] = vals["M_truth"]

                        # 如果你仍然需要整车合量
                        out_dict["F_pred"] += vals["F_pred"]
                        out_dict["F_truth"] += vals["F_truth"]
                        out_dict["M_pred"] += vals["M_pred"]
                        out_dict["M_truth"] += vals["M_truth"]

        truth = paddle.concat(x=truth, axis=0)
        F_M_dict = {}
        F_M_dict.update(out_dict)
        for region, mask in region_masks.items():
            F_M_dict.update({f"F_pred_{region}": F_M_dict[f"F_pressure_{region}_pred"]+F_M_dict[f"F_wallshearstress_{region}_pred"]})
            F_M_dict.update({f"M_pred_{region}": F_M_dict[f"M_pressure_{region}_pred"]+F_M_dict[f"M_wallshearstress_{region}_pred"]})
            F_M_dict.update({f"F_truth_{region}": F_M_dict[f"F_pressure_{region}_truth"]+F_M_dict[f"F_wallshearstress_{region}_truth"]})
            F_M_dict.update({f"M_truth_{region}": F_M_dict[f"M_pressure_{region}_truth"]+F_M_dict[f"M_wallshearstress_{region}_truth"]})
        F_M_dict.update({"F_const": data_dict["F_const"][0]})
        F_M_dict.update({"M_const": data_dict["M_const"][0]})
        F_M_dict = self.integral_cd(F_M_dict, region_masks, self.out_keys)

        return out_dict, pred, truth, F_M_dict, region_masks

    @paddle.no_grad()
    def inference_dict(self, device, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df, area = self.data_dict_to_input(data_dict, device)
        x_in = x_in[:: self.subsample_eval, ...]
        area = area[:: self.subsample_eval]
        if self.max_in_points is not None:
            r = min(self.max_in_points, tuple(x_in.shape)[0])
            pred_chunks = []
            x_in_sections = [r] * (x_in.shape[0] // r)
            if x_in.shape[0] % r != 0:
                x_in_sections.append(-1)
            area_sections = [r] * (area.shape[0] // r)
            if area.shape[0] % r != 0:
                area_sections.append(-1)
            x_in_chunks = paddle.split(x=x_in, num_or_sections=x_in_sections, axis=0)
            area_chunks = paddle.split(x=area, num_or_sections=area_sections, axis=0)
            for j in range(len(x_in_chunks)):
                # t = time.perf_counter()
                pred_index_j = super().forward(
                    x_in,
                    x_out,
                    df,
                    x_in_chunks[j],
                    area_in=area,
                    area_eval=area_chunks[j],
                )
                pred_chunks.append(pred_index_j)
                # paddle.device.cuda.empty_cache()  # clear GPU memory
                # t = time.perf_counter() - t
                # print(f"chunk {j}: {t:.3f} s")
            pred = paddle.concat(x=tuple(pred_chunks), axis=0)
        else:
            pred = self(x_in, x_out, df, area=area)
        pred = pred.transpose(perm=[1, 0])
        if loss_fn is None:
            loss_fn = self.loss
        out_dict = {
            "F_pred": paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True),
            "M_pred": paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True),
        }

        coordinate = data_dict['info'][0]["carriage_offset"]
        if coordinate is not None and ',' in coordinate:
            value_list = [float(i) for i in coordinate.split(',')]
            region_masks = self.build_region_masks(
                data_dict["centroids_no_norms"][0],
                value_list,
            )
        elif coordinate is not None and ',' not in coordinate:
            value_list = [float(coordinate)]
            region_masks = self.build_region_masks(
                data_dict["centroids_no_norms"][0],
                value_list,
            )
        else:
            region_masks = {'carriage_1':paddle.ones(data_dict["centroids_no_norms"][0].shape[0],dtype=paddle.bool)}
        carriage_number = data_dict['info'][0]["carriage_number"]

        for i in range(len(self.out_keys)):
            key = self.out_keys[i]
            # assert not paddle.any(paddle.isnan(truth_key)), "truth_key 存在无效值！"

            st, end = (
                sum(self.out_channels[:i]),
                sum(self.out_channels[:i]) + self.out_channels[i],
            )
            pred_key = pred[st:end, :]
            if decode_fn is not None:
                # 推理时同样需用 q_ref 还原物理量(decode: 反 z-score -> 乘 q_ref)
                q_ref = data_dict["q_ref"][0]
                pred_decode = decode_fn(pred_key, i, q_ref=q_ref)
                r0 = data_dict["info"][0]["reference_point"]
                r0 = json.loads(r0)
                triangle_normals = data_dict["triangle_normals"][0] 
                areas = data_dict["areas"][0].reshape([-1, 1]) 
                centroids = data_dict["centroids_no_norms"][0]
                mask = region_masks[f'carriage_{carriage_number}']
                
                if key == "pressure":
                    traction_pred = -pred_decode.reshape([-1, 1]) * triangle_normals
                    F_per_pred = traction_pred * areas 
                    F_r_pred  = F_per_pred[mask]
                    F_total_pred = F_r_pred.sum(axis=0)   # (3,)
                    # 力矩（关于 r0）： sum( (r_i - r0) x F_i )
                    r_rel = centroids - 0.001*paddle.to_tensor(r0, dtype="float32") # (N,3)
                    r_rel = r_rel[mask]
                    M_per_pred = paddle.cross(r_rel, F_r_pred)
                    M_total_pred = M_per_pred.sum(axis=0)
                elif key == "wallshearstress":
                    traction_pred = pred_decode.T
                    F_per_pred = -traction_pred * areas
                    F_r_pred  = F_per_pred[mask]
                    F_total_pred = F_r_pred.sum(axis=0)

                    r_rel = centroids - 0.001*paddle.to_tensor(r0, dtype="float32") # (N,3)
                    r_rel = r_rel[mask]
                    M_per_pred = paddle.cross(r_rel, F_r_pred)
                    M_total_pred = M_per_pred.sum(axis=0)

                out_dict.update(
                    {
                        f"F_{key}_pred": F_total_pred,
                        f"M_{key}_pred": M_total_pred,
                    }
                )
                out_dict["F_pred"] += F_total_pred
                out_dict["M_pred"] += M_total_pred

        F_M_dict = {}
        F_M_dict.update({"F_pred": out_dict["F_pred"]})
        F_M_dict.update({"M_pred": out_dict["M_pred"]})
        F_M_dict.update({"F_pressure_pred": out_dict["F_pressure_pred"]})
        F_M_dict.update({"F_wallshearstress_pred": out_dict["F_wallshearstress_pred"]})
        F_M_dict.update({"M_pressure_pred": out_dict["M_pressure_pred"]})
        F_M_dict.update({"M_wallshearstress_pred": out_dict["M_wallshearstress_pred"]})
        mass_density = float(data_dict["info"][0]["density"])
        reference_area = float(data_dict["info"][0]["area"])
        typical_length = float(data_dict["info"][0]["typical_length"])
        car_speed = float(data_dict["info"][0]["car_speed"])  # already m/s
        wind_speed = float(data_dict["info"][0]["wind_speed"])
        wind_angle_rad = math.radians(float(data_dict["info"][0]["wind_angle"]))
        vx = car_speed + wind_speed * math.cos(wind_angle_rad)
        vy = wind_speed * math.sin(wind_angle_rad)
        flow_speed = math.sqrt(vx**2 + vy**2)
        F_const = 2.0 / (mass_density * flow_speed**2 * reference_area)
        M_const = 2.0 / (mass_density * flow_speed**2 * reference_area * typical_length)
        F_M_dict.update({"F_const": F_const})
        F_M_dict.update({"M_const": M_const})
        F_M_dict = self.integral_cd(F_M_dict, out_keys=self.out_keys)

        return out_dict, pred, F_M_dict

    def forward(
        self,
        data_dict,
        idx_batch,
        device=None,
        randperm=True,
        loss_fn=None,
        decode_fn=None,
    ):
        x_in, x_out, df, area = self.data_dict_to_input(data_dict, device)
        x_in = x_in[:: self.subsample_train, ...]
        area = area[:: self.subsample_train]
        r = min(self.max_in_points, tuple(x_in.shape)[0])
        if randperm:
            indices = paddle.randperm(n=tuple(x_in.shape)[0])[:r]
        else:
            indices = paddle.linspace(
                start=0, stop=tuple(x_in.shape)[0] - 1, num=r, dtype="int64"
            ).astype("int64")

        truth = []
        for i in range(len(self.out_keys)):
            truth_key = data_dict[self.out_keys[i]][0][:: self.subsample_train]
            if len(tuple(truth_key.shape)) == 1:
                truth_key = truth_key.reshape([-1, 1])
            truth_key = truth_key[indices][:, : self.out_channels[i]].to(x_in.place)
            truth.append(truth_key)
        truth = paddle.concat(x=truth, axis=-1)
        region_masks = None

        if self.integral_cd.parameters()[0].stop_gradient == True:
            pred = super().forward(
                x_in, x_out, df, x_in[indices, ...], area, area[indices]
            )
        else:
            pred = truth
            # paddle.device.cuda.empty_cache()  # clear GPU memory

        F_M_dict = {}
        # 方案 A: Phase-1(backbone 可训, integral_cd 冻结)时, 若开启 phase1_fm_loss_w,
        # 计算可微分区积分力/力矩并写入 F_M_dict, 供 train.py 追加积分损失(梯度直达 backbone)。
        if (
            self.integral_cd.parameters()[0].stop_gradient == True
            and float(getattr(self, "phase1_fm_loss_w", 0.0)) > 0.0
            and decode_fn is not None
        ):
            terms, region_masks, lever_dist = self.phase1_fm_terms(
                data_dict, pred, indices, decode_fn
            )
            F_M_dict.update({"mode": "phase1"})
            # 力臂加权场损失: 子采样点(与 pred 行对齐)的归一化纵向力臂, detach 为常量权重
            F_M_dict.update({"lever_dist": lever_dist.detach()})
            for region, v in terms.items():
                F_M_dict[f"F_pred_{region}"] = v["F_pred"]
                F_M_dict[f"F_truth_{region}"] = v["F_truth"]
                F_M_dict[f"M_pred_{region}"] = v["M_pred"]
                F_M_dict[f"M_truth_{region}"] = v["M_truth"]

        if self.integral_cd.parameters()[0].stop_gradient == False:
            # cd_dict = self.integral_cd(pred, truth, self.out_channels,
            #    data_dict, decode_fn=decode_fn,
            #    out_keys=self.out_keys,
            #    subsample_train=self.subsample_train)

            F_M_dict.update({"OOM": False})
            try:
                out_dict, _, _, F_M, region_masks= self.eval_dict(
                    device, data_dict, loss_fn=loss_fn, decode_fn=decode_fn
                )
                F_M_dict.update(out_dict)
                for region, mask in region_masks.items():
                    F_M_dict.update({f"F_pred_{region}": F_M_dict[f"F_pressure_{region}_pred"]+F_M_dict[f"F_wallshearstress_{region}_pred"]})
                    F_M_dict.update({f"M_pred_{region}": F_M_dict[f"M_pressure_{region}_pred"]+F_M_dict[f"M_wallshearstress_{region}_pred"]})
                    F_M_dict.update({f"F_truth_{region}": F_M_dict[f"F_pressure_{region}_truth"]+F_M_dict[f"F_wallshearstress_{region}_truth"]})
                    F_M_dict.update({f"M_truth_{region}": F_M_dict[f"M_pressure_{region}_truth"]+F_M_dict[f"M_wallshearstress_{region}_truth"]})
                F_M_dict.update({"F_const": data_dict["F_const"][0]})
                F_M_dict.update({"M_const": data_dict["M_const"][0]})
                F_M_dict = self.integral_cd(F_M_dict, region_masks, self.out_keys)

            except MemoryError as e:
                if "Out of memory" in str(e):
                    print(f"WARNING: OOM on sample {idx_batch}, skipping this sample.")
                    if hasattr(paddle.device.cuda, "empty_cache"):
                        paddle.device.cuda.empty_cache()
                    F_M_dict.update({"OOM": True})
                else:
                    raise


        return pred.transpose(perm=[1, 0]), truth.transpose(perm=[1, 0]), F_M_dict, region_masks
