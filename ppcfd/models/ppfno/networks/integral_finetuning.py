import paddle
import paddle.nn as nn


class Integral_Cd(paddle.nn.Layer):
    """
    改进版 Integral_Cd：
      1. 为 F 的三个分量（阻力/升力/侧向力）和 M 的三个分量分别建立独立子网络，
         共 6 个子网络，避免不同物理特性的分量共享参数。
      2. 在「系数空间」做修正（真·系数空间，全链路）：
         输入 = 压差/摩擦物理贡献 × 工况系数 const（F_const/M_const = 2/(ρv²A[·L])）
              = 无量纲系数贡献 Cp/Cf，量级 O(1)、与车速/风速解耦；
         输出 = 无量纲力/力矩系数修正量（不再折回物理力）。
         回归目标恒为 O(1)，梯度量级与工况彻底解耦，条件数与外推稳健性最佳。
         物理力/力矩的还原（× 1/const）由使用侧（train 日志 / inference）负责。
    """

    def __init__(self, layers=None, dropout=False, normalize=False,
                 use_cond_feat=False, coef_space=True, return_coef=True,
                 residual=True, cond_eps=1e-12):
        super().__init__()
        if layers is None:
            layers = [2, 64, 128, 64, 1]

        # coef_space=True：子网络输入用无量纲系数贡献（物理贡献 × const），O(1)。
        # return_coef=True（真·系数空间，推荐）：forward 直接输出无量纲力/力矩系数
        #   修正量，不在网络内折回物理量；物理还原(× 1/const)交由使用侧负责，
        #   使梯度量级与工况彻底解耦。
        # return_coef=False：forward 内部 ×(1/const) 折回物理力/力矩（旧接口，
        #   上下游零改动，但梯度量级仍随工况变化）。
        # coef_space=False：完全退化为旧的物理力空间修正（向后兼容）。
        self.coef_space = coef_space
        self.return_coef = return_coef
        # residual=True（推荐）：子网络输出解释为「残差修正量 Δ」，最终输出
        #   = base_coef + Δ，其中 base_coef 为 FNO 基预测在系数空间的值
        #   (压差贡献+摩擦贡献)×const。这样结构上修正量只能在 base 附近微调，
        #   无法把符号从 base 翻走（base FNO 的力矩符号翻转本就极少）。
        #   Δ 会写入 F_M_dict 供上层做 L2 正则 λ·(Δ/scale)²。
        # residual=False：旧行为，子网络直接输出完整系数。
        self.residual = residual
        self.cond_eps = cond_eps
        self.use_cond_feat = use_cond_feat

        # 系数空间下工况已被 const 归一化掉，默认不再拼接工况特征；
        # 仅当 use_cond_feat=True 时才追加一维（输入维度 +1）。
        in_dim = layers[0] + (1 if use_cond_feat else 0)
        actual_layers = [in_dim] + list(layers[1:])

        # 6 个独立子网络
        # nets_F: [0]=阻力(x), [1]=升力(y), [2]=侧向力(z)
        # nets_M: [0]=倾覆力矩, [1]=俯仰力矩, [2]=横摇力矩
        self.nets_F = nn.LayerList([
            self._build_net(actual_layers, dropout, normalize) for _ in range(3)
        ])
        self.nets_M = nn.LayerList([
            self._build_net(actual_layers, dropout, normalize) for _ in range(3)
        ])

        # 兼容旧接口 forward_v1：self.layers 指向阻力子网络
        self.layers = self.nets_F[0]

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _build_net(self, layers, dropout, normalize):
        """构建单个子网络，返回 nn.LayerList。"""
        net = nn.LayerList()
        n_layers = len(layers) - 1
        for i in range(n_layers):
            net.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            if i < n_layers - 1:
                if normalize:
                    net.append(nn.BatchNorm(num_channels=layers[i + 1]))
                if dropout:
                    net.append(nn.Dropout(p=0.2))
                net.append(nn.GELU())
        return net

    def _forward_net(self, net, x):
        """将输入 x 逐层通过子网络 net。"""
        for layer in net:
            x = layer(x)
        return x

    def _make_input(self, p_val, wss_val, cond_scalar):
        """
        构建子网络输入。

        coef_space=True（推荐）：将压差/摩擦的物理贡献乘以工况系数 const
        （F_const 或 M_const = 2/(ρv²A[·L])）换算为无量纲系数 Cp/Cf，使输入 O(1)、
        与车速/风速解耦。const 缺失时退化为 1.0（等价物理空间）。

        coef_space=False：沿用旧的物理贡献 [p_val, wss_val]。

        use_cond_feat=True 时再额外拼接一维工况特征（默认 False）。
        """
        const = float(cond_scalar) if cond_scalar is not None else None
        if self.coef_space:
            scale = const if (const is not None and abs(const) > self.cond_eps) else 1.0
            feat = paddle.to_tensor(
                [float(p_val) * scale, float(wss_val) * scale], dtype='float32'
            ).cuda(blocking=True)
        else:
            feat = paddle.to_tensor(
                [float(p_val), float(wss_val)], dtype='float32'
            ).cuda(blocking=True)
        if self.use_cond_feat:
            cond_val = const if const is not None else 0.0
            cond = paddle.to_tensor([cond_val], dtype='float32').cuda(blocking=True)
            feat = paddle.concat([feat, cond])
        return feat

    def _base_coef(self, p_val, wss_val, cond_scalar):
        """FNO 基预测在系数空间的值 = (压差贡献 + 摩擦贡献) × const。
        与 _make_input 的缩放一致（输入两维之和即基系数）。const 缺失时退化为
        物理空间之和。返回 python float（脱离梯度图，base 为冻结 backbone 的常量，
        仅残差 Δ 参与训练）。"""
        const = float(cond_scalar) if cond_scalar is not None else None
        if self.coef_space:
            scale = const if (const is not None and abs(const) > self.cond_eps) else 1.0
            return (float(p_val) + float(wss_val)) * scale
        return float(p_val) + float(wss_val)

    def _finalize(self, coef_out, cond_scalar):
        """将子网络输出整理为最终输出量。

        return_coef=True（真·系数空间）：直接返回无量纲系数修正量，不折回物理量。
        return_coef=False：物理值 = 系数 / const（const=2/(ρv²A[·L])）折回物理力/力矩。
        coef_space=False 或 const 无效时按恒等处理（网络已在物理空间）。
        """
        if not self.coef_space or self.return_coef:
            return coef_out
        const = float(cond_scalar) if cond_scalar is not None else None
        if const is None or abs(const) <= self.cond_eps:
            return coef_out
        return coef_out / const

    # ------------------------------------------------------------------
    # 主前向接口
    # ------------------------------------------------------------------

    def forward(self, F_M_dict, region_masks=None, out_keys=None):
        # 工况特征（F_const = 2/(ρv²A)，M_const 类似）
        # eval_dict / train_dict 路径均会将其写入 F_M_dict
        cond_F = F_M_dict.get('F_const', None)
        cond_M = F_M_dict.get('M_const', None)

        if region_masks is None:
            # ---- 不分区域 ----
            F_pred = paddle.to_tensor([0.0, 0.0, 0.0]).cuda(blocking=True)
            M_pred = paddle.to_tensor([0.0, 0.0, 0.0]).cuda(blocking=True)
            F_delta = paddle.to_tensor([0.0, 0.0, 0.0]).cuda(blocking=True)
            M_delta = paddle.to_tensor([0.0, 0.0, 0.0]).cuda(blocking=True)
            F_p   = F_M_dict['F_pressure_pred']
            F_wss = F_M_dict['F_wallshearstress_pred']
            M_p   = F_M_dict['M_pressure_pred']
            M_wss = F_M_dict['M_wallshearstress_pred']

            for i in range(3):
                F_in = self._make_input(F_p[i], F_wss[i], cond_F)
                M_in = self._make_input(M_p[i], M_wss[i], cond_M)
                F_out = self._forward_net(self.nets_F[i], F_in)
                M_out = self._forward_net(self.nets_M[i], M_in)
                if self.residual:
                    # 系数空间残差: 最终系数 = base + Δ，Δ=网络输出（可正则）
                    F_delta[i] = F_out
                    M_delta[i] = M_out
                    F_pred[i] = self._base_coef(F_p[i], F_wss[i], cond_F) + F_out
                    M_pred[i] = self._base_coef(M_p[i], M_wss[i], cond_M) + M_out
                else:
                    F_delta[i] = F_out
                    M_delta[i] = M_out
                    F_pred[i] = self._finalize(F_out, cond_F)
                    M_pred[i] = self._finalize(M_out, cond_M)

            F_M_dict.update({'F_pred_modify': F_pred, 'M_pred_modify': M_pred,
                             'F_delta': F_delta, 'M_delta': M_delta})

        else:
            # ---- 分区域 ----
            for region in region_masks:
                F_pred = paddle.to_tensor([0.0, 0.0, 0.0]).cuda(blocking=True)
                M_pred = paddle.to_tensor([0.0, 0.0, 0.0]).cuda(blocking=True)
                F_delta = paddle.to_tensor([0.0, 0.0, 0.0]).cuda(blocking=True)
                M_delta = paddle.to_tensor([0.0, 0.0, 0.0]).cuda(blocking=True)
                F_p   = F_M_dict[f'F_pressure_{region}_pred']
                F_wss = F_M_dict[f'F_wallshearstress_{region}_pred']
                M_p   = F_M_dict[f'M_pressure_{region}_pred']
                M_wss = F_M_dict[f'M_wallshearstress_{region}_pred']

                for i in range(3):
                    F_in = self._make_input(F_p[i], F_wss[i], cond_F)
                    M_in = self._make_input(M_p[i], M_wss[i], cond_M)
                    F_out = self._forward_net(self.nets_F[i], F_in)
                    M_out = self._forward_net(self.nets_M[i], M_in)
                    if self.residual:
                        # 系数空间残差: 最终系数 = base + Δ
                        F_delta[i] = F_out
                        M_delta[i] = M_out
                        F_pred[i] = self._base_coef(F_p[i], F_wss[i], cond_F) + F_out
                        M_pred[i] = self._base_coef(M_p[i], M_wss[i], cond_M) + M_out
                    else:
                        F_delta[i] = F_out
                        M_delta[i] = M_out
                        F_pred[i] = self._finalize(F_out, cond_F)
                        M_pred[i] = self._finalize(M_out, cond_M)

                F_M_dict.update({
                    f'F_pred_{region}_modify': F_pred,
                    f'M_pred_{region}_modify': M_pred,
                    f'F_delta_{region}': F_delta,
                    f'M_delta_{region}': M_delta,
                })

        return F_M_dict

    # ------------------------------------------------------------------
    # 旧接口（保持向后兼容，forward_v1 / get_cd 逻辑不变）
    # ------------------------------------------------------------------

    def forward_v1(self, pred, truth: paddle.Tensor,
                   out_channels, data_dict,
                   decode_fn=None, out_keys=None, subsample_train=None):
        pred = pred.transpose(perm=[1, 0])
        truth = truth.transpose(perm=[1, 0])
        cd_dict = self.get_cd(pred, truth,
                              out_channels, data_dict,
                              decode_fn=decode_fn, out_keys=out_keys,
                              subsample_train=subsample_train)
        cd_pred = paddle.to_tensor([cd_dict[f'Cd_{out_keys[0]}_pred'],
                                    cd_dict[f'Cd_{out_keys[1]}_pred']]).cuda(blocking=True)
        for _, layer in enumerate(self.layers):
            cd_pred = layer(cd_pred)
        cd_dict.update({'Cd_pred_modify': cd_pred})
        return cd_dict

    def get_cd(self, pred, truth, out_channels, data_dict,
               decode_fn, out_keys, subsample_train):
        cd_dict = {'Cd_pred':  paddle.to_tensor(data=0.0).cuda(blocking=True),
                   'Cd_truth': paddle.to_tensor(data=0.0).cuda(blocking=True)}
        truth = []
        for i in range(len(out_keys)):
            key = out_keys[i]
            truth_key = data_dict[key][0][::subsample_train, ...]
            if len(tuple(truth_key.shape)) == 1:
                truth_key = truth_key.reshape((-1, 1))
            truth_key = truth_key[:, :out_channels[i]].transpose(perm=[1, 0])
            truth.append(truth_key)
            st, end = sum(out_channels[:i]), sum(out_channels[:i]) + out_channels[i]
            pred_key = pred[st:end, :]
            if decode_fn is not None:
                pred_decode = decode_fn(pred_key, i)
                truth_decode = decode_fn(truth_key, i)
                if key == 'pressure':
                    drag_weight = data_dict['dragWeight'][0].cuda(blocking=True)
                    drag_weight = drag_weight[::subsample_train]
                elif key == 'wallshearstress':
                    drag_weight = data_dict['dragWeightWss'][0][:out_channels[i], :].cuda(blocking=True)
                    drag_weight = drag_weight[..., ::subsample_train]
                drag_pred  = paddle.sum(x=drag_weight * pred_decode)
                drag_truth = paddle.sum(x=drag_weight * truth_decode)
                cd_dict.update({f'Cd_{key}_pred':  drag_pred,
                                f'Cd_{key}_truth': drag_truth})
                cd_dict['Cd_pred']  += drag_pred
                cd_dict['Cd_truth'] += drag_truth
        return cd_dict



# class Integral_Cd(paddle.nn.Layer):
#     def __init__(self, layers=None, dropout=False, normalize=False):
#         super().__init__()
#         if layers == None:
#             layers = [2, 64, 128, 64, 1]
#         self.n_layers = len(layers) - 1
#         self.layers = nn.LayerList()
#         for i in range(self.n_layers):
#             self.layers.append(nn.Linear(in_features=layers[i],
#                                          out_features=layers[i + 1]))
#             if i < self.n_layers - 1:                
#                 if normalize:
#                     self.layers.append(paddle.nn.BatchNorm(num_channels=layers[i+1]))
#                 if dropout:
#                     self.layers.append(nn.dropout(p=0.2))
#                 self.layers.append(nn.GELU())         

#     def forward(self, F_M_dict, region_masks=None, out_keys=None):    

#         if region_masks == None:
#             F_pressure_pred = F_M_dict['F_pressure_pred']
#             F_wallshearstress_pred = F_M_dict['F_wallshearstress_pred']
#             M_pressure_pred = F_M_dict['M_pressure_pred']
#             M_wallshearstress_pred = F_M_dict['M_wallshearstress_pred']
#             F_pred = paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True)
#             M_pred = paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True)

#             for i in range(F_pressure_pred.shape[0]):
                
#                 F = paddle.to_tensor([F_pressure_pred[i], F_wallshearstress_pred[i]]).cuda(blocking=True)
#                 M = paddle.to_tensor([M_pressure_pred[i], M_wallshearstress_pred[i]]).cuda(blocking=True)
#                 for _, layer in enumerate(self.layers):
#                     F = layer(F)
#                     M = layer(M)
#                 F_pred[i] = F#paddle.Tensor.sigmoid(F) * (0.6 - 0.1) + 0.1
#                 M_pred[i] = M#paddle.Tensor.sigmoid(M) * (0.6 - 0.1) + 0.1
#             F_M_dict.update({'F_pred_modify': F_pred, 'M_pred_modify': M_pred})
#         else:
#             for region, mask in region_masks.items():
            
#                 F_pressure_pred = F_M_dict[f'F_pressure_{region}_pred']
#                 F_wallshearstress_pred = F_M_dict[f'F_wallshearstress_{region}_pred']
#                 M_pressure_pred = F_M_dict[f'M_pressure_{region}_pred']
#                 M_wallshearstress_pred = F_M_dict[f'M_wallshearstress_{region}_pred']
#                 F_pred = paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True)
#                 M_pred = paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True)

#                 for i in range(F_pressure_pred.shape[0]):
                    
#                     F = paddle.to_tensor([F_pressure_pred[i], F_wallshearstress_pred[i]]).cuda(blocking=True)
#                     M = paddle.to_tensor([M_pressure_pred[i], M_wallshearstress_pred[i]]).cuda(blocking=True)
#                     for _, layer in enumerate(self.layers):
#                         F = layer(F)
#                         M = layer(M)
#                     F_pred[i] = F#paddle.Tensor.sigmoid(F) * (0.6 - 0.1) + 0.1
#                     M_pred[i] = M#paddle.Tensor.sigmoid(M) * (0.6 - 0.1) + 0.1
#                 F_M_dict.update({f'F_pred_{region}_modify': F_pred, f'M_pred_{region}_modify': M_pred})
        
#         return F_M_dict