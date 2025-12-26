import paddle
import paddle.nn as nn


class Integral_Cd(paddle.nn.Layer):
    def __init__(self, layers=None, dropout=False, normalize=False):
        super().__init__()
        if layers == None:
            layers = [2, 64, 128, 64, 1]
        self.n_layers = len(layers) - 1
        self.layers = nn.LayerList()
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(in_features=layers[i],
                                         out_features=layers[i + 1]))
            if i < self.n_layers - 1:                
                if normalize:
                    self.layers.append(paddle.nn.BatchNorm(num_channels=layers[i+1]))
                if dropout:
                    self.layers.append(nn.dropout(p=0.2))
                self.layers.append(nn.GELU())         

    def forward(self, F_M_dict, out_keys=None, ):    
        
        F_pressure_pred = F_M_dict['F_pressure_pred']
        F_wallshearstress_pred = F_M_dict['F_wallshearstress_pred']
        M_pressure_pred = F_M_dict['M_pressure_pred']
        M_wallshearstress_pred = F_M_dict['M_wallshearstress_pred']
        F_pred = paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True)
        M_pred = paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True)

        for i in range(F_pressure_pred.shape[0]):
            
            F = paddle.to_tensor([F_pressure_pred[i], F_wallshearstress_pred[i]]).cuda(blocking=True)
            M = paddle.to_tensor([M_pressure_pred[i], M_wallshearstress_pred[i]]).cuda(blocking=True)
            for _, layer in enumerate(self.layers):
                F = layer(F)
                M = layer(M)
            F_pred[i] = F#paddle.Tensor.sigmoid(F) * (0.6 - 0.1) + 0.1
            M_pred[i] = M#paddle.Tensor.sigmoid(M) * (0.6 - 0.1) + 0.1
        F_M_dict.update({'F_pred_modify': F_pred, 'M_pred_modify': M_pred})
        
        return F_M_dict
    
    
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
        cd_dict = {'Cd_pred': paddle.to_tensor(data=0.0).cuda(blocking=True), 
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
                drag_pred = paddle.sum(x=drag_weight * pred_decode)
                drag_truth = paddle.sum(x=drag_weight * truth_decode)
                cd_dict.update({f'Cd_{key}_pred': drag_pred,
                                f'Cd_{key}_truth': drag_truth})
                cd_dict['Cd_pred'] += drag_pred
                cd_dict['Cd_truth'] += drag_truth
        return cd_dict



# class Integral_Cd(paddle.nn.Layer):
#     def __init__(self, layers=None, dropout=False, normalize=False):
#         super().__init__()

#         # 默认网络结构 6 → 64 → 64 → 3
#         if layers is None:
#             layers = [6, 64, 64, 3]

#         self.n_layers = len(layers) - 1
#         net_layers = []

#         for i in range(self.n_layers):
#             # 线性层
#             net_layers.append(nn.Linear(layers[i], layers[i+1]))

#             # 中间层添加 BN、Dropout、GELU
#             if i < self.n_layers - 1:
#                 if normalize:
#                     net_layers.append(nn.BatchNorm(layers[i+1]))
#                 if dropout:
#                     net_layers.append(nn.Dropout(p=0.2))
#                 net_layers.append(nn.GELU())

#         self.layers = nn.LayerList(net_layers)


#     def forward(self, F_M_dict, out_keys=None, ):    

#         F_pressure_pred = F_M_dict['F_pressure_pred']
#         F_wallshearstress_pred = F_M_dict['F_wallshearstress_pred']
#         M_pressure_pred = F_M_dict['M_pressure_pred']
#         M_wallshearstress_pred = F_M_dict['M_wallshearstress_pred']
#         F_pred = paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True)
#         M_pred = paddle.to_tensor(data=[0.0, 0.0, 0.0]).cuda(blocking=True)

#         F = paddle.to_tensor([F_M_dict['F_pressure_pred'],
#                                 F_M_dict['F_wallshearstress_pred']]).reshape([1,6]).cuda(blocking=True)
#         M = paddle.to_tensor([F_M_dict['M_pressure_pred'],
#                                 F_M_dict['M_wallshearstress_pred']]).reshape([1,6]).cuda(blocking=True)
#         # 网络前向计算
#         for layer in self.layers:
#             F = layer(F)
#             M = layer(M)

#         F_pred = F[0]
#         M_pred = M[0]

#         F_M_dict.update({'F_pred_modify': F_pred, 'M_pred_modify': M_pred})

#         return F_M_dict


# def forward(self, F_M_dict, out_keys=None, ):    
        
#         F_pressure_pred = F_M_dict['F_pressure_pred']
#         F_wallshearstress_pred = F_M_dict['F_wallshearstress_pred']
#         M_pressure_pred = F_M_dict['M_pressure_pred']
#         M_wallshearstress_pred = F_M_dict['M_wallshearstress_pred']
#         F_M_pred = paddle.to_tensor([F_pressure_pred[0], F_wallshearstress_pred[0], 
#                                     F_pressure_pred[1], F_wallshearstress_pred[1],
#                                     F_pressure_pred[2], F_wallshearstress_pred[2],
#                                     M_pressure_pred[0], M_wallshearstress_pred[0],
#                                     M_pressure_pred[1], M_wallshearstress_pred[1],
#                                     M_pressure_pred[2], M_wallshearstress_pred[2]]).cuda(blocking=True)

        
#         for _, layer in enumerate(self.layers):
#             F_M_pred = layer(F_M_pred)
#         F_pred = F_M_pred[:3]#paddle.Tensor.sigmoid(F) * (0.6 - 0.1) + 0.1
#         M_pred = F_M_pred[3:]#paddle.Tensor.sigmoid(M) * (0.6 - 0.1) + 0.1
#         F_M_dict.update({'F_pred_modify': F_pred, 'M_pred_modify': M_pred})
        
#         return F_M_dict
