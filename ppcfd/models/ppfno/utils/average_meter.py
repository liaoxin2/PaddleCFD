class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterDict:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, val, n=1):
        for k, v in val.items():
            if k not in self.val:
                self.val[k] = 0
                self.sum[k] = 0
                self.count[k] = 0
            self.val[k] = v
            self.sum[k] += v * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]

    def sync(self):
        """跨所有 GPU 做 all_reduce，使 avg 反映全局所有样本的均值。"""
        import numpy as np
        import paddle
        import paddle.distributed as dist

        for k in list(self.sum.keys()):
            local_sum = self.sum[k]
            # 支持标量和 numpy 数组两种情况
            if isinstance(local_sum, np.ndarray):
                t_sum = paddle.to_tensor(local_sum.copy(), dtype="float32")
            else:
                t_sum = paddle.to_tensor([float(local_sum)], dtype="float32")

            t_count = paddle.to_tensor([float(self.count[k])], dtype="float32")

            dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_count, op=dist.ReduceOp.SUM)

            global_sum = t_sum.numpy()
            global_count = t_count.numpy()[0]

            if isinstance(local_sum, np.ndarray):
                self.sum[k] = global_sum
            else:
                self.sum[k] = float(global_sum[0])

            self.count[k] = global_count
            self.avg[k] = self.sum[k] / self.count[k]
