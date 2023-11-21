from tqdm import tqdm

class TqdmToQueue(tqdm):
    def __init__(self, *args, log_queue=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_queue = log_queue

    def display(self, msg=None, pos=None):
        # 将 tqdm 的输出信息放入 log_queue
        if self.log_queue is not None:
            self.log_queue.append(self.format_meter(self.n, self.total, self.elapsed))
        super().display(msg=msg, pos=pos)
