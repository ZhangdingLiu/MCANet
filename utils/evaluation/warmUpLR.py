import torch


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

# optimizer: The optimizer whose learning rate needs to be scheduled.
# total_iters: The total number of iterations (not epochs) after which the
#             warm-up period ends. The learning rate will linearly increase over these iterations.
# last_epoch: This argument is used by the base scheduler class to keep track
#             of the epoch index. It defaults to -1, which is commonly used to indicate that
#             the scheduler is starting.

# The learning rate is calculated as a fraction of the base_lr (the initial learning rates
# for each parameter group). This fraction is the ratio of the current epoch (self.last_epoch)
# to the total number of iterations (self.total_iters), effectively creating a linear warm-up
# from 0 to the base learning rate.


# Usage
# The WarmUpLR scheduler should be used at the start of training when the model is potentially
# sensitive to the initial random weights. By slowly ramping up the learning rate, it helps
# in achieving a stable and reliable convergence early in training.
#
# To use this scheduler, you typically instantiate it with the optimizer and the total number
# of iterations you want for the warm-up period, and then update the learning rates during
# training by calling scheduler.step() at each iteration (not epoch, if your warm-up period
# is defined in terms of iterations). Make sure that total_iters aligns with how you define
# an iteration in your training loop (usually, it's num_epochs * num_batches_per_epoch).