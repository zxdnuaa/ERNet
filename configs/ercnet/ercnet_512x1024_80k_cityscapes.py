_base_ = [
    '../_base_/models/ercnet.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
)
