

train_dataloader=dict(
    batch_size=4,
    num_workers=6,
    # num_workers=2,
    dataset=dict(
        type='ScanNetDataset',
        mode='train',
        split='/ibex/ai/home/liz0l/codes/datasets/scannet_pp_select_new/nvs_sem_train_subset.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            degree=1.0,
            random_crop=True,
            random_crop_size=[720, 960],
            network_process_size=[384, 512])))

val_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='ScanNetDataset',
        mode='infer',
        # split='/ibex/ai/home/liz0l/codes/datasets/scannet_pp_val_hr_depth/nvs_sem_val.txt',
        split='/ibex/ai/home/liz0l/codes/datasets/scannet_pp_select_val_lr/nvs_sem_val.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            degree=1.0,
            network_process_size=[384, 512])))


