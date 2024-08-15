_base_ = [
    '../_base_/datasets/cityscapes.py',
    # '../_base_/datasets/u4k.py',
    '../_base_/datasets/general_dataset.py',
    '../_base_/run_time.py',
    './base_pr_s2r_optim.py',
]

max_depth = 250
min_depth = 1e-3

zoe_depth_config=dict(
    type='ZoeDepth',
    
    # depth range setting
    max_depth=max_depth,
    min_depth=min_depth,

    # some important params
    midas_model_type='DPT_BEiT_L_384',
    pretrained_resource='local::/home/liz0l/shortcuts/monodepth3_checkpoints/ZoeDepthv1_30-Dec_16-29-4e2bc436e4e1_best.pt',
    use_pretrained_midas=True,
    train_midas=True,
    freeze_midas_bn=True,
    do_resize=False, # do not resize image in midas

    # default settings
    attractor_alpha=1000,
    attractor_gamma=2,
    attractor_kind='mean',
    attractor_type='inv',
    aug=True,
    bin_centers_type='softplus',
    bin_embedding_dim=128,
    clip_grad=0.1,
    dataset='nyu',
    distributed=True,
    force_keep_ar=True,
    gpu='NULL',
    img_size=[384, 512],
    inverse_midas=False,
    log_images_every=0.1,
    max_temp=50.0,
    max_translation=100,
    memory_efficient=True,
    min_temp=0.0212,
    model='zoedepth',
    n_attractors=[16, 8, 4, 1],
    n_bins=64,
    name='ZoeDepth',
    notes='',
    output_distribution='logbinomial',
    prefetch=False,
    print_losses=False,
    project='ZoeDepth',
    random_crop=False,
    random_translate=False,
    root='.',
    save_dir='',
    shared_dict='NULL',
    tags='',
    translate_prob=0.2,
    uid='NULL',
    use_amp=False,
    use_shared_dict=False,
    validate_every=0.25,
    version_name='v1',
    workers=16,
)


sub_model_student=dict(
    type='PatchRefiner',
    config=dict(
        image_raw_shape=[1024, 2048],
        patch_process_shape=[384, 512],
        patch_raw_shape=[256, 512], 
        patch_split_num=[4, 4],

        fusion_feat_level=6,
        min_depth=min_depth,
        max_depth=max_depth,

        pretrain_coarse_model='./work_dir/zoedepth/cs/coarse_pretrain/checkpoint_05.pth', # will load the coarse model     
        strategy_refiner_target='offset_coarse',
        
        coarse_branch=zoe_depth_config,
        refiner=dict(
            fine_branch=zoe_depth_config,
            fusion_model=dict(
                type='FusionUnet',
                input_chl=[32*2, 256*2, 256*2, 256*2, 256*2, 256*2],
                temp_chl=[32, 256, 256, 256, 256, 256],
                dec_chl=[256, 256, 256, 256, 32])),
        
        sigloss=dict(type='SILogLoss'),
        load_whole=True,
        pretrained='./work_dir/zoedepth/cs/pr/checkpoint_05.pth',
        pre_norm_bbox=True,
))

zoe_depth_teacher_config=dict(
    type='ZoeDepth',
    
    min_depth=min_depth,
    max_depth=max_depth,
    
    # some important params
    midas_model_type='DPT_BEiT_L_384',
    pretrained_resource=None,
    use_pretrained_midas=True,
    train_midas=True,
    freeze_midas_bn=True,
    do_resize=False, # do not resize image in midas

    # default settings
    attractor_alpha=1000,
    attractor_gamma=2,
    attractor_kind='mean',
    attractor_type='inv',
    aug=True,
    bin_centers_type='softplus',
    bin_embedding_dim=128,
    clip_grad=0.1,
    dataset='nyu',
    distributed=True,
    force_keep_ar=True,
    gpu='NULL',
    img_size=[384, 512],
    inverse_midas=False,
    log_images_every=0.1,
    max_temp=50.0,
    max_translation=100,
    memory_efficient=True,
    min_temp=0.0212,
    model='zoedepth',
    n_attractors=[16, 8, 4, 1],
    n_bins=64,
    name='ZoeDepth',
    notes='',
    output_distribution='logbinomial',
    prefetch=False,
    print_losses=False,
    project='ZoeDepth',
    random_crop=False,
    random_translate=False,
    root='.',
    save_dir='',
    shared_dict='NULL',
    tags='',
    translate_prob=0.2,
    uid='NULL',
    use_amp=False,
    use_shared_dict=False,
    validate_every=0.25,
    version_name='v1',
    workers=16,
)

model_cfg_teacher=dict(
    type='PatchRefiner',
    config=dict(
        image_raw_shape=[1024, 2048],
        patch_process_shape=[384, 512],
        patch_raw_shape=[256, 512], 
        patch_split_num=[4, 4],

        fusion_feat_level=6,
        min_depth=min_depth,
        max_depth=max_depth,

        pretrain_coarse_model='./work_dir/zoedepth/u4k/coarse_pretrain/checkpoint_24.pth', # will load the coarse model     
        strategy_refiner_target='offset_coarse',
        
        coarse_branch=zoe_depth_teacher_config,
        refiner=dict(
            fine_branch=zoe_depth_teacher_config,
            fusion_model=dict(
                type='FusionUnet',
                input_chl=[32*2, 256*2, 256*2, 256*2, 256*2, 256*2],
                temp_chl=[32, 256, 256, 256, 256, 256],
                dec_chl=[256, 256, 256, 256, 32])),
        
        sigloss=dict(type='SILogLoss'),
        load_whole=True,
        pretrained='./work_dir/zoedepth/u4k/pr/checkpoint_36.pth',
        pre_norm_bbox=True,
))

model=dict(
    type='PatchRefinerSemi',
    model_cfg_student=sub_model_student,
    model_cfg_teacher=model_cfg_teacher,
    mix_loss=False,
    edge_loss_weight=1,
    edgeloss=dict(
        type='ScaleAndShiftInvariantLoss',
        only_missing_area=False),
    sigloss=dict(type='SILogLoss'),
    min_depth=min_depth,
    max_depth=max_depth,)

collect_input_args=['image_lr', 'image_hr', 'crops_image_hr', 'depth_gt', 'crop_depths', 'bboxs', 'center_mask', 'pseudo_label', 'seg_image']

project='patchrefiner'
# train_cfg=dict(max_epochs=2, val_interval=500, save_checkpoint_interval=2, log_interval=100, train_log_img_interval=100, val_log_img_interval=50, val_type='iter_base', eval_start=0)
train_cfg=dict(max_epochs=2, val_interval=1, save_checkpoint_interval=2, log_interval=100, train_log_img_interval=1000, val_log_img_interval=50, val_type='epoch_base', eval_start=0)

convert_syncbn=True
find_unused_parameters=True

train_dataloader=dict(
    dataset=dict(
        # pseudo_label_path='./work_dir/zoedepth/u4k/patchrefiner/generate_pls_cs',
        pseudo_label_path='./work_dir/project_folder/zoedepth/u4k/patchrefiner/generate_pls_cs_ctnum',
        # with_pseudo_label=True,
        with_pseudo_label=False,
        transform_cfg=dict(
            image_raw_shape=[1024, 2048])))

