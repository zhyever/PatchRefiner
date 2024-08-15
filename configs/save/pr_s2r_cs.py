_base_ = [
    '../_base_/datasets/cityscape.py',
    '../_base_/datasets/general_dataset.py',
    '../_base_/run_time.py',
    './base_pr_s2r_optim.py',
]

max_depth = 80
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
    type='PatchFusionV2Clean',
    patch_raw_shape=[256, 512], 
    
    fusion_feat_level=6,
    min_depth=min_depth,
    max_depth=max_depth,
    load_whole=True,
    pretrained='./work_dir/cityscape/pr/checkpoint_24.pth', # use cs ckp?
    pretrain_model=['', './work_dir/cityscape/coarse_pretrain/checkpoint_24.pth'], # will load the coarse model    
    strategy_supervise='whole_image',
    strategy_inference='whole_image',
    strategy_fine_supervise='whole_image',
    strategy_refiner_target='offset_coarse',
    supervision_weight=0.0,
    
    use_glb=False,
    coarse_branch=zoe_depth_config,
    refiner=dict(
        fine_branch=zoe_depth_config,
        fusion_model=dict(
            type='FusionUnet',
            input_chl=[32*2, 256*2, 256*2, 256*2, 256*2, 256*2],
            temp_chl=[32, 256, 256, 256, 256, 256],
            dec_chl=[256, 256, 256, 256, 32],
            refiner_mask=False)),
    
    sigloss=dict(type='SILogLoss'))

model=dict(
    type='PatchFusionV2SemiModel',
    # model_cfg_teacher=None,
    model_cfg_student=sub_model_student,
    teacher_pretrain='./work_dir/nfs_fixbug/patchfusionv2/checkpoint_36.pth', # teacher ckp is loaded here
    mix_loss=True,
    edge_loss_weight=1,
    ranking_weight=0.1,
    ssi_weight=0.1,
    edgeloss_ranking=dict(
        type='EdgeguidedRankingLoss',
        min_depth=-1e-3, 
        max_depth=max_depth,
        alpha=1,
        reweight_target=False,
        only_missing_area=False,
        point_pairs=10000),
    edgeloss_ssi=dict(
        type='ScaleAndShiftInvariantLoss',
        only_missing_area=False,),
    sigloss=dict(type='SILogLoss'),
    min_depth=min_depth,
    max_depth=max_depth,)

collect_input_args=['image_lr', 'image_hr', 'crops_image_hr', 'depth_gt', 'crop_depths', 'bboxs', 'center_mask', 'pseudo_label', 'seg_image']

project='patchrefiner'
train_cfg=dict(max_epochs=6, val_interval=2, save_checkpoint_interval=6, log_interval=100, train_log_img_interval=100, val_log_img_interval=50, val_type='epoch_base', eval_start=0)

convert_syncbn=True
find_unused_parameters=True

train_dataloader=dict(
    dataset=dict(
        with_pseudo_label=True,))