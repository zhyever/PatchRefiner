


optim_wrapper=dict(
    optimizer=dict(type='AdamW', lr=0.000161, weight_decay=0.01),
    clip_grad=dict(type='norm', max_norm=0.1, norm_type=2), # norm clip
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'student_model.refiner_fine_branch.core': dict(lr_mult=0.1, decay_mult=1.0),
        }))

param_scheduler=dict(
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=1,
    final_div_factor=100,
    pct_start=0.3,
    three_phase=False,)
