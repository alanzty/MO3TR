total_epochs = 20
load_from = ""
fp_rate = 0.5
dup_rate = 0
fpdb_rate = 0.5
grad = "separate"
bs = 1
num_workers = 0
frame_range = 3
num_ref_imgs = 5
noise = 0
root_work = "/storage/alan/workspace/mmStorage/mot/"
work_dir = root_work + f"mo3tr_temphs_fr{num_ref_imgs}_randseq"

img_scale = (800, 1440)
optimizer = dict(
    type='AdamW',
    lr=0.00002,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[int(0.5 * total_epochs)])
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
model = dict(
    detector=dict(
        type='YOLOX',
        input_size=(800, 1440),
        random_size_range=(18, 32),
        random_size_interval=10,
        backbone=dict(
            type='CSPDarknet',
            deepen_factor=1.33,
            widen_factor=1.25,
            frozen_stages=4),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[320, 640, 1280],
            out_channels=320,
            num_csp_blocks=4,
            freeze=True),
        bbox_head=dict(
            type='Mo3trDetrHead',
            num_query=300,
            num_classes=1,
            in_channels=320,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            transformer=dict(
                type='MO3TRTransformer',
                sa=False,
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=1,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=320,
                            num_levels=3),
                        feedforward_channels=1024,
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=320,
                            feedforward_channels=1024,
                            num_fcs=2,
                            ffn_drop=0.0,
                            act_cfg=dict(type='ReLU', inplace=True)),
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DeformableDetrTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=320,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=320,
                                num_levels=3)
                        ],
                        feedforward_channels=1024,
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=320,
                            feedforward_channels=1024,
                            num_fcs=2,
                            ffn_drop=0.0,
                            act_cfg=dict(type='ReLU', inplace=True)),
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm')))),
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=160,
                normalize=True,
                offset=-0.5),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssignerMO3TR',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        test_cfg=dict(max_per_img=100)),
    type='MO3TRnF',
    tracker=dict(
        type='Mo3trTracker',
        init_track_thr=0.5,
        prop_thr=0.5,
        num_frames_retain=1),
    fp_rate=fp_rate,
    dup_rate=dup_rate,
    noise=noise,
    fpdb_rate=fpdb_rate,
    grad=grad  # use "one" or "separate"
    )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=(800, 1440),
        share_params=True,
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqPad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='MatchInstancesMO3TR', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids'
        ]),
    dict(type='SeqDefaultFormatBundleMO3TR')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 1440),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='ImageToFloatTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data_root = 'MOT17/'
data = dict(
    samples_per_gpu=bs,
    workers_per_gpu=num_workers,
    persistent_workers=False,
    val=dict(
        type='MO3TRDataset',
        ann_file=data_root + 'annotations/half-val-SDP_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline,
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)),
    test=dict(
        type='MO3TRDataset',
        ann_file=data_root + 'annotations/half-val-SDP_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline,
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)),
    train=dict(
        type='MO3TRDataset',
        visibility_thr=-1,
        ann_file=data_root + 'annotations/half-train-SDP_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=dict(
            num_ref_imgs=num_ref_imgs,
            frame_range=frame_range,
            filter_key_img=True,
            method='uniform'),
        pipeline=train_pipeline))
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='SyncNormHook', num_last_epochs=15, interval=5, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = ""
workflow = [('train', 1)]
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
gpu_ids = range(0, 1)
