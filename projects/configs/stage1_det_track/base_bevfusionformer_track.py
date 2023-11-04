_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]

# Update-2023-06-12: f vr                    
# [Enhance] Update some freezing args of UniAD 
# [Bugfix] Reproduce the from-scratch results of stage1
# 1. Remove loss_past_traj in stage1 training
# 2. Unfreeze neck and BN
# --> Reproduced tracking result: AMOTA 0.393

# fp16 = dict(loss_scale='dynamic')

# Unfreeze neck and BN, the from-scratch results of stage1 could be reproduced
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
grid_size = [512, 512, 1]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=True
)
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
# TODO: 
bev_h_ = 200
bev_w_ = 200
num_query = 900
transformer_num_layers = 6
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)

# NOTE: You can change queue_length from 5 to 3 to save GPU memory, but at risk of performance drop.
queue_length = 5  # each sequence contains `queue_length` frames.

### traj prediction args ###
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True

## occflow setting  
occ_n_future = 4    
occ_n_future_plan = 6
occ_n_future_max = max([occ_n_future, occ_n_future_plan])   

### planning ###
planning_steps = 6
use_col_optim = True

### Occ args ### 
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

# Other settings
train_gt_iou_threshold=0.3

model = dict(
    type='FusionAD',
    freeze_img_backbone=True,
    freeze_img_neck=False,
    freeze_pts_middle_encoder=True,
    freeze_pts_backbone=False,
    freeze_pts_neck=False,
    freeze_bn=False,
    gt_iou_threshold=train_gt_iou_threshold,
    queue_length=queue_length,
    use_grid_mask=True,
    video_test_mode=True,
    num_query=num_query,
    num_classes=10,
    pc_range=point_cloud_range,
    img_backbone=dict(
        type='ResNet',
        with_cp=False,
        # with_cp=True,
        # pretrained='open-mmlab://detectron2/resnet50_caffe',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=4,
        norm_cfg=dict(type='BN2d'),
        relu_before_extra_convs=True),
    # lidar
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        max_num_points=10, voxel_size=voxel_size, max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=False),
        in_channels=[128, 256],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        relu_before_extra_convs=True,
        ),
    # track
    score_thresh=0.4,
    filter_score_thresh=0.35,
    qim_args=dict(
        qim_type="QIMBase",
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1,
    ),  # hyper-param for query dropping mentioned in MOTR
    mem_args=dict(
        memory_bank_type="MemoryBank",
        memory_bank_score_thresh=0.0,
        memory_bank_len=2,
    ),
    loss_cfg=dict(
        type="ClipMatcher",
        num_classes=10,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3DTrack",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_past_traj_weight=0.0,
    ),  # loss cfg for tracking
    pts_bbox_head=dict(
        type='FusionFormerHead',
        num_query=num_query,
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='FusionTransformer',
            decoder=dict(
                type='FusionTransformerDecoder',
                num_layers=transformer_num_layers,
                return_intermediate=True,
                transformerlayers=dict(
                    type='FusionDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='BEVMSDeformableAttention',  # lidar
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='Detr3DCrossAtten',  # camera
                            pc_range=point_cloud_range,
                            num_points=1,
                            embed_dims=_dim_)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 
                                     'cross_attn', 'norm',
                                     'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        # TODO: LearnedPositionalEncoding
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    # TODO: test_cfg?
    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(type="IoUCost", weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        )
    ),
)
dataset_type = "NuScenesFusionE2EDataset"
data_root = "data/nuscenes/"
info_root = "data/infos/"
img_root = ""
file_client_args = dict(backend="disk")
ann_file_train=info_root + f"nuscenes_infos_temporal_train.pkl"  # nuscenes_infos_temporal_val.pkl for debug
ann_file_val=info_root + f"nuscenes_infos_temporal_val.pkl"
ann_file_test=info_root + f"nuscenes_infos_temporal_val.pkl"

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4], ),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, use_dim=[0, 1, 2, 3, 4], ),
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True, file_client_args=file_client_args, img_root=img_root),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D_E2E",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_future_anns=True,  # occ_flow gt
        with_ins_inds_3d=True,  # ins_inds 
        ins_inds_add_1=True,    # ins_inds start from 1
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                    filter_invisible=False),  # NOTE: Currently vis_token is not in pkl 
    dict(type="ObjectRangeFilterTrack", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilterTrack", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "gt_bboxes_3d", "gt_labels_3d", "gt_inds",
            'points', "img",
            "timestamp", "l2g_r_mat", "l2g_t",
            "gt_fut_traj", "gt_fut_traj_mask",
            "gt_past_traj", "gt_past_traj_mask",
            "gt_sdc_bbox", "gt_sdc_label",
            "gt_sdc_fut_traj", "gt_sdc_fut_traj_mask",
            "gt_lane_labels", "gt_lane_bboxes", "gt_lane_masks",
             # Occ gt
            "gt_segmentation", "gt_instance", "gt_centerness", "gt_offset", 
            "gt_flow", "gt_backward_flow", "gt_occ_has_invalid_frame", "gt_occ_img_is_valid",
            # gt future bbox for plan   
            "gt_future_boxes", "gt_future_labels",  
            # planning  
            "sdc_planning", "sdc_planning_mask", "command",
        ],
        meta_keys=['filename', 'ori_shape', 'img_shape', 
                   'lidar2img', 'cam_intrinsic', 'lidar2cam', 'cam2lidar',
                   'pad_shape', 'scale_factor', 
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'prev_idx', 'next_idx', 
                    'pts_filename', 'scene_token', 'can_bus']
    ),
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4], ),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, use_dim=[0, 1, 2, 3, 4], ),
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True,
            file_client_args=file_client_args, img_root=img_root),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type='LoadAnnotations3D_E2E', 
         with_bbox_3d=False,
         with_label_3d=False, 
         with_attr_label=False,
         with_future_anns=True,
         with_ins_inds_3d=False,
         ins_inds_add_1=True, # ins_inds start from 1
         ),
    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                       filter_invisible=False),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="CustomCollect3D", 
                keys=[
                    "points", "img",
                    "timestamp", "l2g_r_mat", "l2g_t",
                    "gt_lane_labels", "gt_lane_bboxes", "gt_lane_masks", 
                    "gt_segmentation", "gt_instance", 
                    "gt_centerness", "gt_offset", 
                    "gt_flow","gt_backward_flow",
                    "gt_occ_has_invalid_frame", "gt_occ_img_is_valid",
                        # planning  
                    "sdc_planning", "sdc_planning_mask", "command",
                ],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 
                    'lidar2img', 'cam_intrinsic', 'lidar2cam', 'cam2lidar',
                    'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'prev_idx', 'next_idx', 
                    'pcd_scale_factor', 'pts_filename', 'scene_token', 'can_bus',
                ],
            ),
        ],
    ),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,  # 
    train=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        # occ
        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        eval_mod=['det', 'track'],  # ['det', 'track', 'map']
        # occ
        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
    ),
    test=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_n_future=occ_n_future_max,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        eval_mod=['det', 'track'],  # ['det', 'track', 'map']
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)
optimizer = dict(
    type="AdamW",
    lr=1e-4,  # 2e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            # 'img_neck': dict(lr_mult=0.1),
            # 'pts_middle_encoder': dict(lr_mult=0.1),
            # 'pts_backbone': dict(lr_mult=0.1),
            # 'pts_neck': dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 12
evaluation = dict(interval=12, pipeline=test_pipeline)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type="WandbLoggerHook")
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = '/home/zhangqiang/work/ad/fusionformer/work_dirs/base_fusion_test/fusion_epoch_12.pth'
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True
