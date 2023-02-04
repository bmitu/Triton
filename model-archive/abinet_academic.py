log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    step=[16, 18],
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=128,
        max_width=128,
        keep_aspect_ratio=False,
        width_downsample_ratio=0.25),
    dict(
        type='RandomWrapper',
        p=0.5,
        transforms=[
            dict(
                type='OneOfWrapper',
                transforms=[
                    dict(type='RandomRotateTextDet', max_angle=15),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAffine',
                        degrees=15,
                        translate=(0.3, 0.3),
                        scale=(0.5, 2.0),
                        shear=(-45, 45)),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomPerspective',
                        distortion_scale=0.5,
                        p=1)
                ])
        ]),
    dict(
        type='RandomWrapper',
        p=0.25,
        transforms=[
            dict(type='PyramidRescale'),
            dict(
                type='Albu',
                transforms=[
                    dict(type='GaussNoise', var_limit=(20, 20), p=0.5),
                    dict(type='MotionBlur', blur_limit=6, p=0.5)
                ])
        ]),
    dict(
        type='RandomWrapper',
        p=0.25,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1)
        ]),
    dict(type='ToTensorOCR'),
    dict(
        type='NormalizeOCR',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio',
            'resize_shape'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=32,
                min_width=128,
                max_width=128,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'resize_shape', 'img_norm_cfg', 'ori_filename'
                ])
        ])
]
num_chars = 37
max_seq_len = 26
label_convertor = dict(
    type='ABIConvertor',
    dict_type='DICT36',
    with_unknown=False,
    with_padding=False,
    lower=True,
    max_seq_len=26)
model = dict(
    type='ABINet',
    backbone=dict(type='ResNetABI'),
    encoder=dict(
        type='ABIVisionModel',
        encoder=dict(
            type='TransformerEncoder',
            n_layers=3,
            n_head=8,
            d_model=512,
            d_inner=2048,
            dropout=0.1,
            max_len=256),
        decoder=dict(
            type='ABIVisionDecoder',
            in_channels=512,
            num_channels=64,
            attn_height=8,
            attn_width=32,
            attn_mode='nearest',
            use_result='feature',
            num_chars=37,
            max_seq_len=26,
            init_cfg=dict(type='Xavier', layer='Conv2d'))),
    decoder=dict(
        type='ABILanguageDecoder',
        d_model=512,
        n_head=8,
        d_inner=2048,
        n_layers=4,
        dropout=0.1,
        detach_tokens=True,
        use_self_attn=False,
        pad_idx=36,
        num_chars=37,
        max_seq_len=26,
        init_cfg=None),
    fuser=dict(
        type='ABIFuser',
        d_model=512,
        num_chars=37,
        init_cfg=None,
        max_seq_len=26),
    loss=dict(
        type='ABILoss',
        enc_weight=1.0,
        dec_weight=1.0,
        fusion_weight=1.0,
        num_classes=37),
    label_convertor=dict(
        type='ABIConvertor',
        dict_type='DICT36',
        with_unknown=False,
        with_padding=False,
        lower=True,
        max_seq_len=26),
    max_seq_len=26,
    iter_size=3)
train_root = 'data/mixture'
train_img_prefix1 = 'data/mixture/Syn90k/mnt/ramdisk/max/90kDICT32px'
train_ann_file1 = 'data/mixture/Syn90k/label.lmdb'
train1 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/Syn90k/mnt/ramdisk/max/90kDICT32px',
    ann_file='data/mixture/Syn90k/label.lmdb',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)
train_img_prefix2 = 'data/mixture/SynthText/synthtext/SynthText_patch_horizontal'
train_ann_file2 = 'data/mixture/SynthText/alphanumeric_label.lmdb'
train2 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/SynthText/synthtext/SynthText_patch_horizontal',
    ann_file='data/mixture/SynthText/alphanumeric_label.lmdb',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)
train_list = [
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/Syn90k/mnt/ramdisk/max/90kDICT32px',
        ann_file='data/mixture/Syn90k/label.lmdb',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='lmdb',
            parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
        pipeline=None,
        test_mode=False),
    dict(
        type='OCRDataset',
        img_prefix=
        'data/mixture/SynthText/synthtext/SynthText_patch_horizontal',
        ann_file='data/mixture/SynthText/alphanumeric_label.lmdb',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='lmdb',
            parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
        pipeline=None,
        test_mode=False)
]
test_root = 'data/mixture'
test_img_prefix1 = 'data/mixture/IIIT5K/'
test_img_prefix2 = 'data/mixture/svt/'
test_img_prefix3 = 'data/mixture/icdar_2013/'
test_img_prefix4 = 'data/mixture/icdar_2015/'
test_img_prefix5 = 'data/mixture/svtp/'
test_img_prefix6 = 'data/mixture/ct80/'
test_ann_file1 = 'data/mixture/IIIT5K/test_label.txt'
test_ann_file2 = 'data/mixture/svt/test_label.txt'
test_ann_file3 = 'data/mixture/icdar_2013/test_label_1015.txt'
test_ann_file4 = 'data/mixture/icdar_2015/test_label.txt'
test_ann_file5 = 'data/mixture/svtp/test_label.txt'
test_ann_file6 = 'data/mixture/ct80/test_label.txt'
test1 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/IIIT5K/',
    ann_file='data/mixture/IIIT5K/test_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test2 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/svt/',
    ann_file='data/mixture/svt/test_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test3 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/icdar_2013/',
    ann_file='data/mixture/icdar_2013/test_label_1015.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test4 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/icdar_2015/',
    ann_file='data/mixture/icdar_2015/test_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test5 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/svtp/',
    ann_file='data/mixture/svtp/test_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test6 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/ct80/',
    ann_file='data/mixture/ct80/test_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test_list = [
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/IIIT5K/',
        ann_file='data/mixture/IIIT5K/test_label.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True),
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/svt/',
        ann_file='data/mixture/svt/test_label.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True),
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/icdar_2013/',
        ann_file='data/mixture/icdar_2013/test_label_1015.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True),
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/icdar_2015/',
        ann_file='data/mixture/icdar_2015/test_label.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True),
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/svtp/',
        ann_file='data/mixture/svtp/test_label.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True),
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/ct80/',
        ann_file='data/mixture/ct80/test_label.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True)
]
data = dict(
    samples_per_gpu=192,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/Syn90k/mnt/ramdisk/max/90kDICT32px',
                ann_file='data/mixture/Syn90k/label.lmdb',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='lmdb',
                    parser=dict(
                        type='LineJsonParser', keys=['filename', 'text'])),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix=
                'data/mixture/SynthText/synthtext/SynthText_patch_horizontal',
                ann_file='data/mixture/SynthText/alphanumeric_label.lmdb',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='lmdb',
                    parser=dict(
                        type='LineJsonParser', keys=['filename', 'text'])),
                pipeline=None,
                test_mode=False)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=32,
                min_width=128,
                max_width=128,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(
                type='RandomWrapper',
                p=0.5,
                transforms=[
                    dict(
                        type='OneOfWrapper',
                        transforms=[
                            dict(type='RandomRotateTextDet', max_angle=15),
                            dict(
                                type='TorchVisionWrapper',
                                op='RandomAffine',
                                degrees=15,
                                translate=(0.3, 0.3),
                                scale=(0.5, 2.0),
                                shear=(-45, 45)),
                            dict(
                                type='TorchVisionWrapper',
                                op='RandomPerspective',
                                distortion_scale=0.5,
                                p=1)
                        ])
                ]),
            dict(
                type='RandomWrapper',
                p=0.25,
                transforms=[
                    dict(type='PyramidRescale'),
                    dict(
                        type='Albu',
                        transforms=[
                            dict(type='GaussNoise', var_limit=(20, 20), p=0.5),
                            dict(type='MotionBlur', blur_limit=6, p=0.5)
                        ])
                ]),
            dict(
                type='RandomWrapper',
                p=0.25,
                transforms=[
                    dict(
                        type='TorchVisionWrapper',
                        op='ColorJitter',
                        brightness=0.5,
                        saturation=0.5,
                        contrast=0.5,
                        hue=0.1)
                ]),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'text',
                    'valid_ratio', 'resize_shape'
                ])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/IIIT5K/',
                ann_file='data/mixture/IIIT5K/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svt/',
                ann_file='data/mixture/svt/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2013/',
                ann_file='data/mixture/icdar_2013/test_label_1015.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2015/',
                ann_file='data/mixture/icdar_2015/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svtp/',
                ann_file='data/mixture/svtp/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/ct80/',
                ann_file='data/mixture/ct80/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=32,
                        min_width=128,
                        max_width=128,
                        keep_aspect_ratio=False,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape',
                            'valid_ratio', 'resize_shape', 'img_norm_cfg',
                            'ori_filename'
                        ])
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/IIIT5K/',
                ann_file='data/mixture/IIIT5K/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svt/',
                ann_file='data/mixture/svt/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2013/',
                ann_file='data/mixture/icdar_2013/test_label_1015.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2015/',
                ann_file='data/mixture/icdar_2015/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svtp/',
                ann_file='data/mixture/svtp/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/ct80/',
                ann_file='data/mixture/ct80/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=32,
                        min_width=128,
                        max_width=128,
                        keep_aspect_ratio=False,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape',
                            'valid_ratio', 'resize_shape', 'img_norm_cfg',
                            'ori_filename'
                        ])
                ])
        ]))
evaluation = dict(interval=1, metric='acc')
