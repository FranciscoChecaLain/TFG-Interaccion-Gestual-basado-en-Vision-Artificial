model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='custom', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=120, in_channels=256))

# Only necessary for defining model structure
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

# You can omit dataset loading here
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type='PoseDataset',
        ann_file='/path/to/annotations.pkl',  # Dummy path, won't be used
        pipeline=test_pipeline,
        split='xsub_val'))

