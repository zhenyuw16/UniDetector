import sys
sys.path.append('.')
import mmcv
import json

#a = mmcv.load('pls/lvisv1nov0_norcal.pkl')
#a = mmcv.load('pls/v0train/nooverlapwithgt/vdv0train_norcal_nooverlapwithgt.pkl')
#a = mmcv.load('data/lvis_v0.5/annotations/lvis_v0.5_train.pkl')
#a = mmcv.load('pls/v0val/lvisv0val.pkl')
#a = mmcv.load('pls/v1coco120/lvisv1_coco120_norcal.pkl')
#a = mmcv.load('pls_mask/v1nov0/pcb.pkl')
#a = mmcv.load('oursseesaw_apfix.pkl')
#a = mmcv.load('pls/objcoco05_withgt_nonms.pkl')
a = mmcv.load('vd.pkl')
#a = mmcv.load('ow_pl/unvd_oidobjcoco_lvis10_fastrcnn.pkl')
#a = mmcv.load('ow_pl/wnvd_oidobjcoco_lvis15_multiobjects.pkl')
#a = mmcv.load('ow_pl/semivd_coco.pkl')

print('read done')

from mmdet.datasets import build_dataloader, build_dataset

#cfg = mmcv.Config.fromfile('configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_cocofmt_zs.py')
cfg = mmcv.Config.fromfile('configs/inference/clip_decouple_faster_rcnn_r50_c4_1x_lvis_v0.5_2ndstage.py')
#cfg = mmcv.Config.fromfile('configs/faster_rcnn/clip_faster_rcnn_r50_c4_1x_coco_un_lvis5_zs_pl.py')
#cfg = mmcv.Config.fromfile('configs/fast_rcnn/clip_fast_rcnn_r50_c4_1x_coco_zs.py')
#cfg = mmcv.Config.fromfile('configs/lvis/faster_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v0.5.py')
#cfg = mmcv.Config.fromfile('configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py')
#cfg = mmcv.Config.fromfile('configs/lvis/faster_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py')
#cfg = mmcv.Config.fromfile('configs/seesaw_loss/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py')
#cfg = mmcv.Config.fromfile('configs/otherdataset/faster_rcnn_r50_c4_1x_oid_zs.py')
#cfg = mmcv.Config.fromfile('configs/otherdataset/faster_rcnn_r50_fpn_1x_obj365_coco_zs_together.py')
#cfg = mmcv.Config.fromfile('configs/rdet2/clip_rdet2_cascade_fast_rcnn_r50x4_c4_1x_coco_zs.py')

cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)

dataset.results2json(a, 'b')
