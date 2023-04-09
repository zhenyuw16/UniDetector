The datasets we use are put in the data/ folder.

### COCO

download the [COCO dataset](https://cocodataset.org/#home) and place them in the following way:

```
coco/
    train2017/
    val2017/
    images/
        train2014/
        val2014/
    annotations/
        instances_valminusminival2014.json
        instances_train2014.json
        instances_train2017.json 
        instances_val2017.json
```

### LVIS

download the [LVIS dataset](https://www.lvisdataset.org/) and place them in the following way:

```
lvis_v0.5/
    train2017/
    val2017/
    annotations/
        lvis_v0.5_train.json
        lvis_v0.5_val.json
```

### Objects365 v2

download the [Objects365 v2 dataset](https://www.objects365.org/overview.html) and place them in the following way:

```
objects365/
    annotations/
        zhiyuan_objv2_train.json
    train/
        images/
            v1/
                patch0/
                ...
                patch15/
            v2/
                patch16/
                ...
                patch49/
```

extract random images for the training subset by:

```
python scripts/extract_random_images.py --seed 1 --percent 3.5 --ann data/objects365/annotations/zhiyuan_objv2_train.json
```

to avoid categories with no images in training, run the following script to statics this:

```
python scripts/get_cat_info.py --seed 1 --ann data/objects365/annotations/zhiyuan_objv2_train.1@3.5.json
```
this will create the file `data/objects365/annotations/zhiyuan_objv2_train.1@3.5_cat_info.json`

### OpenImages

download the [OpenImages dataset](https://storage.googleapis.com/openimages/web/index.html) and place them in the following way:

```
openimages/
    annotations/
        oid_challenge_2019_train_bbox.json
        oid_challenge_2019_val_expanded.json
    0/
    1/
    2/
    ...
```

extract random images for the training subset by:

```
python scripts/extract_random_images.py --ann OID/oid_challenge_2019_train_bbox.json --seed 1 --percent 4.5
```

to avoid categories with no images in training, run the following script to statics this:

```
python scripts/get_cat_info.py --seed 1 --ann data/openimages/oid_challenge_2019_train_bbox.1@4.5.json
```

[Note]: As we only use the subsets for Objects365 and OpenImages, it is not necessary to download the whole dataset. 