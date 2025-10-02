from detectron2.data.datasets import register_coco_instances

register_coco_instances("TRAIN_DATASET", {}, "/home/coder/trong/KLTN_SEMI/data/train/_annotations.coco.json", "/home/coder/trong/KLTN_SEMI/data/train")
register_coco_instances("VAL_DATASET", {}, "/home/coder/trong/KLTN_SEMI/data/valid/_annotations.coco.json", "/home/coder/trong/KLTN_SEMI/data/valid")