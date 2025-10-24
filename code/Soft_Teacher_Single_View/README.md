# End-to-End Semi-Supervised Object Detection with Soft Teacher

## Citation

```bib
@article{xu2021end,
  title={End-to-End Semi-Supervised Object Detection with Soft Teacher},
  author={Xu, Mengde and Zhang, Zheng and Hu, Han and Wang, Jianfeng and Wang, Lijuan and Wei, Fangyun and Bai, Xiang and Liu, Zicheng},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## Usage

### Requirements

- `python=3.10`
- `Pytorch=2.0.0`
- `cuda=11.7`
- `mmcv=2.0.1`
- `mmengine=0.9.1`

### Install

- Install cuda + pytorch

```shell script

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1

```

- Install mccv

```shell script

pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

```

- Install mmengine

```shell script
cd ./mmengine
pip install -e . -v

```

- Install mmdet

```shell script
cd ./mmdetection
pip install -e . -v

```

### Data Preparation

- generate data set splits:

```shell script
python tools/misc/split_coco.py --data-root 'path your dataset'

```

### Train

```shell script
python ./mmdetection/tools/train.py ./mmdetection/configs/soft_teacher/soft_teacher_custom_20.py
```

### Inference

```shell script
!python ./mmdetection/demo/image_demo.py \
    <img_path>.jpg \
    ./mmdetection/work_dirs/soft_teacher_custom_20/soft_teacher_custom_20.py \
  --weights ./mmdetection/work_dirs/soft_teacher_custom_20/epoch_1.pth \
  --out-dir ./mmdetection/work_dirs/

```
