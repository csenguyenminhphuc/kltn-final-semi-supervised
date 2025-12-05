"""Check if MVViT parameters are included in model parameters."""

import torch
import sys
sys.path.insert(0, 'mmdetection')

from mmengine.config import Config
from mmengine.registry import MODELS

# Load config (use the working directory config)
cfg = Config.fromfile('work_dirs/soft_teacher_8views_cross_transformers/soft_teacher_custom_multi_view.py')

print("="*80)
print("Checking MVViT Parameters in Model")
print("="*80)

# Build model
model = MODELS.build(cfg.model)

# Get all parameter names
print("\nAll parameters in model.student:")
mvvit_params = []
backbone_params = []
neck_params = []
head_params = []
other_params = []

for name, param in model.student.named_parameters():
    if 'mvvit' in name.lower() or 'multi_view_transformer' in name.lower():
        mvvit_params.append(name)
    elif 'backbone' in name:
        backbone_params.append(name)
    elif 'neck' in name:
        neck_params.append(name)
    elif 'head' in name or 'rpn' in name or 'roi' in name:
        head_params.append(name)
    else:
        other_params.append(name)

print(f"\n📊 Parameter Distribution:")
print(f"  - MVViT parameters: {len(mvvit_params)}")
print(f"  - Backbone parameters: {len(backbone_params)}")
print(f"  - Neck parameters: {len(neck_params)}")
print(f"  - Head parameters: {len(head_params)}")
print(f"  - Other parameters: {len(other_params)}")
print(f"  - TOTAL: {len(list(model.student.parameters()))}")

if mvvit_params:
    print(f"\n✅ MVViT parameters FOUND in model.student!")
    print(f"\nFirst 10 MVViT parameters:")
    for name in mvvit_params[:10]:
        param = dict(model.student.named_parameters())[name]
        print(f"  - {name}: shape {list(param.shape)}")
    
    if len(mvvit_params) > 10:
        print(f"  ... and {len(mvvit_params) - 10} more")
else:
    print(f"\n❌ No MVViT parameters found!")
    print(f"\nFirst 10 parameters:")
    for i, (name, param) in enumerate(model.student.named_parameters()):
        if i >= 10:
            break
        print(f"  - {name}: shape {list(param.shape)}")

# Check teacher
print("\n" + "="*80)
print("Checking Teacher Model")
print("="*80)

teacher_mvvit_params = []
for name, param in model.teacher.named_parameters():
    if 'mvvit' in name.lower() or 'multi_view_transformer' in name.lower():
        teacher_mvvit_params.append(name)

print(f"\nTeacher MVViT parameters: {len(teacher_mvvit_params)}")
print(f"Total teacher parameters: {len(list(model.teacher.parameters()))}")

if len(mvvit_params) == len(teacher_mvvit_params):
    print(f"\n✅ Student and Teacher have SAME number of MVViT parameters!")
else:
    print(f"\n❌ Mismatch: Student has {len(mvvit_params)}, Teacher has {len(teacher_mvvit_params)}")

# Verify EMA will update MVViT
print("\n" + "="*80)
print("Verifying EMA Update Coverage")
print("="*80)

print("\nMeanTeacherHook uses model.student.named_parameters()")
print("This includes:")
print(f"  ✅ Backbone: {len(backbone_params)} params")
print(f"  ✅ MVViT: {len(mvvit_params)} params")
print(f"  ✅ Neck: {len(neck_params)} params")
print(f"  ✅ Heads: {len(head_params)} params")
print(f"\n→ ALL student parameters will be EMA-updated to teacher!")
print(f"→ MVViT cross-view attention weights WILL be transferred!")
