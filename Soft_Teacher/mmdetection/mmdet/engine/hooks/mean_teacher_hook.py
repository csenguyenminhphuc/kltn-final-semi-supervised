# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class MeanTeacherHook(Hook):
    """Mean Teacher Hook.

    Mean Teacher is an efficient semi-supervised learning method in
    `Mean Teacher <https://arxiv.org/abs/1703.01780>`_.
    This method requires two models with exactly the same structure,
    as the student model and the teacher model, respectively.
    The student model updates the parameters through gradient descent,
    and the teacher model updates the parameters through
    exponential moving average of the student model.
    Compared with the student model, the teacher model
    is smoother and accumulates more knowledge.

    Args:
        momentum (float): The momentum used for updating teacher's parameter.
            Teacher's parameter are updated with the formula:
           `teacher = momentum * teacher + (1 - momentum) * student`.
            This means a HIGH momentum (e.g., 0.999) results in SLOW updates.
            Defaults to 0.001.
        interval (int): Update teacher's parameter every interval iteration.
            Defaults to 1.
        skip_buffers (bool): Whether to skip the model buffers, such as
            batchnorm running stats (running_mean, running_var), it does not
            perform the ema operation. Default to True.
    """

    def __init__(self,
                 momentum: float = 0.001,
                 interval: int = 1,
                 skip_buffer=True) -> None:
        assert 0 < momentum < 1
        self.momentum = momentum
        self.interval = interval
        self.skip_buffers = skip_buffer

    def before_train(self, runner: Runner) -> None:
        """To check that teacher model and student model exist."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')
        # only do it at initial stage
        if runner.iter == 0:
            # CRITICAL FIX: Use momentum=0 to COPY student → teacher (not keep random init!)
            # Formula: teacher = momentum * teacher + (1-momentum) * student
            # With momentum=0: teacher = 0 * teacher + 1 * student = student (COPY)
            # With momentum=1: teacher = 1 * teacher + 0 * student = teacher (KEEP RANDOM - WRONG!)
            self.momentum_update(model, 0)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Update teacher's parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.momentum_update(model, self.momentum)

    def momentum_update(self, model: nn.Module, momentum: float) -> None:
        """Compute the moving average of the parameters using exponential
        moving average.
        
        Formula: teacher = momentum * teacher + (1 - momentum) * student
        
        With momentum=0.999 (recommended):
            teacher = 0.999 * teacher + 0.001 * student
            → Teacher updates SLOWLY (retains 99.9% of old weights)
            → Stable pseudo-labels for semi-supervised learning
        """
        if self.skip_buffers:
            # Use state_dict to handle dynamically created parameters
            student_params = dict(model.student.named_parameters())
            teacher_params = dict(model.teacher.named_parameters())
            
            for name, src_parm in student_params.items():
                if name in teacher_params:
                    dst_parm = teacher_params[name]
                    # Check shape compatibility
                    if src_parm.shape == dst_parm.shape:
                        # CORRECT EMA: teacher = momentum * teacher + (1-momentum) * student
                        dst_parm.data.mul_(momentum).add_(
                            src_parm.data, alpha=(1.0 - momentum))
                    else:
                        # Shape mismatch - likely due to lazy initialization
                        # Copy student param to teacher
                        dst_parm.data.copy_(src_parm.data)
                else:
                    # Teacher missing this parameter - skip or warn
                    # This can happen with lazy-initialized parameters
                    pass
        else:
            for (src_parm,
                 dst_parm) in zip(model.student.state_dict().values(),
                                  model.teacher.state_dict().values()):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:
                    # CORRECT EMA: teacher = momentum * teacher + (1-momentum) * student
                    dst_parm.data.mul_(momentum).add_(
                        src_parm.data, alpha=(1.0 - momentum))
