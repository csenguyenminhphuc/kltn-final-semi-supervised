"""Script to evaluate checkpoint and compare with training results."""
import argparse
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--work-dir',
        help='the directory to save evaluation results',
        default=None)
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved',
        default=None)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Override work_dir if specified
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # Use config filename as default work_dir if not specified
        cfg.work_dir = './work_dirs/eval_results'
    
    # Load checkpoint
    cfg.load_from = args.checkpoint
    
    # Build the runner
    runner = Runner.from_cfg(cfg)
    
    # Start evaluation
    print(f"\n{'='*80}")
    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Work dir: {cfg.work_dir}")
    print(f"{'='*80}\n")
    
    # Run evaluation
    metrics = runner.test()
    
    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS:")
    print(f"{'='*80}")
    
    # Teacher results
    print("\n[TEACHER MODEL]")
    teacher_keys = [k for k in metrics.keys() if k.startswith('teacher/')]
    for key in sorted(teacher_keys):
        print(f"  {key}: {metrics[key]:.4f}")
    
    # Student results  
    print("\n[STUDENT MODEL]")
    student_keys = [k for k in metrics.keys() if k.startswith('student/')]
    for key in sorted(student_keys):
        print(f"  {key}: {metrics[key]:.4f}")
    
    # Compare with training results
    print(f"\n{'='*80}")
    print("COMPARISON WITH TRAINING:")
    print(f"{'='*80}")
    
    # Expected training results (from log)
    expected_teacher_map50 = 0.1290
    expected_teacher_map = 0.0410
    
    actual_teacher_map50 = metrics.get('teacher/coco/bbox_mAP_50', -1)
    actual_teacher_map = metrics.get('teacher/coco/bbox_mAP', -1)
    
    print(f"\nTeacher bbox_mAP_50:")
    print(f"  Training:   {expected_teacher_map50:.4f} (12.90%)")
    print(f"  Re-eval:    {actual_teacher_map50:.4f} ({actual_teacher_map50*100:.2f}%)")
    print(f"  Difference: {abs(actual_teacher_map50 - expected_teacher_map50):.4f}")
    print(f"  Match:      {'✅ YES' if abs(actual_teacher_map50 - expected_teacher_map50) < 0.001 else '❌ NO'}")
    
    print(f"\nTeacher bbox_mAP:")
    print(f"  Training:   {expected_teacher_map:.4f} (4.10%)")
    print(f"  Re-eval:    {actual_teacher_map:.4f} ({actual_teacher_map*100:.2f}%)")
    print(f"  Difference: {abs(actual_teacher_map - expected_teacher_map):.4f}")
    print(f"  Match:      {'✅ YES' if abs(actual_teacher_map - expected_teacher_map) < 0.001 else '❌ NO'}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
