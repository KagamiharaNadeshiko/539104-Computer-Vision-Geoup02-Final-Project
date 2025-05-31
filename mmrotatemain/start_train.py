import os
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='启动旋转目标检测模型训练')
    parser.add_argument('--config', default='./configs/train_config.py', help='配置文件路径')
    parser.add_argument('--work-dir', default='./work_dirs/new_training', help='工作目录，用于保存模型和日志')
    parser.add_argument('--gpus', default=1, type=int, help='使用的GPU数量')
    parser.add_argument('--seed', default=None, type=int, help='随机种子')
    parser.add_argument('--deterministic', action='store_true', help='是否使用确定性训练')
    parser.add_argument('--resume-from', default=None, help='从检查点恢复训练')
    parser.add_argument('--no-validate', action='store_true', help='是否在训练期间不执行验证')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 确保工作目录存在
    os.makedirs(args.work_dir, exist_ok=True)
    
    # 构建训练命令
    cmd = [
        'python', 'tools/train.py',
        args.config,
        f'--work-dir={args.work_dir}'
    ]
    
    if args.gpus:
        cmd.append(f'--gpu-ids={",".join([str(i) for i in range(args.gpus)])}')
    
    if args.seed:
        cmd.append(f'--seed={args.seed}')
    
    if args.deterministic:
        cmd.append('--deterministic')
    
    if args.resume_from:
        cmd.append(f'--resume-from={args.resume_from}')
    
    if args.no_validate:
        cmd.append('--no-validate')
    
    # 输出将要执行的命令
    print('执行命令:', ' '.join(cmd))
    
    # 执行训练命令
    subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    main() 