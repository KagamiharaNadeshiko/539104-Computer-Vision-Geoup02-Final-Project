
import os
import argparse
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmrotate.datasets import build_dataloader, build_dataset
from mmrotate.models import build_detector
from mmrotate.apis.test import single_gpu_test
from mmrotate.core.evaluation import eval_rbbox_map


def parse_args():
    parser = argparse.ArgumentParser(description='评估旋转目标检测模型性能')
    parser.add_argument('--config', default='../work_dir/rotated_imted_faster_rcnn_vit_small_1x_dota_le90_8h.py',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', default='./latest.pth', help='模型文件路径')
    parser.add_argument('--out', default='eval_results.pkl', help='输出结果文件路径')
    parser.add_argument('--device', default='cuda:0', help='设备，可选 cuda:0 或 cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 修复数据路径（如果需要）
    if 'data' in cfg and 'test' in cfg.data:
        # 检查并替换路径，如有必要
        test_img_prefix = cfg.data.test.img_prefix
        test_ann_file = cfg.data.test.ann_file
        if not os.path.exists(test_img_prefix):
            print(f"警告: 测试图像路径不存在: {test_img_prefix}")
            # 尝试调整为当前环境路径
            if test_img_prefix.startswith('E:'):
                new_path = test_img_prefix.replace('E:', 'D:')
                if os.path.exists(new_path):
                    cfg.data.test.img_prefix = new_path
                    print(f"已将测试图像路径调整为: {new_path}")
        
        if not os.path.exists(test_ann_file):
            print(f"警告: 测试标注文件路径不存在: {test_ann_file}")
            # 尝试调整为当前环境路径
            if test_ann_file.startswith('E:'):
                new_path = test_ann_file.replace('E:', 'D:')
                if os.path.exists(new_path):
                    cfg.data.test.ann_file = new_path
                    print(f"已将测试标注文件路径调整为: {new_path}")
    
    # 设置GPU设备
    device = torch.device(args.device)
    
    # 构建数据集
    test_dataset = build_dataset(cfg.data.test)
    
    # 构建数据加载器
    data_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False
    )
    
    # 构建模型
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # 将模型放到指定设备
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    # 执行测试
    print('开始评估...')
    results = single_gpu_test(model, data_loader)
    
    # 评估
    print('计算评估指标...')
    eval_results = test_dataset.evaluate(results)
    
    # 打印评估结果
    print('\n评估结果:')
    for metric, value in eval_results.items():
        print(f'{metric}: {value}')
    
    print(f'评估完成!')


if __name__ == '__main__':
    main() 