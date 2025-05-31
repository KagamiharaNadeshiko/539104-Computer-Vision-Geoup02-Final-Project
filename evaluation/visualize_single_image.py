import os
import cv2
import numpy as np
import torch
import mmcv
import argparse
from mmcv import Config
from mmrotate.apis import init_detector, inference_detector
from mmrotate.core.visualization import imshow_det_rbboxes


def parse_args():
    parser = argparse.ArgumentParser(description='可视化单张图像的预测结果')
    parser.add_argument('--config', default='../work_dir/rotated_imted_faster_rcnn_vit_small_1x_dota_le90_8h.py',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', default='./latest.pth', help='模型文件路径')
    parser.add_argument('--img', required=True, help='输入图像路径')
    parser.add_argument('--out', default=None, help='输出图像路径')
    parser.add_argument('--score-thr', type=float, default=0.3, help='检测结果分数阈值')
    parser.add_argument('--device', default='cuda:0', help='设备，可选 cuda:0 或 cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 加载模型
    model = init_detector(cfg, args.checkpoint, device=args.device)
    
    # 获取类别名称
    if hasattr(model, 'CLASSES'):
        class_names = model.CLASSES
    else:
        from mmrotate.datasets import DOTADataset
        class_names = DOTADataset.CLASSES
    
    # 检测阈值
    score_thr = args.score_thr
    
    # 读取图像
    img = mmcv.imread(args.img)
    
    # 执行推理
    result = inference_detector(model, img)
    
    # 处理结果格式
    if isinstance(result, tuple):
        bbox_result, _ = result
    else:
        bbox_result = result
    
    # 可视化结果
    out_file = args.out if args.out else 'output_result.jpg'
    imshow_det_rbboxes(
        img,
        bbox_result,
        class_names=class_names,
        score_thr=score_thr,
        out_file=out_file
    )
    
    print(f'检测结果已保存到 {out_file}')


if __name__ == '__main__':
    main() 