import os
import cv2
import numpy as np
import torch
import mmcv
from mmcv import Config
from mmrotate.apis import init_detector, inference_detector
from mmrotate.core.visualization import imshow_det_rbboxes
from mmrotate.datasets import build_dataset


def main():
    # 配置路径
    config_file = '../work_dir/rotated_imted_faster_rcnn_vit_small_1x_dota_le90_8h.py'
    checkpoint_file = './latest.pth'
    output_dir = './visualization_results'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置
    cfg = Config.fromfile(config_file)
    
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
    
    # 构建测试数据集以获取ground truth
    test_dataset = build_dataset(cfg.data.test)
    
    # 加载模型
    model = init_detector(cfg, checkpoint_file, device='cuda:0')
    
    # 获取类别名称
    class_names = test_dataset.CLASSES
    
    # 设置颜色
    gt_color = (0, 255, 0)  # 绿色表示ground truth
    pred_color = (255, 0, 0)  # 红色表示预测结果
    
    # 遍历测试数据集
    for i, item in enumerate(test_dataset):
        if i >= 20:  # 仅处理前20张图像，可以根据需要调整
            break
        
        # 获取图像文件路径
        img_path = item['img_metas'].data['filename']
        img_id = os.path.basename(img_path).split('.')[0]
        
        # 读取图像
        img = mmcv.imread(img_path)
        
        # 获取ground truth
        gt_bboxes = item['gt_bboxes'].data.numpy()
        gt_labels = item['gt_labels'].data.numpy()
        
        # 执行推理
        result = inference_detector(model, img)
        
        # 可视化ground truth和预测结果
        # 首先绘制ground truth
        img_with_gt = img.copy()
        if len(gt_bboxes) > 0:
            img_with_gt = imshow_det_rbboxes(
                img_with_gt,
                gt_bboxes,
                gt_labels,
                class_names=class_names,
                score_thr=0.0,  # 显示所有ground truth
                bbox_color=gt_color,
                text_color=gt_color,
                thickness=2,
                show=False
            )
        
        # 然后在相同图像上绘制预测结果
        img_with_both = img_with_gt.copy()
        if isinstance(result, tuple):
            bbox_result, _ = result
        else:
            bbox_result = result
        
        # 分数阈值
        score_thr = 0.3
        
        for class_id, bboxes in enumerate(bbox_result):
            if len(bboxes) > 0:
                # 过滤掉低分数的检测结果
                filtered_bboxes = bboxes[bboxes[:, -1] > score_thr]
                if len(filtered_bboxes) > 0:
                    # 准备用于可视化的数据格式
                    labels = np.full(len(filtered_bboxes), class_id, dtype=np.int32)
                    img_with_both = imshow_det_rbboxes(
                        img_with_both,
                        filtered_bboxes[:, :-1],  # 移除分数列
                        labels,
                        class_names=class_names,
                        score_thr=0.0,  # 已经过滤过分数，所以这里不需要
                        bbox_color=pred_color,
                        text_color=pred_color,
                        thickness=2,
                        show=False
                    )
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{img_id}_result.jpg")
        mmcv.imwrite(img_with_both, output_path)
        print(f"已保存可视化结果: {output_path}")
    
    print(f"可视化完成! 结果保存在: {output_dir}")


if __name__ == '__main__':
    main() 