"""
ViT 项目入口文件（main.py）
功能：解析命令行参数，初始化 Solver 并启动训练
"""
import sys
import os
import argparse

# 添加项目根路径，确保能导入 src 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入拆分后的训练模块
from src.training.trainer import Solver


def main():
    # 解析命令行参数（保留原逻辑）
    parser = argparse.ArgumentParser()
    # 粘贴原 main.py 中的参数定义（如 --dataset、--epochs 等）
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--epochs', type=int, default=2)
    # ... 其他参数（复制原 main.py 的 argparse 部分）

    args = parser.parse_args()

    # 初始化训练器并启动训练
    solver = Solver(args)
    solver.train()
    # 训练结束后可视化
    solver.plot_graphs()  # 若已在 trainer.py 中调用，可注释


if __name__ == '__main__':
    main()