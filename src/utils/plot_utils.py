"""
可视化工具模块（plot_utils.py）
功能：绘制训练/测试的 loss/accuracy 曲线、混淆矩阵可视化
对应原 solver.py 中的 plot_graphs() 函数
"""

import matplotlib.pyplot as plt
import os


def plot_graphs(solver):
    # Plot graph of loss values
    plt.plot(solver.train_losses, color='b', label='Train')
    plt.plot(solver.test_losses, color='r', label='Test')

    plt.ylabel('Loss', fontsize=18)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize=18)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=15, frameon=False)

    # plt.show()  # Uncomment to display graph
    plt.savefig(os.path.join(solver.args.output_path, 'graph_loss.png'), bbox_inches='tight')
    plt.close('all')

    # Plot graph of accuracies
    plt.plot(solver.train_accuracies, color='b', label='Train')
    plt.plot(solver.test_accuracies, color='r', label='Test')

    plt.ylabel('Accuracy', fontsize=18)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize=18)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=15, frameon=False)

    # plt.show()  # Uncomment to display graph
    plt.savefig(os.path.join(solver.args.output_path, 'graph_accuracy.png'), bbox_inches='tight')
    plt.close('all')