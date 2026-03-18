"""
评估模块（evaluator.py）
功能：实现模型测试、验证、混淆矩阵计算、准确率/损失统计
对应原 solver.py 中的 test_dataset()、test() 函数
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


def test_dataset(solver, loader):
    # Set Vision Transformer to evaluation mode
    solver.model.eval()

    # Arrays to record all labels and logits
    all_labels = []
    all_logits = []

    # Testing loop
    for (x, y) in loader:
        if solver.args.is_cuda:
            x = x.cuda()

        # Avoid capturing gradients in evaluation time for faster speed
        with torch.no_grad():
            logits = solver.model(x)

        all_labels.append(y)
        all_logits.append(logits.cpu())

    # Convert all captured variables to torch
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)
    all_pred = all_logits.max(1)[1]

    # Compute loss, accuracy and confusion matrix
    loss = solver.loss_fn(all_logits, all_labels).item()
    acc = accuracy_score(y_true=all_labels, y_pred=all_pred)
    cm = confusion_matrix(y_true=all_labels, y_pred=all_pred, labels=range(solver.args.n_classes))

    return acc, cm, loss


def test(solver, train=True):
    if train:
        # Test using train loader
        acc, cm, loss = test_dataset(solver, solver.train_loader)
        print(f"Train acc: {acc:.2%}\tTrain loss: {loss:.4f}\nTrain Confusion Matrix:")
        print(cm)

    # Test using test loader
    acc, cm, loss = test_dataset(solver, solver.test_loader)
    print(f"Test acc: {acc:.2%}\tTest loss: {loss:.4f}\nTest Confusion Matrix:")
    print(cm)

    return acc, loss