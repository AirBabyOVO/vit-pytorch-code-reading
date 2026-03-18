"""
训练核心模块（trainer.py）
功能：实现模型训练循环、学习率调度、模型保存等逻辑
对应原 solver.py 中的 Solver 类初始化、train() 函数及相关训练辅助逻辑
"""

import os
import torch
import torch.nn as nn
from torch import optim

# 修正导入路径（适配拆分后的目录结构）
from src.data.data_loader import get_loader  # 从新路径导入数据加载
from src.models.vit_scratch import VisionTransformer  # 从零实现版ViT
from src.models.vit_torch import VisionTransformer_pytorch  # PyTorch内置版ViT
# 导入拆分后的评估/可视化模块
from src.training.evaluator import test_dataset, test
from src.utils.plot_utils import plot_graphs


class Solver(object):
    def __init__(self, args):
        self.args = args

        # Get data loaders
        self.train_loader, self.test_loader = get_loader(args)

        # Create object of the Vision Transformer
        if self.args.use_torch_transformer_layers:
            # Uses new Pytorch inbuilt transformer encoder layers
            self.model = VisionTransformer_pytorch(n_channels=self.args.n_channels, embed_dim=self.args.embed_dim,
                                                   n_layers=self.args.n_layers,
                                                   n_attention_heads=self.args.n_attention_heads,
                                                   forward_mul=self.args.forward_mul, image_size=self.args.image_size,
                                                   patch_size=self.args.patch_size, n_classes=self.args.n_classes,
                                                   dropout=self.args.dropout)
        else:
            # model from Scratch
            self.model = VisionTransformer(n_channels=self.args.n_channels, embed_dim=self.args.embed_dim,
                                           n_layers=self.args.n_layers, n_attention_heads=self.args.n_attention_heads,
                                           forward_mul=self.args.forward_mul, image_size=self.args.image_size,
                                           patch_size=self.args.patch_size, n_classes=self.args.n_classes,
                                           dropout=self.args.dropout)

        # Push to GPU
        if self.args.is_cuda:
            self.model = self.model.cuda()

        # Display Vision Transformer
        print('--------Network--------')
        print(self.model)

        # Training parameters stats
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in the model: {n_parameters}")

        # Option to load pretrained model
        if self.args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'ViT_model.pt')))

        # Training loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Arrays to record training progression
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def train(self):
        iters_per_epoch = len(self.train_loader)

        # Define optimizer for training the model
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-3)

        # scheduler for linear warmup of lr and then cosine decay to 1e-5
        linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / self.args.warmup_epochs, end_factor=1.0,
                                                    total_iters=self.args.warmup_epochs - 1, last_epoch=-1,
                                                    verbose=True)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=self.args.epochs - self.args.warmup_epochs, eta_min=1e-5,
                                                         verbose=True)

        # Variable to capture best test accuracy
        best_acc = 0

        # Training loop
        for epoch in range(self.args.epochs):

            # Set model to training mode
            self.model.train()

            # Arrays to record epoch loss and accuracy
            train_epoch_loss = []
            train_epoch_accuracy = []

            # Loop on loader
            for i, (x, y) in enumerate(self.train_loader):

                # Push to GPU
                if self.args.is_cuda:
                    x, y = x.cuda(), y.cuda()

                # Get output logits from the model
                logits = self.model(x)

                # Compute training loss
                loss = self.loss_fn(logits, y)

                # Updating the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Batch metrics
                batch_pred = logits.max(1)[1]
                batch_accuracy = (y == batch_pred).float().mean()
                train_epoch_loss += [loss.item()]
                train_epoch_accuracy += [batch_accuracy.item()]

                # Log training progress
                if i % 50 == 0 or i == (iters_per_epoch - 1):
                    print(
                        f'Ep: {epoch + 1}/{self.args.epochs}\tIt: {i + 1}/{iters_per_epoch}\tbatch_loss: {loss:.4f}\tbatch_accuracy: {batch_accuracy:.2%}')

            # 调用拆分后的 test 函数（评估模块）
            test_acc, test_loss = test(self, train=((epoch + 1) % 25 == 0))  # Test training set every 25 epochs

            # Capture best test accuracy
            best_acc = max(test_acc, best_acc)
            print(f"Best test acc: {best_acc:.2%}\n")

            # Save model
            torch.save(self.model.state_dict(), os.path.join(self.args.model_path, "ViT_model.pt"))

            # Update learning rate using schedulers
            if epoch < self.args.warmup_epochs:
                linear_warmup.step()
            else:
                cos_decay.step()

            # Update training progression metric arrays
            self.train_losses += [sum(train_epoch_loss) / iters_per_epoch]
            self.test_losses += [test_loss]
            self.train_accuracies += [sum(train_epoch_accuracy) / iters_per_epoch]
            self.test_accuracies += [test_acc]

        # 训练结束后调用可视化函数（工具模块）
        plot_graphs(self)