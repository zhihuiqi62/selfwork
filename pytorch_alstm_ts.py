# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 导入未来版本的除法特性，确保除法行为一致
from __future__ import division
# 导入未来版本的print函数特性，确保print行为一致
from __future__ import print_function

# 导入数值计算库
import numpy as np
# 导入数据分析库
import pandas as pd
# 导入类型提示相关模块
from typing import Text, Union
# 导入深拷贝模块
import copy
# 导入路径处理工具
from ...utils import get_or_create_path
# 导入日志模块
from ...log import get_module_logger

# 导入PyTorch相关模块
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入模型参数计数工具
from .pytorch_utils import count_parameters
# 导入基础模型类
from ...model.base import Model
# 导入数据集类
from ...data.dataset import DatasetH
# 导入数据处理器
from ...data.dataset.handler import DataHandlerLP
# 导入数据集合并工具
from ...model.utils import ConcatDataset
# 导入权重处理工具
from ...data.dataset.weight import Reweighter


class ALSTM(Model):
    """ALSTM (Attention-based Long Short-Term Memory) 模型
    
    这是一个基于注意力机制的LSTM模型，用于时间序列预测任务。
    继承自基础Model类，实现了训练、评估和预测功能。
    
    Parameters
    ----------
    d_feat : int
        每个时间步的输入特征维度，默认为6
    hidden_size : int
        LSTM隐藏层大小，默认为64
    num_layers : int
        LSTM层数，默认为2
    dropout : float
        Dropout概率，默认为0.0(不启用)
    n_epochs : int
        训练的最大epoch数，默认为200
    lr : float
        学习率，默认为0.001
    metric : str
        用于早停的评估指标，默认为空(使用loss)
    batch_size : int
        批量大小，默认为2000
    early_stop : int
        早停轮数，验证集性能连续不提升的epoch数，默认为20
    loss : str
        损失函数类型，目前支持'mse'，默认为'mse'
    optimizer : str
        优化器类型，支持'adam'和'gd'，默认为'adam'
    n_jobs : int
        数据加载时的并行工作数，默认为10
    GPU : int
        使用的GPU ID，默认为0(使用第一个GPU)
    seed : int, optional
        随机种子，用于复现结果，默认为None(不设置)
    **kwargs
        其他关键字参数
    """

    def __init__(
        self,
        d_feat=6,  # 输入特征维度
        hidden_size=64,  # 隐藏层大小
        num_layers=2,  # LSTM层数
        dropout=0.0,  # dropout概率
        n_epochs=200,  # 训练epoch数
        lr=0.001,  # 学习率
        metric="",  # 评估指标
        batch_size=2000,  # 批量大小
        early_stop=20,  # 早停轮数
        loss="mse",  # 损失函数类型
        optimizer="adam",  # 优化器类型
        n_jobs=10,  # 并行工作数
        GPU=0,  # GPU ID
        seed=None,  # 随机种子
        **kwargs,  # 其他参数
    ):
        # 设置日志记录器
        self.logger = get_module_logger("ALSTM")
        self.logger.info("ALSTM pytorch version...")

        # 设置模型超参数
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()  # 优化器名称转为小写
        self.loss = loss
        # 设置设备(CPU或GPU)
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed

        # 记录模型参数设置
        self.logger.info(
            "ALSTM parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
            )
        )

        # 设置随机种子(如果提供)
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # 初始化ALSTM模型
        self.ALSTM_model = ALSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        # 记录模型结构和参数数量
        self.logger.info("model:\n{:}".format(self.ALSTM_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.ALSTM_model)))

        # 初始化优化器
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.ALSTM_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.ALSTM_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        # 标记模型尚未训练
        self.fitted = False
        # 将模型移动到指定设备
        self.ALSTM_model.to(self.device)

    @property
    def use_gpu(self):
        """检查是否使用GPU
        
        Returns
        -------
        bool
            如果设备不是CPU则返回True
        """
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        """计算加权均方误差损失
        
        Parameters
        ----------
        pred : torch.Tensor
            模型预测值
        label : torch.Tensor
            真实标签
        weight : torch.Tensor
            样本权重
            
        Returns
        -------
        torch.Tensor
            加权MSE损失
        """
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight=None):
        """计算损失函数
        
        Parameters
        ----------
        pred : torch.Tensor
            模型预测值
        label : torch.Tensor
            真实标签
        weight : torch.Tensor, optional
            样本权重，默认为None(等权重)
            
        Returns
        -------
        torch.Tensor
            计算得到的损失值
            
        Raises
        ------
        ValueError
            当损失函数类型未知时抛出异常
        """
        # 创建掩码，过滤NaN值
        mask = ~torch.isnan(label)

        # 如果没有提供权重，则使用等权重
        if weight is None:
            weight = torch.ones_like(label)

        # 根据配置选择损失函数
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        """计算评估指标
        
        Parameters
        ----------
        pred : torch.Tensor
            模型预测值
        label : torch.Tensor
            真实标签
            
        Returns
        -------
        torch.Tensor
            计算得到的评估指标值
            
        Raises
        ------
        ValueError
            当评估指标类型未知时抛出异常
        """
        # 创建掩码，过滤无穷大值
        mask = torch.isfinite(label)

        # 根据配置选择评估指标
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        elif self.metric == "mse":
            mask = ~torch.isnan(label)
            weight = torch.ones_like(label)
            return -self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):
        """训练一个epoch
        
        Parameters
        ----------
        data_loader : DataLoader
            训练数据加载器
        """
        # 设置模型为训练模式
        self.ALSTM_model.train()

        # 遍历数据加载器中的每个批次
        for data, weight in data_loader:
            # 提取特征和标签，并移动到指定设备
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # 前向传播
            pred = self.ALSTM_model(feature.float())
            # 计算损失
            loss = self.loss_fn(pred, label, weight.to(self.device))

            # 反向传播和优化
            self.train_optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_value_(self.ALSTM_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        """测试一个epoch
        
        Parameters
        ----------
        data_loader : DataLoader
            测试数据加载器
            
        Returns
        -------
        tuple
            平均损失和平均评估指标
        """
        # 设置模型为评估模式
        self.ALSTM_model.eval()

        scores = []  # 存储每个批次的评估指标
        losses = []   # 存储每个批次的损失值

        # 遍历数据加载器中的每个批次
        for data, weight in data_loader:
            # 提取特征和标签，并移动到指定设备
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # 禁用梯度计算
            with torch.no_grad():
                # 前向传播
                pred = self.ALSTM_model(feature.float())
                # 计算损失
                loss = self.loss_fn(pred, label, weight.to(self.device))
                losses.append(loss.item())

                # 计算评估指标
                score = self.metric_fn(pred, label)
                scores.append(score.item())

        # 返回平均损失和平均评估指标
        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,  # 数据集对象
        evals_result=dict(),  # 评估结果字典
        save_path=None,  # 模型保存路径
        reweighter=None,  # 权重处理器
    ):
        """训练模型
        
        Parameters
        ----------
        dataset : DatasetH
            训练数据集
        evals_result : dict, optional
            用于存储训练过程中的评估结果，默认为空字典
        save_path : str, optional
            模型保存路径，默认为None(不保存)
        reweighter : Reweighter, optional
            样本权重处理器，默认为None(不使用)
            
        Raises
        ------
        ValueError
            当数据集为空或权重处理器类型不支持时抛出异常
        """
        # 准备训练和验证数据
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # 检查数据是否为空
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        # 配置NaN填充方式
        dl_train.config(fillna_type="ffill+bfill")  # 前向后向填充
        dl_valid.config(fillna_type="ffill+bfill")  # 前向后向填充

        # 处理样本权重
        if reweighter is None:
            wl_train = np.ones(len(dl_train))  # 等权重
            wl_valid = np.ones(len(dl_valid))  # 等权重
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)  # 计算训练集权重
            wl_valid = reweighter.reweight(dl_valid)  # 计算验证集权重
        else:
            raise ValueError("Unsupported reweighter type.")

        # 创建数据加载器
        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),  # 合并数据和权重
            batch_size=self.batch_size,
            shuffle=True,  # 训练集打乱
            num_workers=self.n_jobs,
            drop_last=True,  # 丢弃最后不完整的批次
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),  # 合并数据和权重
            batch_size=self.batch_size,
            shuffle=False,  # 验证集不打乱
            num_workers=self.n_jobs,
            drop_last=True,  # 丢弃最后不完整的批次
        )

        # 创建保存路径(如果提供)
        save_path = get_or_create_path(save_path)

        # 初始化训练变量
        stop_steps = 0  # 早停计数器
        train_loss = 0  # 训练损失
        best_score = -np.inf  # 最佳评估指标
        best_epoch = 0  # 最佳epoch
        evals_result["train"] = []  # 训练集评估结果
        evals_result["valid"] = []  # 验证集评估结果

        # 开始训练
        self.logger.info("training...")
        self.fitted = True  # 标记模型已训练

        # 训练循环
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            # 训练一个epoch
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            # 评估训练集和验证集
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            # 记录评估结果
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            # 早停逻辑
            if val_score > best_score:
                best_score = val_score  # 更新最佳分数
                stop_steps = 0  # 重置早停计数器
                best_epoch = step  # 记录最佳epoch
                best_param = copy.deepcopy(self.ALSTM_model.state_dict())  # 保存最佳参数
            else:
                stop_steps += 1  # 增加早停计数器
                if stop_steps >= self.early_stop:  # 达到早停条件
                    self.logger.info("early stop")
                    break  # 停止训练

        # 训练结束处理
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        # 加载最佳参数
        self.ALSTM_model.load_state_dict(best_param)
        # 保存模型(如果提供路径)
        torch.save(best_param, save_path)

        # 清理GPU缓存(如果使用GPU)
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        """使用训练好的模型进行预测
        
        Parameters
        ----------
        dataset : DatasetH
            预测数据集
        segment : str or slice, optional
            数据集分段，默认为'test'
            
        Returns
        -------
        pd.Series
            预测结果，索引与数据集一致
            
        Raises
        ------
        ValueError
            当模型未训练时抛出异常
        """
        # 检查模型是否已训练
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        # 准备测试数据
        dl_test = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        # 配置NaN填充方式
        dl_test.config(fillna_type="ffill+bfill")
        # 创建测试数据加载器
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        # 设置模型为评估模式
        self.ALSTM_model.eval()
        preds = []  # 存储预测结果

        # 遍历测试数据
        for data in test_loader:
            # 提取特征并移动到指定设备
            feature = data[:, :, 0:-1].to(self.device)

            # 禁用梯度计算
            with torch.no_grad():
                # 前向传播并获取预测结果
                pred = self.ALSTM_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)  # 保存当前批次的预测结果

        # 合并所有批次的预测结果，并返回为pd.Series
        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class ALSTMModel(nn.Module):
    """ALSTM模型的核心网络结构
    
    实现了一个基于注意力机制的LSTM/GRU网络。
    
    Parameters
    ----------
    d_feat : int, optional
        输入特征维度，默认为6
    hidden_size : int, optional
        隐藏层大小，默认为64
    num_layers : int, optional
        RNN层数，默认为2
    dropout : float, optional
        Dropout概率，默认为0.0
    rnn_type : str, optional
        RNN类型，支持'GRU'或'LSTM'，默认为'GRU'
    """

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        # 设置模型参数
        self.hid_size = hidden_size  # 隐藏层大小
        self.input_size = d_feat  # 输入特征维度
        self.dropout = dropout  # dropout概率
        self.rnn_type = rnn_type  # RNN类型
        self.rnn_layer = num_layers  # RNN层数
        # 构建模型
        self._build_model()

    def _build_model(self):
        """构建模型网络结构"""
        try:
            # 根据rnn_type获取对应的RNN类
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        
        # 构建输入网络
        self.net = nn.Sequential()
        # 输入全连接层
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        # Tanh激活函数
        self.net.add_module("act", nn.Tanh())
        
        # 构建RNN层
        self.rnn = klass(
            input_size=self.hid_size,  # 输入大小
            hidden_size=self.hid_size,  # 隐藏层大小
            num_layers=self.rnn_layer,  # 层数
            batch_first=True,  # 输入格式为(batch, seq, feature)
            dropout=self.dropout,  # dropout概率
        )
        
        # 输出全连接层
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        
        # 构建注意力网络
        self.att_net = nn.Sequential()
        # 注意力输入全连接层
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        # 注意力dropout层
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        # 注意力激活函数
        self.att_net.add_module("att_act", nn.Tanh())
        # 注意力输出全连接层
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        # 注意力softmax归一化
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))


        # 理解self.rnn原生，还是需要，pytorch的源码

    def forward(self, inputs):
        """前向传播
        
        Parameters
        ----------
        inputs : torch.Tensor
            输入张量，形状为(batch_size, seq_len, input_size)
            
        Returns
        -------
        torch.Tensor
            输出张量，形状为(batch_size,)
        """
        # 通过输入网络处理输入
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        # 计算注意力分数
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        # 应用注意力权重
        out_att = torch.mul(rnn_out, attention_score)
        # 求和得到注意力加权输出
        out_att = torch.sum(out_att, dim=1)
        # 拼接RNN最后一步输出和注意力加权输出，并通过输出层
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        # 返回展平后的输出
        return out[..., 0]
    


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 导入必要的库
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union  # 用于类型提示
import copy  # 用于深拷贝对象
from ...utils import get_or_create_path  # 工具函数：获取或创建路径
from ...log import get_module_logger  # 日志工具

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader  # 数据加载器

from .pytorch_utils import count_parameters  # 工具函数：计算模型参数数量
from ...model.base import Model  # 基础模型类
from ...data.dataset import DatasetH  # 数据集类
from ...data.dataset.handler import DataHandlerLP  # 数据处理句柄
from ...model.utils import ConcatDataset  # 工具类：拼接数据集和权重
from ...data.dataset.weight import Reweighter  # 权重调整器


class ALSTM(Model):
    """
    ALSTM模型（Attention-based LSTM）
    
    这是一个基于注意力机制的LSTM模型，用于时序数据预测任务。
    模型通过注意力机制对LSTM的输出进行加权，重点关注对预测更重要的时间步。

    参数
    ----------
    d_feat : int
        每个时间步的输入特征维度
    metric: str
        早停机制使用的评估指标
    optimizer : str
        优化器名称
    GPU : int
        用于训练的GPU ID
    hidden_size : int
        LSTM隐藏层大小
    num_layers : int
        LSTM层数
    dropout : float
        dropout比率，用于防止过拟合
    n_epochs : int
        训练轮数
    lr : float
        学习率
    batch_size : int
        批处理大小
    early_stop : int
        早停轮数，连续这么多轮性能没有提升则停止训练
    loss : str
        损失函数类型
    n_jobs : int
        数据加载的并行进程数
    seed : int
        随机种子，用于复现实验结果
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,** kwargs,
    ):
        # 设置日志器
        self.logger = get_module_logger("ALSTM")
        self.logger.info("ALSTM pytorch version...")

        # 设置超参数
        self.d_feat = d_feat  # 特征维度
        self.hidden_size = hidden_size  # 隐藏层大小
        self.num_layers = num_layers  # LSTM层数
        self.dropout = dropout  # dropout比率
        self.n_epochs = n_epochs  # 训练轮数
        self.lr = lr  # 学习率
        self.metric = metric  # 评估指标
        self.batch_size = batch_size  # 批大小
        self.early_stop = early_stop  # 早停参数
        self.optimizer = optimizer.lower()  # 优化器名称（转为小写）
        self.loss = loss  # 损失函数类型
        # 设备配置：优先使用GPU，否则使用CPU
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs  # 并行加载数据的进程数
        self.seed = seed  # 随机种子

        # 记录模型参数到日志
        self.logger.info(
            "ALSTM parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
            )
        )

        # 设置随机种子，保证实验可复现
        if self.seed is not None:
            np.random.seed(self.seed)  # 设置numpy随机种子
            torch.manual_seed(self.seed)  # 设置torch随机种子

        # 初始化ALSTM模型
        self.ALSTM_model = ALSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.logger.info("model:\n{:}".format(self.ALSTM_model))  # 打印模型结构
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.ALSTM_model)))  # 打印模型大小

        # 根据选择的优化器类型初始化优化器
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.ALSTM_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.ALSTM_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False  # 标记模型是否已训练
        self.ALSTM_model.to(self.device)  # 将模型移动到指定设备（GPU/CPU）

    @property
    def use_gpu(self):
        """判断是否使用GPU"""
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        """
        计算加权均方误差
        
        参数:
            pred: 预测值
            label: 真实标签
            weight: 每个样本的权重
        
        返回:
            加权MSE损失
        """
        loss = weight * (pred - label) **2  # 计算加权平方误差
        return torch.mean(loss)  # 返回平均损失

    def loss_fn(self, pred, label, weight=None):
        """
        损失函数计算
        
        参数:
            pred: 预测值
            label: 真实标签
            weight: 每个样本的权重，默认为None
        
        返回:
            计算得到的损失值
        """
        mask = ~torch.isnan(label)  # 创建掩码，排除NaN值

        # 如果未提供权重，则使用全1权重
        if weight is None:
            weight = torch.ones_like(label)

        # 根据配置选择损失函数
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])  # 应用掩码并计算MSE

        raise ValueError("unknown loss `%s`" % self.loss)  # 抛出未知损失函数的错误

    def metric_fn(self, pred, label):
        """
        评估指标计算
        
        参数:
            pred: 预测值
            label: 真实标签
        
        返回:
            计算得到的评估指标值
        """
        mask = torch.isfinite(label)  # 创建掩码，排除非有限值（inf, nan等）

        # 根据配置选择评估指标
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])  # 负损失值（用于最大化指标）
        elif self.metric == "mse":
            mask = ~torch.isnan(label)
            weight = torch.ones_like(label)
            return -self.mse(pred[mask], label[mask], weight[mask])  # 负MSE

        raise ValueError("unknown metric `%s`" % self.metric)  # 抛出未知评估指标的错误

    def train_epoch(self, data_loader):
        """
        训练一个轮次
        
        参数:
            data_loader: 训练数据加载器
        """
        self.ALSTM_model.train()  # 设置模型为训练模式

        # 遍历数据加载器中的批次
        for data, weight in data_loader:
            # 提取特征：取所有时间步的所有特征（排除最后一列）
            feature = data[:, :, 0:-1].to(self.device)
            # 提取标签：取最后一个时间步的最后一列
            label = data[:, -1, -1].to(self.device)

            # 模型前向传播，得到预测值
            pred = self.ALSTM_model(feature.float())
            # 计算损失
            loss = self.loss_fn(pred, label, weight.to(self.device))

            # 反向传播和参数更新
            self.train_optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            torch.nn.utils.clip_grad_value_(self.ALSTM_model.parameters(), 3.0)  # 梯度裁剪，防止梯度爆炸
            self.train_optimizer.step()  # 更新参数

    def test_epoch(self, data_loader):
        """
        测试/验证一个轮次
        
        参数:
            data_loader: 测试/验证数据加载器
        
        返回:
            平均损失和平均评估指标值
        """
        self.ALSTM_model.eval()  # 设置模型为评估模式

        scores = []  # 存储每个批次的评估指标
        losses = []  # 存储每个批次的损失

        # 遍历数据加载器中的批次
        for data, weight in data_loader:
            # 提取特征
            feature = data[:, :, 0:-1].to(self.device)
            # 提取标签
            label = data[:, -1, -1].to(self.device)

            # 不计算梯度，加速推理
            with torch.no_grad():
                pred = self.ALSTM_model(feature.float())  # 模型预测
                loss = self.loss_fn(pred, label, weight.to(self.device))  # 计算损失
                losses.append(loss.item())  # 记录损失

                score = self.metric_fn(pred, label)  # 计算评估指标
                scores.append(score.item())  # 记录评估指标

        # 返回平均损失和平均评估指标
        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        """
        训练模型
        
        参数:
            dataset: 数据集
            evals_result: 用于存储评估结果的字典
            save_path: 模型保存路径
            reweighter: 权重调整器，用于样本加权
        """
        # 准备训练和验证数据
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # 检查数据是否为空
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        # 配置数据加载器的缺失值处理方式
        dl_train.config(fillna_type="ffill+bfill")  # 前向填充+后向填充处理缺失值
        dl_valid.config(fillna_type="ffill+bfill")

        # 处理样本权重
        if reweighter is None:
            # 如果没有权重调整器，使用全1权重
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            # 使用权重调整器计算权重
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        # 创建训练数据加载器
        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),  # 拼接数据集和权重
            batch_size=self.batch_size,
            shuffle=True,  # 训练集打乱顺序
            num_workers=self.n_jobs,  # 并行加载的进程数
            drop_last=True,  # 丢弃最后一个不完整的批次
        )
        # 创建验证数据加载器
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,  # 验证集不打乱顺序
            num_workers=self.n_jobs,
            drop_last=True,
        )

        # 获取或创建模型保存路径
        save_path = get_or_create_path(save_path)

        # 初始化早停相关参数
        stop_steps = 0  # 记录连续未提升的轮数
        train_loss = 0  # 训练损失
        best_score = -np.inf  # 最佳评估指标值
        best_epoch = 0  # 最佳轮次
        evals_result["train"] = []  # 存储训练评估结果
        evals_result["valid"] = []  # 存储验证评估结果

        # 开始训练
        self.logger.info("training...")
        self.fitted = True  # 标记模型开始训练

        # 遍历每个训练轮次
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)  # 训练一个轮次
            
            self.logger.info("evaluating...")
            # 在训练集和验证集上评估
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            
            # 记录评估结果
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            # 检查是否是最佳模型
            if val_score > best_score:
                best_score = val_score  # 更新最佳评估指标
                stop_steps = 0  # 重置早停计数器
                best_epoch = step  # 更新最佳轮次
                best_param = copy.deepcopy(self.ALSTM_model.state_dict())  # 保存最佳参数
            else:
                stop_steps += 1  # 早停计数器加1
                # 检查是否触发早停
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        # 训练结束，记录最佳结果
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.ALSTM_model.load_state_dict(best_param)  # 加载最佳参数
        torch.save(best_param, save_path)  # 保存最佳模型参数

        # 如果使用GPU，清空缓存
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        """
        模型预测
        
        参数:
            dataset: 数据集
            segment: 预测的数据集片段（如"test"）
        
        返回:
            预测结果的Series
        """
        # 检查模型是否已训练
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        # 准备测试数据
        dl_test = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")  # 处理缺失值
        # 创建测试数据加载器
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.ALSTM_model.eval()  # 设置模型为评估模式
        preds = []  # 存储预测结果

        # 遍历测试数据批次
        for data in test_loader:
            # 提取特征
            feature = data[:, :, 0:-1].to(self.device)

            # 不计算梯度，加速推理
            with torch.no_grad():
                # 模型预测并转换为numpy数组
                pred = self.ALSTM_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)  # 收集预测结果

        # 拼接所有批次的预测结果并返回
        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class ALSTMModel(nn.Module):
    """
    ALSTM模型的网络结构实现
    
    包含一个LSTM网络和一个注意力机制，用于对时序数据进行预测。
    注意力机制用于突出重要的时间步特征。
    """
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()  # 调用父类构造函数
        self.hid_size = hidden_size  # 隐藏层大小
        self.input_size = d_feat  # 输入特征维度
        self.dropout = dropout  # dropout比率
        self.rnn_type = rnn_type  # RNN类型（LSTM或GRU）
        self.rnn_layer = num_layers  # RNN层数
        self._build_model()  # 构建模型结构

    def _build_model(self):
        """构建模型的网络结构"""
        try:
            # 根据指定的RNN类型获取对应的PyTorch模块
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            # 如果RNN类型不支持，抛出错误
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
            
        # 创建输入处理的Sequential网络
        self.net = nn.Sequential()
        # 添加线性层：将输入特征维度转换为隐藏层大小
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        # 添加Tanh激活函数
        self.net.add_module("act", nn.Tanh())
        
        # 创建RNN层
        self.rnn = klass(
            input_size=self.hid_size,  # 输入大小（与前一层输出匹配）
            hidden_size=self.hid_size,  # 隐藏层大小
            num_layers=self.rnn_layer,  # 层数
            batch_first=True,  # 批处理维度在前
            dropout=self.dropout,  # dropout比率
        )
        
        # 输出层：将RNN输出转换为预测结果
        # 输入是RNN最后一步输出和注意力加权输出的拼接，所以大小是2*hid_size
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        
        # 注意力网络：计算每个时间步的注意力权重
        self.att_net = nn.Sequential()
        # 注意力网络第一层：降维
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        # 注意力网络dropout层
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        # 注意力网络激活函数
        self.att_net.add_module("att_act", nn.Tanh())
        # 注意力网络输出层：得到每个时间步的权重
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        # 注意力网络softmax层：将权重归一化
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        """
        模型前向传播
        
        参数:
            inputs: 输入数据，形状为[batch, seq_len, input_size]
            
        返回:
            预测结果，形状为[batch]
        """
        # 输入数据经过预处理网络，转换为RNN的输入格式
        # rnn_out形状: [batch, seq_len, hidden_size]
        rnn_out, _ = self.rnn(self.net(inputs))
        
        # 计算注意力分数: [batch, seq_len, 1]
        attention_score = self.att_net(rnn_out)
        
        # 应用注意力权重：将RNN输出与注意力分数相乘
        out_att = torch.mul(rnn_out, attention_score)
        # 对时间步维度求和，得到注意力加权的特征: [batch, hidden_size]
        out_att = torch.sum(out_att, dim=1)
        
        # 拼接RNN最后一个时间步的输出和注意力加权输出
        # rnn_out[:, -1, :]是最后一个时间步的输出
        # 拼接后形状: [batch, 2*hidden_size]
        # 通过输出层得到预测结果: [batch, 1]
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )
        
        # 去除最后一个维度，返回形状为[batch]的结果
        return out[..., 0]
    