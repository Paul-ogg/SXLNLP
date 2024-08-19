# -*- coding: utf-8 -*-
"""
bug1:too many dimensions 'str'
这个错误通常发生在尝试对字符串（str）进行类似于多维数组或矩阵的操作时

bug2:invalid literal for int() with base 10: 'label'
如果字符串包含非数字字符（如字母、空格、特殊字符等），则int()函数会抛出ValueError异常，提示“invalid literal for int() with base 10”。


在Python中，当字典作为函数参数传递时，实际上是传递了字典的引用（或者说是指针），
而不是字典的深拷贝或浅拷贝。这意味着在函数内部对字典所做的任何修改都会反映到原始字典上。

尝试使用均方差损失函数（MSE loss）来训练模型，遇到未知难题：
任何模型的每一轮的准确率都在0.67左右

"""
import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import csv
import time
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""
 

seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"model_type={config['model_type']} lr={config['learning_rate']} hidden_size={Config['hidden_size']} batch_size={Config['batch_size']} pooling_style={Config['pooling_style']}")
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":


    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    with open("models-accuracy.csv", "a",encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "lr", "hidden_size", "batch_size", "pooling_style", "accuracy", "time"])

    # for model in ["bert_mid_layer","bert_cnn","bert_lstm","bert","rcnn","stack_gated_cnn","gated_cnn","cnn","rnn","gru","lstm","fast_text"]:
    for model in ["bert_cnn","bert_lstm","rcnn"]:
        Config["model_type"] = model
        for lr in [1e-3]:
            Config["learning_rate"] = lr
            for hidden_size in [64]:
                Config["hidden_size"] = hidden_size
                for batch_size in [256]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg"]:
                        Config["pooling_style"] = pooling_style
                        start_time = time.time()
                        acc = main(Config)
                        end_time = time.time()
                        total_time = end_time-start_time
                        with open("models-accuracy.csv", "a",encoding='utf-8') as f:
                             writer = csv.writer(f)
                             writer.writerow([model, lr, hidden_size, batch_size, pooling_style, round(acc,4),int(total_time)])

