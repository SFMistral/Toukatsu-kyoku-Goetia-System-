#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging 模块使用示例

演示如何使用logging模块进行日志记录和指标追踪。
"""

import time
import random
from logging import setup_logging, get_logger, log_metrics, log_hyperparams


def main():
    """主函数"""
    
    # 1. 初始化日志系统
    config = {
        'level': 'INFO',
        'console': {
            'enabled': True,
            'level': 'INFO',
            'use_colors': True,
            'show_progress': True
        },
        'file': {
            'enabled': True,
            'log_dir': 'logs',
            'filename': 'example.log',
            'level': 'DEBUG'
        },
        'json': {
            'enabled': True,
            'log_dir': 'logs',
            'filename': 'example.jsonl',
            'separate_metrics': True
        }
    }
    
    setup_logging(config)
    
    # 2. 获取日志器
    logger = get_logger("example")
    
    # 3. 记录超参数
    hyperparams = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'model': 'ResNet50'
    }
    log_hyperparams(hyperparams)
    logger.info("Training started with hyperparameters")
    
    # 4. 模拟训练过程
    simulate_training(logger)
    
    # 5. 演示日志解析
    demonstrate_log_parsing()
    
    logger.info("Example completed successfully")


def simulate_training(logger):
    """模拟训练过程"""
    
    epochs = 5
    steps_per_epoch = 10
    
    for epoch in range(epochs):
        with logger.epoch(epoch):
            logger.info(f"Starting epoch {epoch}")
            
            # 训练阶段
            with logger.train():
                epoch_loss = 0.0
                epoch_acc = 0.0
                
                for step in range(steps_per_epoch):
                    with logger.step(step):
                        # 模拟训练步骤
                        loss = random.uniform(0.1, 2.0) * (0.9 ** epoch)
                        acc = random.uniform(0.7, 0.95) + (epoch * 0.02)
                        lr = 0.001 * (0.9 ** epoch)
                        
                        # 记录指标
                        logger.log_scalar("loss", loss, step)
                        logger.log_scalar("accuracy", acc, step)
                        logger.log_scalar("learning_rate", lr, step)
                        
                        # 批量记录指标
                        metrics = {
                            "train/loss": loss,
                            "train/accuracy": acc,
                            "train/lr": lr
                        }
                        log_metrics(metrics, step)
                        
                        epoch_loss += loss
                        epoch_acc += acc
                        
                        # 模拟处理时间
                        time.sleep(0.1)
                        
                        if step % 5 == 0:
                            logger.info(f"Step {step}: loss={loss:.4f}, acc={acc:.4f}")
                            
            # 验证阶段
            with logger.val():
                val_loss = epoch_loss / steps_per_epoch * 0.8
                val_acc = epoch_acc / steps_per_epoch * 1.1
                
                logger.log_scalar("val_loss", val_loss, epoch)
                logger.log_scalar("val_accuracy", val_acc, epoch)
                
                logger.info(f"Validation: loss={val_loss:.4f}, acc={val_acc:.4f}")
                
                # 记录权重直方图（模拟）
                weights = [random.gauss(0, 1) for _ in range(100)]
                logger.log_histogram("model/weights", weights, epoch)
                
            logger.info(f"Epoch {epoch} completed")
            
    logger.info("Training completed")


def demonstrate_log_parsing():
    """演示日志解析功能"""
    
    from logging.utils import LogParser, LogAggregator
    
    logger = get_logger("parser_demo")
    logger.info("Demonstrating log parsing capabilities")
    
    # 解析日志文件
    parser = LogParser()
    
    try:
        # 解析文本日志
        if os.path.exists("logs/example.log"):
            results = parser.parse_file("logs/example.log")
            logger.info(f"Parsed {len(results['records'])} log records")
            logger.info(f"Found {len(results['metrics'])} metrics")
            
            # 获取摘要
            summary = parser.get_log_summary("logs/example.log")
            logger.info(f"Log summary: {summary['record_counts']}")
            
        # 解析JSON日志
        if os.path.exists("logs/example.jsonl"):
            json_results = parser.parse_json_file("logs/example.jsonl")
            logger.info(f"Parsed JSON log with {len(json_results['records'])} records")
            
    except Exception as e:
        logger.error(f"Log parsing failed: {e}", exc_info=True)


def demonstrate_decorators():
    """演示装饰器功能"""
    
    from logging import log_function_call, log_execution_time
    
    @log_function_call("decorator_demo")
    @log_execution_time("decorator_demo")
    def slow_function(n):
        """一个慢函数用于演示"""
        time.sleep(n)
        return n * 2
        
    logger = get_logger("decorator_demo")
    logger.info("Testing decorators")
    
    result = slow_function(0.5)
    logger.info(f"Function result: {result}")


def demonstrate_context_managers():
    """演示上下文管理器"""
    
    logger = get_logger("context_demo")
    
    # 计时器上下文
    with logger.timer("data_loading"):
        time.sleep(0.2)
        logger.info("Loading data...")
        
    with logger.timer("model_inference"):
        time.sleep(0.1)
        logger.info("Running inference...")
        
    # 阶段上下文
    with logger.train():
        logger.info("This is in training phase")
        
        with logger.epoch(1):
            logger.info("This is in epoch 1")
            
            with logger.step(10):
                logger.info("This is at step 10")


if __name__ == "__main__":
    import os
    
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    
    try:
        main()
        demonstrate_decorators()
        demonstrate_context_managers()
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()