import logging
import sys

def setup_logging():
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 创建简单的格式化器，与 Uvicorn 格式保持一致
    formatter = logging.Formatter('INFO:     %(message)s')
    console_handler.setFormatter(formatter)

    # 添加处理器到根日志记录器
    root_logger.addHandler(console_handler)

    # 设置特定模块的日志级别
    logging.getLogger('api.core.managers').setLevel(logging.INFO)
    logging.getLogger('api.services.task_manager').setLevel(logging.INFO) 