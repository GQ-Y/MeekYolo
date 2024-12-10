import os
import ctypes
import logging
from ctypes.util import find_library
from typing import Optional

logger = logging.getLogger(__name__)

# 全局缓存已加载的库
_lib_cache: Optional[ctypes.CDLL] = None

def load_zlmediakit() -> ctypes.CDLL:
    """
    加载ZLMediaKit动态库
    
    Returns:
        ctypes.CDLL: 加载的动态库对象
    
    Raises:
        RuntimeError: 如果找不到动态库
    """
    global _lib_cache
    
    # 如果已经加载过，直接返回缓存的实例
    if _lib_cache is not None:
        return _lib_cache

    # 获取系统类型
    is_darwin = os.uname().sysname == 'Darwin'
    lib_name = 'libmk_api.dylib' if is_darwin else 'libmk_api.so'
    
    # 从环境变量获取额外的搜索路径
    ld_library_paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    
    # 定义搜索路径
    search_paths = []
    
    # 添加环境变量中的路径
    for path in ld_library_paths:
        if path:
            search_paths.append(os.path.join(path, lib_name))
    
    # 添加其他标准路径
    search_paths.extend([
        # 用户目录（优先）
        os.path.join('/home/appuser/ZLMediaKit/release',
                    'darwin' if is_darwin else 'linux',
                    'Release', lib_name),
        
        # 系统库路径
        os.path.join('/usr/local/lib', lib_name),
        os.path.join('/usr/lib', lib_name),
        
        # 当前目录
        os.path.join('.', lib_name),
        
        # 使用系统的库查找机制
        find_library('mk_api')
    ])
    
    # 记录尝试过的路径和错误
    errors = []
    
    # 尝试加载库文件
    for lib_path in search_paths:
        if not lib_path:
            continue
            
        try:
            if os.path.exists(lib_path):
                logger.debug(f"尝试加载库文件: {lib_path}")
                # 使用RTLD_GLOBAL确保符号对其他库可见
                _lib_cache = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                logger.info(f"成功加载ZLMediaKit动态库: {lib_path}")
                return _lib_cache
        except Exception as e:
            error_msg = f"加载 {lib_path} 失败: {str(e)}"
            logger.debug(error_msg)
            errors.append(error_msg)
    
    # 如果所有路径都失败，抛出详细的异常
    error_details = "\n".join(errors)
    error_msg = f"找不到ZLMediaKit动态库，尝试过以下路径:\n{error_details}"
    logger.error(error_msg)
    raise RuntimeError(error_msg) 