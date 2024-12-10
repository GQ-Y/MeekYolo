from ctypes import *
import os
import logging
from urllib.parse import urlparse
import time
from threading import Event

logger = logging.getLogger(__name__)

class ZLMediaKitSDK:
    def __init__(self):
        self.players = {}
        self.callbacks = {}  # 保存回调函数的引用
        self.events = {}  # 保存事件对象
        
        # 加载动态库
        self._load_library()
    
    def _load_library(self):
        """加载动态库"""
        try:
            # 尝试多个可能的路径
            possible_paths = [
                "/usr/local/lib/libmk_api.dylib",  # macOS默认安装路径
                "/usr/lib/libmk_api.so",           # Linux默认安装路径
                os.path.expanduser("~/ZLMediaKit/release/darwin/Release/libmk_api.dylib"),  # macOS编译路径
                os.path.expanduser("~/ZLMediaKit/release/linux/Release/libmk_api.so"),      # Linux编译路径
                "./libmk_api.dylib",  # 当前目录
                "./libmk_api.so",
            ]
            
            lib_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    lib_path = path
                    break
                    
            if not lib_path:
                raise RuntimeError(f"找不到ZLMediaKit动态库，尝试过以下路径: {possible_paths}")
                
            self.lib = cdll.LoadLibrary(lib_path)
            self._init_functions()
            
        except Exception as e:
            logger.error(f"加载动态库失败: {str(e)}")
            raise
    
    def _init_functions(self):
        """初始化SDK函数"""
        try:
            # 创建代理播放器
            self.lib.mk_proxy_player_create3.argtypes = [c_char_p, c_char_p, c_char_p, c_int]
            self.lib.mk_proxy_player_create3.restype = c_void_p
            
            # 开始播放
            self.lib.mk_proxy_player_play.argtypes = [c_void_p, c_char_p]
            self.lib.mk_proxy_player_play.restype = c_int
            
            # 设置播放结果回调
            self.lib.mk_proxy_player_set_on_play_result.argtypes = [c_void_p, CFUNCTYPE(None, c_void_p, c_int, c_void_p), c_void_p]
            self.lib.mk_proxy_player_set_on_play_result.restype = None
            
            # 设置关闭回调
            self.lib.mk_proxy_player_set_on_close.argtypes = [c_void_p, CFUNCTYPE(None, c_void_p, c_void_p), c_void_p]
            self.lib.mk_proxy_player_set_on_close.restype = None
            
            # 关闭播放器
            self.lib.mk_proxy_player_release.argtypes = [c_void_p]
            self.lib.mk_proxy_player_release.restype = None
            
        except Exception as e:
            logger.error(f"初始化SDK函数失败: {str(e)}")
            raise
    
    def create_proxy(self, url: str, stream_id: str, tcp_mode: bool = True) -> tuple:
        """创建RTSP代理"""
        try:
            # 设置代理参数
            vhost = "__defaultVhost__"
            app = "live"
            stream = stream_id
            
            # 创建事件对象
            self.events[stream_id] = Event()
            
            # 创建代理播放器
            player = self.lib.mk_proxy_player_create3(
                vhost.encode(),
                app.encode(),
                stream.encode(),
                1  # 1表示开启RTSP
            )
            
            if not player:
                raise Exception("创建代理播放器失败")
            
            # 生成代理URL，使用8554端口
            proxy_url = f"rtsp://localhost:8554/{app}/{stream}"
            
            # 设置播放结果回调
            @CFUNCTYPE(None, c_void_p, c_int, c_void_p)
            def on_play_result(user_data, err_code, err_msg):
                try:
                    if err_code == 0:
                        self.players[stream_id] = player
                        self.events[stream_id].set()
                    else:
                        logger.error(f"代理播放失败: {err_code}")
                        self.close_proxy(player)
                        self.events[stream_id].set()
                except Exception as e:
                    logger.error(f"回调函数异常: {e}")
            
            # 设置关闭回调
            @CFUNCTYPE(None, c_void_p, c_void_p)
            def on_close(user_data, err_msg):
                if stream_id in self.players:
                    del self.players[stream_id]
                if stream_id in self.callbacks:
                    del self.callbacks[stream_id]
                if stream_id in self.events:
                    self.events[stream_id].set()
            
            # 保存回调函数的引用
            self.callbacks[stream_id] = {
                'play': on_play_result,
                'close': on_close
            }
            
            # 设置回调
            self.lib.mk_proxy_player_set_on_play_result(
                player, 
                self.callbacks[stream_id]['play'],
                None
            )
            self.lib.mk_proxy_player_set_on_close(
                player,
                self.callbacks[stream_id]['close'],
                None
            )
            
            # 开始播放
            ret = self.lib.mk_proxy_player_play(player, url.encode())
            
            # 等待回调执行
            if not self.events[stream_id].wait(timeout=5.0):  # 最多等待5秒
                raise Exception("等待播放结果超时")
            
            # 检查播放器是否还在活跃列表中
            if stream_id not in self.players:
                raise Exception("代理播放器创建失败")
            
            return player, proxy_url
            
        except Exception as e:
            logger.error(f"创建RTSP代理失败: {str(e)}")
            return None, None
        finally:
            # 清理事件
            if stream_id in self.events:
                del self.events[stream_id]
    
    def close_proxy(self, player):
        """关闭RTSP代理"""
        if player:
            try:
                # 从活跃列表中移除
                for stream_id, p in list(self.players.items()):
                    if p == player:
                        del self.players[stream_id]
                        # 同时移除回调函数引用
                        if stream_id in self.callbacks:
                            del self.callbacks[stream_id]
                        break
                
                self.lib.mk_proxy_player_release(player)
            except Exception as e:
                logger.error(f"关闭RTSP代理失败: {str(e)}")