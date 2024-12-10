import os
import shutil
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, UploadFile
from api.core.config import settings
from api.models.models import ModelInfo

class ModelService:
    def __init__(self):
        self.model_dir = 'model'
        self.temp_dir = 'temp/models'
        self.model_config_template = {
            # 模型管理信息
            'code': '',            # 模型编码
            'version': '1.0.0',    # 模型版本
            'name': '',            # 模型名称
            'description': '',      # 模型描述
            'author': '',          # 作者
            'create_time': '',     # 创建时间
            'update_time': '',     # 更新时间
            
            # 训练配置
            'path': '',            # 模型路径
            'train': 'images/train', # 训练集路径
            'val': 'images/val',    # 验证集路径
            'nc': 0,               # 类别数
            'names': {}            # 类别名称映射
        }
        self.required_files = ['best.pt', 'data.yaml']
        
        # 确保目录存在
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    async def upload_model(self, files: List[UploadFile], model_info: ModelInfo) -> Dict[str, Any]:
        """上传模型文件"""
        try:
            # 创建临时目录
            temp_model_dir = os.path.join(self.temp_dir, model_info.code)
            os.makedirs(temp_model_dir, exist_ok=True)
            
            # 保存上传的文件
            saved_files = []
            for file in files:
                if not file or not file.filename:  # 添加文件检查
                    continue
                
                file_path = os.path.join(temp_model_dir, file.filename)
                try:
                    content = await file.read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    saved_files.append(file.filename)
                except Exception as e:
                    print(f"保存文件失败 {file.filename}: {str(e)}")
                    continue
            
            if not saved_files:
                raise HTTPException(
                    status_code=400,
                    detail="没有成功保存任何文件"
                )
            
            # 检查是否有必需的文件类型
            has_pt = any(f.endswith('.pt') for f in saved_files)
            has_yaml = any(f.endswith('.yaml') for f in saved_files)
            
            if not has_pt or not has_yaml:
                missing = []
                if not has_pt:
                    missing.append(".pt模型文件")
                if not has_yaml:
                    missing.append("YAML配置文件")
                # 清理临时文件
                shutil.rmtree(temp_model_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"缺少必需的文件: {', '.join(missing)}"
                )
            
            # 创建模型目录
            model_path = os.path.join(self.model_dir, model_info.code)
            if os.path.exists(model_path):
                # 如果存在，先备份
                backup_path = f"{model_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.move(model_path, backup_path)
            
            # 移动文件到正式目录
            shutil.move(temp_model_dir, model_path)
            
            # 保存模型信息
            config_path = os.path.join(model_path, 'config.yaml')
            model_config = {
                # 模型管理信息
                'code': model_info.code,
                'version': model_info.version,
                'name': model_info.name,
                'description': model_info.description,
                'author': model_info.author,
                'create_time': datetime.now().isoformat(),
                'update_time': datetime.now().isoformat(),
                
                # 训练配置（从data.yaml读取）
                'path': '',
                'train': 'images/train',
                'val': 'images/val',
                'nc': 0,
                'names': {}
            }
            
            # 如果存在data.yaml，读取其中的配置
            data_yaml_path = os.path.join(model_path, 'data.yaml')
            if os.path.exists(data_yaml_path):
                try:
                    with open(data_yaml_path, 'r', encoding='utf-8') as f:
                        data_config = yaml.safe_load(f)
                        # 更新训练配置
                        for key in ['train', 'val', 'nc', 'names']:
                            if key in data_config:
                                model_config[key] = data_config[key]
                except Exception as e:
                    print(f"读取data.yaml失败: {str(e)}")
            
            # 保存配置
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(model_config, f, allow_unicode=True, sort_keys=False)
            
            return {
                'code': model_info.code,
                'files': saved_files,
                'path': model_path,
                'config': model_config
            }
            
        except Exception as e:
            # 确保清理临时文件
            if 'temp_model_dir' in locals():
                shutil.rmtree(temp_model_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_models(self) -> List[Dict[str, Any]]:
        """列出所有模型"""
        try:
            print("\n开始扫描模型目录:")
            print(f"模型目录路径: {os.path.abspath(self.model_dir)}")
            
            models = []
            
            # 检查目录是否存在
            if not os.path.exists(self.model_dir):
                print(f"模型目录不存在: {self.model_dir}")
                return []
            
            # 列出目录内容
            dir_contents = os.listdir(self.model_dir)
            print(f"目录内容: {dir_contents}")
            
            for model_code in dir_contents:
                model_path = os.path.join(self.model_dir, model_code)
                print(f"\n检查模型: {model_code}")
                print(f"完整路径: {model_path}")
                
                if not os.path.isdir(model_path):
                    print(f"跳过非目录项: {model_code}")
                    continue
                
                config_path = os.path.join(model_path, 'config.yaml')
                print(f"配置文件路径: {config_path}")
                
                if not os.path.exists(config_path):
                    print(f"配置文件不存在: {config_path}")
                    # 尝试创建配置文件
                    try:
                        model_config = {
                            **self.model_config_template,
                            'name': model_code,
                            'code': model_code,
                            'create_time': datetime.now().isoformat(),
                            'update_time': datetime.now().isoformat()
                        }
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)
                        with open(config_path, 'w') as f:
                            yaml.dump(model_config, f)
                        print(f"已创建配置文件: {config_path}")
                    except Exception as e:
                        print(f"创建配置文件失败: {str(e)}")
                        continue
                
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        models.append(config)
                        print(f"成功加载模型配置: {model_code}")
                except Exception as e:
                    print(f"读取模型配置失败 {model_code}: {str(e)}")
                    continue
            
            print(f"\n扫描完成，找到 {len(models)} 个模型")
            return models
            
        except Exception as e:
            print(f"列出模型失败: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"列出模型失败: {str(e)}"
            )

    def get_model(self, code: str) -> Optional[Dict[str, Any]]:
        """获取指定模型的信息"""
        try:
            model_path = os.path.join(self.model_dir, code)
            if not os.path.exists(model_path):
                return None
            
            config_path = os.path.join(model_path, 'config.yaml')
            if not os.path.exists(config_path):
                return None
            
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def delete_model(self, code: str) -> bool:
        """删除指定的模型"""
        try:
            model_path = os.path.join(self.model_dir, code)
            if not os.path.exists(model_path):
                return False
            
            # 移动到备份目录而不是直接删除
            backup_path = f"{model_path}_deleted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(model_path, backup_path)
            return True
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))