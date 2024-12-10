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
        self.model_dir = settings.model_management['model_dir']
        self.temp_dir = settings.model_management['temp_dir']
        self.model_config_template = settings.model_management['model_config_template']
        self.required_files = settings.model_management['required_files']
        
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
                file_path = os.path.join(temp_model_dir, file.filename)
                with open(file_path, 'wb') as f:
                    content = await file.read()
                    f.write(content)
                saved_files.append(file.filename)
            
            # 验证必需文件
            missing_files = [f for f in self.required_files if f not in saved_files]
            if missing_files:
                # 清理临时文件
                shutil.rmtree(temp_model_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"缺少必需的文件: {', '.join(missing_files)}"
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
                **self.model_config_template,  # 使用模板作为基础
                'name': model_info.name,
                'code': model_info.code,
                'version': model_info.version,
                'author': model_info.author,
                'description': model_info.description,
                'create_time': datetime.now().isoformat(),
                'update_time': datetime.now().isoformat()
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(model_config, f)
            
            return {
                'code': model_info.code,
                'files': saved_files,
                'path': model_path
            }
            
        except Exception as e:
            # 确保清理临时文件
            if 'temp_model_dir' in locals():
                shutil.rmtree(temp_model_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(e))

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有模型"""
        try:
            models = []
            for model_code in os.listdir(self.model_dir):
                model_path = os.path.join(self.model_dir, model_code)
                if not os.path.isdir(model_path):
                    continue
                
                config_path = os.path.join(model_path, 'config.yaml')
                if not os.path.exists(config_path):
                    continue
                
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        models.append(config)
                except Exception as e:
                    print(f"读取模型配置失败 {model_code}: {str(e)}")
            
            return models
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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