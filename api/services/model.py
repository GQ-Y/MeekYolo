import os
import yaml
import shutil
import zipfile
from fastapi import HTTPException, UploadFile
from datetime import datetime
from api.core.config import settings
from api.models.models import ModelInfo, ModelList

class ModelService:
    def __init__(self):
        self.model_dir = settings.MODEL_CONFIG['model_dir']
        self.temp_dir = settings.MODEL_CONFIG['temp_dir']
        os.makedirs(self.temp_dir, exist_ok=True)
    
    async def get_model_list(self) -> ModelList:
        """获取模型列表"""
        models = []
        for model_dir in os.listdir(self.model_dir):
            model_path = os.path.join(self.model_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            
            config_path = os.path.join(model_path, 'data.yaml')
            if not os.path.exists(config_path):
                continue
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config = yaml.safe_load(f)
                    if not self._validate_model_config(model_config):
                        continue
                    
                    models.append(ModelInfo(
                        code=model_config['code'],
                        version=model_config['version'],
                        name=model_config['name'],
                        description=model_config['description'],
                        author=model_config['author'],
                        create_time=datetime.fromisoformat(model_config['create_time']),
                        update_time=datetime.fromisoformat(model_config['update_time']),
                        path=model_path
                    ))
            except Exception as e:
                print(f"读取模型配置失败: {str(e)}")
                continue
        
        # 获取当前模型
        current_model = None
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            current_model = config['model']['path'].split('/')[-2]
        
        return ModelList(
            models=models,
            current_model=current_model
        )
    
    def _validate_model_config(self, config: dict) -> bool:
        """验证模型配置是否完整"""
        required_fields = [
            'code', 'version', 'name', 'description',
            'author', 'create_time', 'update_time'
        ]
        
        if not all(field in config for field in required_fields):
            return False
        
        if not config['code'] or not config['version'] or not config['name']:
            return False
        
        return True
    
    async def upload_model(self, file: UploadFile) -> ModelInfo:
        """上传模型"""
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="只支持zip格式的模型")
        
        # 保存上传的文件
        temp_zip = os.path.join(self.temp_dir, file.filename)
        temp_extract = os.path.join(self.temp_dir, 'extract')
        
        try:
            # 确保临时目录存在
            os.makedirs(self.temp_dir, exist_ok=True)
            
            print(f"\n开始处理模型上传:")
            print(f"文件名: {file.filename}")
            print(f"临时压缩包路径: {temp_zip}")
            print(f"临时解压目录: {temp_extract}")
            
            # 保存上传的文件
            with open(temp_zip, 'wb') as f:
                content = await file.read()
                f.write(content)
            print(f"文件保存到: {temp_zip}")
            
            # 清理并创建解压目录
            if os.path.exists(temp_extract):
                shutil.rmtree(temp_extract)
            os.makedirs(temp_extract)
            
            # 解压文件
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"\n压缩包内文件列表:")
                for f in file_list:
                    print(f"  - {f}")
                
                # 检查是否有子目录
                root_files = [f for f in file_list if '/' not in f]
                if not root_files:
                    print("\n警告: 所有文件都在子目录中")
                    # 获取第一级目录
                    top_dir = file_list[0].split('/')[0]
                    print(f"使用顶级目录: {top_dir}")
                
                zip_ref.extractall(temp_extract)
                print(f"\n文件已解压到: {temp_extract}")
                
                # 列出解压后的文件
                print("\n解压目录内容:")
                for root, dirs, files in os.walk(temp_extract):
                    level = root.replace(temp_extract, '').count(os.sep)
                    indent = ' ' * 4 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 4 * (level + 1)
                    for f in files:
                        print(f"{subindent}{f}")
            
            # 确定配置文件和模型文件的路径
            if root_files:
                config_path = os.path.join(temp_extract, 'data.yaml')
                model_path = os.path.join(temp_extract, 'best.pt')
            else:
                config_path = os.path.join(temp_extract, top_dir, 'data.yaml')
                model_path = os.path.join(temp_extract, top_dir, 'best.pt')
            
            print(f"\n检查必需文件:")
            print(f"配置文件路径: {config_path} (存在: {os.path.exists(config_path)})")
            print(f"模型文件路径: {model_path} (存在: {os.path.exists(model_path)})")
            
            if not (os.path.exists(config_path) and os.path.exists(model_path)):
                raise HTTPException(status_code=400, detail="模型文件不完整")
            
            # 读取配置
            with open(config_path, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
                print(f"\n模型配置内容:")
                print(yaml.dump(model_config, allow_unicode=True))
                
                if not self._validate_model_config(model_config):
                    raise HTTPException(status_code=400, detail="模型配置不完整")
            
            # 检查是否存在相同编码的模型
            model_code = model_config['code']
            target_dir = os.path.join(self.model_dir, model_code)
            
            print(f"\n目标目录: {target_dir}")
            if os.path.exists(target_dir):
                print("发现已存在的模型，准备更新...")
                shutil.rmtree(target_dir)
            
            # 移动文件到目标目录
            if root_files:
                # 直接移动解压目录
                shutil.move(temp_extract, target_dir)
            else:
                # 移动子目录内容
                source_dir = os.path.join(temp_extract, top_dir)
                shutil.move(source_dir, target_dir)
            
            print(f"文件已移动到目标目录")
            
            # 返回模型信息
            return ModelInfo(
                code=model_config['code'],
                version=model_config['version'],
                name=model_config['name'],
                description=model_config['description'],
                author=model_config['author'],
                create_time=datetime.fromisoformat(model_config['create_time']),
                update_time=datetime.fromisoformat(model_config['update_time']),
                path=target_dir
            )
            
        except Exception as e:
            print(f"\n处理失败: {str(e)}")
            raise
            
        finally:
            # 清理临时文件
            print("\n清理临时文件...")
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
            if os.path.exists(temp_extract):
                shutil.rmtree(temp_extract)
    
    async def set_current_model(self, model_code: str):
        """设置当前使用的模型"""
        # 验证模型是否存在
        model_path = os.path.join(self.model_dir, model_code, 'best.pt')
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="模型不存在")
        
        try:
            # 更新配置
            with open('config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            config['model']['path'] = model_path
            
            # 保存配置
            with open('config/config.yaml', 'w') as f:
                yaml.dump(config, f)
                
            print(f"已将当前模型设置为: {model_code}")
            print(f"模型路径: {model_path}")
            
        except Exception as e:
            print(f"设置当前模型失败: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"设置当前模型失败: {str(e)}"
            )