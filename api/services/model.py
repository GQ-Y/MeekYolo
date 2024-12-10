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
        temp_model_dir = None
        try:
            # 创建临时目录
            temp_model_dir = os.path.join(self.temp_dir, model_info.code)
            os.makedirs(temp_model_dir, exist_ok=True)
            
            # 保存上传的文件
            saved_files = []
            for file in files:
                if not file or not file.filename:
                    continue
                
                file_path = os.path.join(temp_model_dir, file.filename)
                try:
                    # 重置文件指针
                    await file.seek(0)
                    content = await file.read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    saved_files.append(file.filename)
                    print(f"成功保存文件: {file.filename}")
                except Exception as e:
                    print(f"保存文件失败 {file.filename}: {str(e)}")
                    continue
            
            if not saved_files:
                raise HTTPException(
                    status_code=400,
                    detail="没有成功保存任何文件"
                )
            
            print(f"已保存的文件: {saved_files}")
            
            # 检查是否有必需的文件类型
            has_pt = any(f.endswith('.pt') for f in saved_files)
            has_yaml = any(f.endswith('.yaml') for f in saved_files)
            
            if not has_pt or not has_yaml:
                missing = []
                if not has_pt:
                    missing.append(".pt模型文件")
                if not has_yaml:
                    missing.append("YAML配置文件")
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
                print(f"已备份现有模型: {backup_path}")
            
            # 移动文件到正式目录
            shutil.move(temp_model_dir, model_path)
            print(f"已移动文件到模型目录: {model_path}")
            
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
                    print(f"已读取data.yaml配置")
                except Exception as e:
                    print(f"读取data.yaml失败: {str(e)}")
            
            # 保存配置
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(model_config, f, allow_unicode=True, sort_keys=False)
            print(f"已保存模型配置: {config_path}")
            
            return {
                'code': model_info.code,
                'files': saved_files,
                'path': model_path,
                'config': model_config
            }
            
        except Exception as e:
            # 确保清理临时文件
            if temp_model_dir and os.path.exists(temp_model_dir):
                shutil.rmtree(temp_model_dir, ignore_errors=True)
                print(f"已清理临时目录: {temp_model_dir}")
            raise HTTPException(status_code=500, detail=str(e))

    async def list_models(self) -> List[Dict[str, Any]]:
        """列出所有模型"""
        try:
            print("\n开始扫描模型目录:")
            print(f"模型目录路径: {os.path.abspath(self.model_dir)}")
            
            models = []
            
            # 确保目录存在
            os.makedirs(self.model_dir, exist_ok=True)
            
            # 列出目录内容
            dir_contents = [d for d in os.listdir(self.model_dir) 
                           if os.path.isdir(os.path.join(self.model_dir, d))]
            print(f"找到模型目录: {dir_contents}")
            
            for model_code in dir_contents:
                model_path = os.path.join(self.model_dir, model_code)
                print(f"\n检查模型: {model_code}")
                
                # 检查必需文件
                has_pt = any(f.endswith('.pt') 
                            for f in os.listdir(model_path) 
                            if os.path.isfile(os.path.join(model_path, f)))
                
                if not has_pt:
                    print(f"跳过无效模型(缺少.pt文件): {model_code}")
                    continue
                
                config_path = os.path.join(model_path, 'config.yaml')
                if not os.path.exists(config_path):
                    print(f"配置文件不存在，尝试从data.yaml创建")
                    
                    # 尝试从data.yaml读取信息
                    data_yaml_path = os.path.join(model_path, 'data.yaml')
                    if os.path.exists(data_yaml_path):
                        try:
                            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                                data_config = yaml.safe_load(f)
                            
                            model_config = {
                                **self.model_config_template,
                                'name': data_config.get('name', model_code),
                                'code': model_code,
                                'create_time': datetime.now().isoformat(),
                                'update_time': datetime.now().isoformat()
                            }
                            
                            # 更新训练配置
                            for key in ['train', 'val', 'nc', 'names']:
                                if key in data_config:
                                    model_config[key] = data_config[key]
                            
                            # 保存配置
                            with open(config_path, 'w', encoding='utf-8') as f:
                                yaml.dump(model_config, f, allow_unicode=True, sort_keys=False)
                            print(f"已创建配置文件: {config_path}")
                            
                            models.append(model_config)
                            continue
                            
                        except Exception as e:
                            print(f"从data.yaml创建配置失败: {str(e)}")
                
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        models.append(config)
                        print(f"成功加载模型配置: {model_code}")
                except Exception as e:
                    print(f"读取模型配置失败 {model_code}: {str(e)}")
                    continue
            
            print(f"\n扫描完成，找到 {len(models)} 个有效模型")
            return models
            
        except Exception as e:
            print(f"列出模型失败: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"列出模型失败: {str(e)}"
            )

    async def get_model(self, code: str) -> Optional[Dict[str, Any]]:
        """获取指定模型的信息"""
        try:
            print(f"\n获取模型信息: {code}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"模型根目录: {self.model_dir}")
            
            # 尝试所有可能的路径
            possible_paths = [
                os.path.join(self.model_dir, code),
                os.path.join(self.model_dir, f"model-{code}"),
                os.path.join(self.model_dir, code.replace('model-', '')),
            ]
            
            model_path = None
            for path in possible_paths:
                print(f"尝试路径: {path}")
                if os.path.exists(path):
                    model_path = path
                    print(f"找到有效路径: {path}")
                    break
            
            if not model_path:
                print(f"未找到模型目录，尝试过的路径: {possible_paths}")
                return None
            
            # 检查目录内容
            try:
                files = os.listdir(model_path)
                print(f"目录内容: {files}")
            except Exception as e:
                print(f"读取目录失败: {str(e)}")
                return None
            
            # 过滤出文件
            files = [f for f in files if os.path.isfile(os.path.join(model_path, f))]
            print(f"文件列表: {files}")
            
            # 检查必需文件
            has_pt = any(f.endswith('.pt') for f in files)
            if not has_pt:
                print(f"无效模型(缺少.pt文件): {code}")
                return None
            
            # 读取配置
            config_path = os.path.join(model_path, 'config.yaml')
            if not os.path.exists(config_path):
                print(f"配置文件不存在: {config_path}")
                data_yaml_path = os.path.join(model_path, 'data.yaml')
                if os.path.exists(data_yaml_path):
                    print(f"尝试从data.yaml创建配置: {data_yaml_path}")
                    try:
                        with open(data_yaml_path, 'r', encoding='utf-8') as f:
                            data_config = yaml.safe_load(f)
                            print(f"data.yaml内容: {data_config}")
                        
                        model_config = {
                            **self.model_config_template,
                            'name': data_config.get('name', code),
                            'code': code,
                            'create_time': datetime.now().isoformat(),
                            'update_time': datetime.now().isoformat()
                        }
                        
                        # 更新训练配置
                        for key in ['train', 'val', 'nc', 'names']:
                            if key in data_config:
                                model_config[key] = data_config[key]
                        
                        # 保存配置
                        with open(config_path, 'w', encoding='utf-8') as f:
                            yaml.dump(model_config, f, allow_unicode=True, sort_keys=False)
                        print(f"已创建配置文件: {config_path}")
                        
                        # 添加文件信息
                        model_config['files'] = files
                        model_config['main_model'] = 'best.pt' if 'best.pt' in files else [f for f in files if f.endswith('.pt')][0]
                        
                        return model_config
                        
                    except Exception as e:
                        print(f"从data.yaml创建配置失败: {str(e)}")
                        return None
                
                return None
            
            # 读取配置文件
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    print(f"读取到的配置: {config}")
                    
                    # 添加文件信息
                    config['files'] = files
                    
                    # 添加主要模型文件信息
                    pt_files = [f for f in files if f.endswith('.pt')]
                    if 'best.pt' in pt_files:
                        config['main_model'] = 'best.pt'
                    elif pt_files:
                        config['main_model'] = pt_files[0]
                    
                    print(f"最终配置: {config}")
                    return config
                    
            except Exception as e:
                print(f"读取配置文件失败: {str(e)}")
                return None
            
        except Exception as e:
            print(f"获取模型信息失败: {str(e)}")
            print(f"异常类型: {type(e)}")
            print(f"异常详情: {str(e)}")
            import traceback
            print(f"堆栈跟踪:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_model(self, code: str) -> Dict[str, Any]:
        """删除指定的模型"""
        try:
            print(f"\n删除模型: {code}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"模型根目录: {self.model_dir}")
            
            # 尝试所有可能的路径
            possible_paths = [
                os.path.join(self.model_dir, code),
                os.path.join(self.model_dir, f"model-{code}"),
                os.path.join(self.model_dir, code.replace('model-', '')),
            ]
            
            model_path = None
            for path in possible_paths:
                print(f"尝试路径: {path}")
                if os.path.exists(path):
                    model_path = path
                    print(f"找到有效路径: {path}")
                    break
            
            if not model_path:
                print(f"未找到模型目录，尝试过的路径: {possible_paths}")
                raise HTTPException(
                    status_code=404,
                    detail=f"模型不存在: {code}"
                )
            
            # 获取模型信息（用于返回）
            model_info = await self.get_model(code)
            if not model_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"无效的模型: {code}"
                )
            
            # 移动到备份目录而不是直接删除
            backup_path = f"{model_path}_deleted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                shutil.move(model_path, backup_path)
                print(f"已备份模型到: {backup_path}")
                
                return {
                    "status": "success",
                    "message": "模型已删除",
                    "code": code,
                    "backup_path": backup_path,
                    "model_info": model_info
                }
                
            except Exception as e:
                print(f"备份模型失败: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"删除模型失败: {str(e)}"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            print(f"删除模型失败: {str(e)}")
            print(f"异常类型: {type(e)}")
            print(f"异常详情: {str(e)}")
            import traceback
            print(f"堆栈跟踪:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))