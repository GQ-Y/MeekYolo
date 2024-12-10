from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import zipfile
import io
import yaml
import os
import shutil
from api.models.models import ModelInfo
from api.services.model import ModelService

router = APIRouter()

model_service = ModelService()

@router.get("")
@router.get("/")
async def list_models():
    """列出所有模型"""
    try:
        return await model_service.list_models()
    except Exception as e:
        print(f"列出模型失败: {str(e)}")
        raise

@router.get("/{code}")
async def get_model(code: str):
    """获取指定模型的信息"""
    return await model_service.get_model(code)

@router.post("/upload")
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(None),
    code: str = Form(None),
    version: str = Form("1.0.0"),
    author: str = Form(""),
    description: str = Form("")
):
    """
    上传模型文件（ZIP格式）
    
    - file: ZIP文件，包含模型文件和配置文件
    - name: 模型名称（可选，如果不提供则从yaml中读取）
    - code: 模���代码（可选，如果不提供则从yaml中读取）
    - version: 版本号（可选）
    - author: 作者（可选）
    - description: 描述（可选）
    """
    try:
        content = await file.read()
        zip_file = zipfile.ZipFile(io.BytesIO(content))
        
        # 过滤掉 __MACOSX 目录和隐藏文件
        file_list = [
            f for f in zip_file.namelist()
            if not f.startswith('__MACOSX') and not f.startswith('.')
        ]
        print(f"有效文件列表: {file_list}")
        
        # 查找模型文件
        pt_files = [f for f in file_list if f.endswith('.pt')]
        if not pt_files:
            raise HTTPException(
                status_code=400,
                detail="ZIP文件中未找到.pt模型文件"
            )
        
        # 确定主要模型文件
        main_model_file = None
        if len(pt_files) == 1:
            main_model_file = pt_files[0]
        else:
            # 如果有多个.pt文件，优先使用best.pt
            best_pt = next((f for f in pt_files if os.path.basename(f) == 'best.pt'), None)
            if best_pt:
                main_model_file = best_pt
            else:
                main_model_file = pt_files[0]  # 使用第一个.pt文件
        
        print(f"主要模型文件: {main_model_file}")
        
        # 查找并读取配置文件
        yaml_files = [f for f in file_list if f.endswith('.yaml')]
        yaml_content = None
        config_file = None
        
        for yaml_file in yaml_files:
            with zip_file.open(yaml_file) as f:
                try:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict) and 'code' in content:
                        yaml_content = content
                        config_file = yaml_file
                        print(f"找到模型配置文件: {yaml_file}")
                        break
                except Exception as e:
                    print(f"解析YAML文件失败 {yaml_file}: {str(e)}")
                    continue
        
        # 从配置或参数中获取模型信息
        model_info = ModelInfo(
            name=name or (yaml_content.get('name') if yaml_content else os.path.splitext(file.filename)[0]),
            code=code or (yaml_content.get('code') if yaml_content else os.path.splitext(file.filename)[0]),
            version=version,
            author=author or (yaml_content.get('author') if yaml_content else ''),
            description=description or (yaml_content.get('description') if yaml_content else '')
        )
        
        print(f"模型信息: {model_info}")
        
        # 解压文件并上传
        temp_dir = f"temp/models/{model_info.code}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 只解压有效文件
        for file_path in file_list:
            if not file_path.endswith('/'):  # 跳过目录
                target_path = os.path.join(temp_dir, os.path.basename(file_path))
                with zip_file.open(file_path) as source, open(target_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
        
        # 如果主要模型文件不是best.pt，重命名为best.pt
        if os.path.basename(main_model_file) != 'best.pt':
            src_path = os.path.join(temp_dir, os.path.basename(main_model_file))
            dst_path = os.path.join(temp_dir, 'best.pt')
            os.rename(src_path, dst_path)
            print(f"重命名主要模型文件: {main_model_file} -> best.pt")
        
        # 转换为文件列表
        files = []
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                files.append(UploadFile(
                    filename=filename,
                    file=open(file_path, 'rb')
                ))
        
        print(f"准备上传文件: {[f.filename for f in files]}")
        
        # 调用服务处理
        result = await model_service.upload_model(files, model_info)
        
        # 清理临时文件
        for f in files:
            f.file.close()
        
        return {
            "message": "模型上传成功",
            "model": result,
            "main_model": "best.pt",  # 主要模型文件始终是best.pt
            "config_file": os.path.basename(config_file) if config_file else None
        }
        
    except Exception as e:
        print(f"上传失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"上传失败: {str(e)}"
        )

@router.delete("/{code}")
async def delete_model(code: str):
    """删除模型"""
    return await model_service.delete_model(code)

@router.get("")
async def list_models_root():
    """列出所有模型（根路由）"""
    return await list_models() 