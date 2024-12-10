from .analysis import router as analysis_router
from .tasks import router as tasks_router
from .models import router as models_router
from .config import router as config_router
from .system import router as system_router

__all__ = [
    'analysis_router',
    'tasks_router',
    'models_router',
    'config_router',
    'system_router'
] 