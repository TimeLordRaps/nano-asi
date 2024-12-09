"""Advanced dependency injection framework for NanoASI."""

from typing import Dict, Any, Type, Callable, Optional, List, Set
from functools import wraps
import inspect
import logging
from .interfaces import ComponentProtocol

logger = logging.getLogger(__name__)

class DependencyContainer:
    """Advanced dependency injection container with lifecycle management."""
    
    def __init__(self):
        self._dependencies: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Set[str] = set()
        self._initialized: Set[str] = set()
        self._dependency_graph: Dict[str, List[str]] = {}
    
    def register(self, key: str, dependency: Any, singleton: bool = True):
        """Register a concrete dependency."""
        if key in self._dependencies:
            logger.warning(f"Overwriting existing dependency: {key}")
        self._dependencies[key] = dependency
        if singleton:
            self._singletons.add(key)
        self._analyze_dependencies(key, dependency)
    
    def register_factory(
        self, 
        key: str, 
        factory: Callable,
        singleton: bool = True
    ):
        """Register a factory for creating dependencies."""
        self._factories[key] = factory
        if singleton:
            self._singletons.add(key)
        self._analyze_dependencies(key, factory)
    
    async def initialize(self):
        """Initialize all registered dependencies in correct order."""
        for key in self._get_initialization_order():
            if key not in self._initialized:
                await self._initialize_dependency(key)
    
    async def _initialize_dependency(self, key: str):
        """Initialize a single dependency."""
        dependency = self.resolve(key)
        if isinstance(dependency, ComponentProtocol):
            try:
                await dependency.initialize({})
                self._initialized.add(key)
            except Exception as e:
                logger.error(f"Failed to initialize {key}: {str(e)}")
                raise
    
    def resolve(self, key: str) -> Any:
        """Resolve a dependency, creating it if necessary."""
        if key in self._dependencies:
            return self._dependencies[key]
        
        if key in self._factories:
            dependency = self._factories[key]()
            if key in self._singletons:
                self._dependencies[key] = dependency
            return dependency
        
        raise ValueError(f"No dependency registered for {key}")
    
    def _analyze_dependencies(self, key: str, dependency: Any):
        """Analyze and record dependency relationships."""
        if inspect.isclass(dependency):
            deps = self._get_constructor_dependencies(dependency)
        elif callable(dependency):
            deps = self._get_function_dependencies(dependency)
        else:
            deps = []
        self._dependency_graph[key] = deps
    
    def _get_constructor_dependencies(self, cls: Type) -> List[str]:
        """Get dependencies from class constructor."""
        try:
            sig = inspect.signature(cls.__init__)
            return [
                param.name for param in sig.parameters.values()
                if param.name != 'self'
            ]
        except ValueError:
            return []
    
    def _get_function_dependencies(self, func: Callable) -> List[str]:
        """Get dependencies from function signature."""
        try:
            sig = inspect.signature(func)
            return list(sig.parameters.keys())
        except ValueError:
            return []
    
    def _get_initialization_order(self) -> List[str]:
        """Get correct dependency initialization order."""
        visited = set()
        order = []
        
        def visit(key: str):
            if key in visited:
                return
            visited.add(key)
            for dep in self._dependency_graph.get(key, []):
                visit(dep)
            order.append(key)
        
        for key in self._dependencies:
            visit(key)
        
        return order

def inject(*dependencies: str):
    """Enhanced decorator for dependency injection."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            container = DependencyContainer()
            await container.initialize()
            injected_kwargs = {}
            
            for dep in dependencies:
                if dep not in kwargs:
                    injected_kwargs[dep] = container.resolve(dep)
            
            kwargs.update(injected_kwargs)
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            container = DependencyContainer()
            injected_kwargs = {}
            
            for dep in dependencies:
                if dep not in kwargs:
                    injected_kwargs[dep] = container.resolve(dep)
            
            kwargs.update(injected_kwargs)
            return func(*args, **kwargs)
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator
