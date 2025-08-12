"""Dependency injection container for the ML pipeline."""

import inspect
from typing import Any, Callable, Dict, Optional, Type, TypeVar, get_type_hints
from functools import wraps
import threading

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class DIContainer:
    """Dependency injection container with singleton and factory support."""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[[], Any]] = {}
        self._lock = threading.Lock()
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a singleton service."""
        with self._lock:
            self._services[interface] = implementation
            logger.debug(f"Registered singleton {interface.__name__} -> {implementation.__name__}")
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory for creating instances."""
        with self._lock:
            self._factories[interface] = factory
            logger.debug(f"Registered factory for {interface.__name__}")
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a specific instance."""
        with self._lock:
            self._singletons[interface] = instance
            logger.debug(f"Registered instance for {interface.__name__}")
    
    def get(self, interface: Type[T]) -> T:
        """Get an instance of the specified interface."""
        with self._lock:
            # Check if we have a pre-created instance
            if interface in self._singletons:
                return self._singletons[interface]
            
            # Check if we have a factory
            if interface in self._factories:
                instance = self._factories[interface]()
                logger.debug(f"Created instance via factory for {interface.__name__}")
                return instance
            
            # Check if we have a registered service
            if interface in self._services:
                implementation = self._services[interface]
                # Create singleton instance
                instance = self._create_instance(implementation)
                self._singletons[interface] = instance
                logger.debug(f"Created singleton instance for {interface.__name__}")
                return instance
            
            raise ValueError(f"No registration found for {interface.__name__}")
    
    def _create_instance(self, cls: Type[T]) -> T:
        """Create an instance with dependency injection."""
        try:
            # Get constructor signature
            sig = inspect.signature(cls.__init__)
            params = {}
            
            # Resolve dependencies
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                # Try to resolve from type hints
                if param.annotation != inspect.Parameter.empty:
                    try:
                        dependency = self.get(param.annotation)
                        params[param_name] = dependency
                    except ValueError:
                        # If no dependency found and no default, this is an error
                        if param.default == inspect.Parameter.empty:
                            logger.warning(f"Could not resolve dependency {param.annotation} for {cls.__name__}")
                            raise ValueError(f"Could not resolve dependency {param.annotation}")
            
            return cls(**params)
            
        except Exception as e:
            logger.error(f"Failed to create instance of {cls.__name__}: {e}")
            raise
    
    def clear(self) -> None:
        """Clear all registrations."""
        with self._lock:
            self._services.clear()
            self._singletons.clear()
            self._factories.clear()
            logger.debug("Cleared all DI registrations")
    
    def list_registrations(self) -> Dict[str, str]:
        """List all current registrations."""
        with self._lock:
            registrations = {}
            
            for interface, impl in self._services.items():
                registrations[interface.__name__] = f"Singleton: {impl.__name__}"
            
            for interface in self._factories:
                registrations[interface.__name__] = "Factory"
            
            for interface in self._singletons:
                registrations[interface.__name__] = "Instance"
            
            return registrations


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get the global DI container."""
    return _container


def inject(interface: Type[T]) -> T:
    """Inject a dependency."""
    return _container.get(interface)


def injectable(cls: Type[T]) -> Type[T]:
    """Decorator to mark a class as injectable."""
    original_init = cls.__init__
    
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # Auto-inject dependencies if not provided
        sig = inspect.signature(original_init)
        bound_args = sig.bind_partial(self, *args, **kwargs)
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param_name not in bound_args.arguments:
                if param.annotation != inspect.Parameter.empty:
                    try:
                        dependency = _container.get(param.annotation)
                        bound_args.arguments[param_name] = dependency
                    except ValueError:
                        if param.default == inspect.Parameter.empty:
                            logger.warning(f"Could not auto-inject {param.annotation} for {cls.__name__}")
        
        original_init(*bound_args.args, **bound_args.kwargs)
    
    cls.__init__ = new_init
    return cls


def register_singleton(interface: Type[T], implementation: Type[T]):
    """Decorator to register a singleton."""
    def decorator(cls):
        _container.register_singleton(interface, implementation or cls)
        return cls
    return decorator


def register_factory(interface: Type[T]):
    """Decorator to register a factory function."""
    def decorator(factory_func):
        _container.register_factory(interface, factory_func)
        return factory_func
    return decorator


def auto_wire(func: Callable) -> Callable:
    """Decorator to auto-wire function parameters with DI."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        
        # Get type hints
        type_hints = get_type_hints(func)
        
        for param_name, param in sig.parameters.items():
            if param_name not in bound_args.arguments:
                if param_name in type_hints:
                    param_type = type_hints[param_name]
                    try:
                        dependency = _container.get(param_type)
                        bound_args.arguments[param_name] = dependency
                    except ValueError:
                        if param.default == inspect.Parameter.empty:
                            logger.warning(f"Could not auto-wire {param_type} for parameter {param_name}")
        
        return func(*bound_args.args, **bound_args.kwargs)
    
    return wrapper


class DIContextManager:
    """Context manager for DI configuration."""
    
    def __init__(self):
        self._original_state = {}
    
    def __enter__(self):
        # Save current state
        self._original_state = {
            'services': _container._services.copy(),
            'singletons': _container._singletons.copy(),
            'factories': _container._factories.copy()
        }
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        _container._services = self._original_state['services']
        _container._singletons = self._original_state['singletons']
        _container._factories = self._original_state['factories']
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> 'DIContextManager':
        """Register a singleton in this context."""
        _container.register_singleton(interface, implementation)
        return self
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> 'DIContextManager':
        """Register a factory in this context."""
        _container.register_factory(interface, factory)
        return self
    
    def register_instance(self, interface: Type[T], instance: T) -> 'DIContextManager':
        """Register an instance in this context."""
        _container.register_instance(interface, instance)
        return self


def di_context() -> DIContextManager:
    """Create a DI context manager for temporary registrations."""
    return DIContextManager()


# Configuration helper
class DIConfiguration:
    """Helper class for configuring DI container."""
    
    @staticmethod
    def configure_default_services():
        """Configure default service registrations."""
        from utils.logger import get_logger
        from core.interfaces import ILogger
        
        # Register default implementations
        # This would be expanded with actual implementations
        logger.info("Configuring default DI services")
    
    @staticmethod
    def configure_from_config(config: Dict[str, Any]):
        """Configure DI container from configuration."""
        di_config = config.get('dependency_injection', {})
        
        for service_name, service_config in di_config.items():
            interface_path = service_config.get('interface')
            implementation_path = service_config.get('implementation')
            registration_type = service_config.get('type', 'singleton')
            
            if interface_path and implementation_path:
                try:
                    # Dynamic import and registration
                    interface_cls = _import_class(interface_path)
                    implementation_cls = _import_class(implementation_path)
                    
                    if registration_type == 'singleton':
                        _container.register_singleton(interface_cls, implementation_cls)
                    elif registration_type == 'factory':
                        # For factories, implementation should be a factory function
                        _container.register_factory(interface_cls, implementation_cls)
                    
                    logger.info(f"Configured DI: {interface_path} -> {implementation_path} ({registration_type})")
                    
                except Exception as e:
                    logger.error(f"Failed to configure DI for {service_name}: {e}")


def _import_class(class_path: str) -> Type:
    """Import a class from a module path."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)