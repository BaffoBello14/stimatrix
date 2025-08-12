"""Performance optimization utilities for the ML pipeline."""

import time
import psutil
import gc
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from functools import wraps
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pickle
import hashlib
import json
from datetime import datetime, timedelta

from utils.logger import get_logger
from utils.exceptions import MemoryError as PipelineMemoryError

logger = get_logger(__name__)


class MemoryMonitor:
    """Monitor and manage memory usage."""
    
    def __init__(self, max_memory_mb: Optional[int] = None):
        """Initialize with optional maximum memory limit in MB."""
        self.max_memory_mb = max_memory_mb or (psutil.virtual_memory().total // (1024 * 1024) * 0.8)
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": self.process.memory_percent(),
            "available_mb": virtual_memory.available / (1024 * 1024),
            "total_mb": virtual_memory.total / (1024 * 1024),
            "system_percent": virtual_memory.percent
        }
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds the limit."""
        current_usage = self.get_memory_usage()
        return current_usage["rss_mb"] > self.max_memory_mb
    
    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        before_count = len(gc.get_objects())
        collected = gc.collect()
        after_count = len(gc.get_objects())
        
        result = {
            "collected": collected,
            "objects_before": before_count,
            "objects_after": after_count,
            "objects_freed": before_count - after_count
        }
        
        logger.debug(f"GC: {result}")
        return result


class DataFrameChunker:
    """Efficiently process large DataFrames in chunks."""
    
    def __init__(self, chunk_size: Optional[int] = None, memory_limit_mb: Optional[int] = None):
        """Initialize with chunk size or memory-based chunking."""
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb or 500  # Default 500MB per chunk
        self.memory_monitor = MemoryMonitor()
    
    def calculate_optimal_chunk_size(self, df: pd.DataFrame) -> int:
        """Calculate optimal chunk size based on DataFrame size and memory."""
        if self.chunk_size:
            return self.chunk_size
        
        # Estimate memory per row
        memory_per_row = df.memory_usage(deep=True).sum() / len(df)
        
        # Calculate chunk size based on memory limit
        max_rows_per_chunk = int((self.memory_limit_mb * 1024 * 1024) / memory_per_row)
        
        # Ensure minimum viable chunk size
        min_chunk_size = max(1000, len(df) // 100)  # At least 1000 rows or 1% of data
        optimal_chunk_size = max(min_chunk_size, max_rows_per_chunk)
        
        logger.info(f"Calculated optimal chunk size: {optimal_chunk_size} rows")
        return optimal_chunk_size
    
    def chunk_dataframe(self, df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """Generate DataFrame chunks."""
        chunk_size = self.calculate_optimal_chunk_size(df)
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            
            # Monitor memory usage
            memory_stats = self.memory_monitor.get_memory_usage()
            if memory_stats["rss_mb"] > self.memory_limit_mb * 2:
                logger.warning(f"High memory usage detected: {memory_stats['rss_mb']:.1f}MB")
                self.memory_monitor.force_gc()
            
            yield chunk
    
    def process_in_chunks(self, 
                         df: pd.DataFrame, 
                         processor_func: Callable[[pd.DataFrame], pd.DataFrame],
                         combine_func: Optional[Callable[[List[pd.DataFrame]], pd.DataFrame]] = None) -> pd.DataFrame:
        """Process DataFrame in chunks and combine results."""
        if combine_func is None:
            combine_func = pd.concat
        
        results = []
        total_chunks = (len(df) + self.calculate_optimal_chunk_size(df) - 1) // self.calculate_optimal_chunk_size(df)
        
        logger.info(f"Processing {len(df)} rows in {total_chunks} chunks")
        
        for i, chunk in enumerate(self.chunk_dataframe(df)):
            try:
                result = processor_func(chunk)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{total_chunks} chunks")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                raise
        
        logger.info("Combining chunk results")
        return combine_func(results)


class ParallelProcessor:
    """Parallel processing utilities with automatic resource management."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = True):
        """Initialize with worker count and processing type."""
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Limit to 8 to avoid overwhelming
        self.use_processes = use_processes
    
    def process_parallel(self, 
                        items: List[Any], 
                        processor_func: Callable[[Any], Any],
                        **kwargs) -> List[Any]:
        """Process items in parallel."""
        if len(items) <= 1:
            return [processor_func(item, **kwargs) for item in items]
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        logger.info(f"Processing {len(items)} items with {self.max_workers} {'processes' if self.use_processes else 'threads'}")
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(processor_func, item, **kwargs): item 
                for item in items
            }
            
            results = []
            completed = 0
            
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % max(1, len(items) // 10) == 0:
                        logger.info(f"Completed {completed}/{len(items)} tasks")
                        
                except Exception as e:
                    item = future_to_item[future]
                    logger.error(f"Error processing item {item}: {e}")
                    raise
        
        return results
    
    def process_dataframe_parallel(self, 
                                  df: pd.DataFrame,
                                  processor_func: Callable[[pd.DataFrame], pd.DataFrame],
                                  chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Process DataFrame chunks in parallel."""
        chunker = DataFrameChunker(chunk_size)
        chunks = list(chunker.chunk_dataframe(df))
        
        if len(chunks) <= 1:
            return processor_func(df)
        
        # Process chunks in parallel
        processed_chunks = self.process_parallel(chunks, processor_func)
        
        # Combine results
        logger.info("Combining parallel processing results")
        return pd.concat(processed_chunks, ignore_index=True)


class CacheManager:
    """Intelligent caching system for expensive operations."""
    
    def __init__(self, cache_dir: Union[str, Path] = "cache", max_size_mb: int = 1000):
        """Initialize cache manager."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_mb = max_size_mb
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create a unique key based on function name and arguments
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is expired
        if key in self.metadata:
            created_time = datetime.fromisoformat(self.metadata[key]['created'])
            if datetime.now() - created_time > timedelta(days=7):  # 7 day expiry
                logger.debug(f"Cache expired for key {key}")
                self.delete(key)
                return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached value for key {key}: {e}")
            self.delete(key)
            return None
    
    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set cached value."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Update metadata
            file_size = cache_file.stat().st_size
            self.metadata[key] = {
                'created': datetime.now().isoformat(),
                'size_bytes': file_size,
                'metadata': metadata or {}
            }
            
            self._save_metadata()
            self._cleanup_if_needed()
            
            logger.debug(f"Cached value for key {key} ({file_size / 1024:.1f} KB)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache value for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete cached value for key {key}: {e}")
            return False
    
    def _cleanup_if_needed(self):
        """Clean up cache if it exceeds size limit."""
        total_size = sum(item['size_bytes'] for item in self.metadata.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            logger.info(f"Cache size ({total_size / 1024 / 1024:.1f}MB) exceeds limit, cleaning up")
            
            # Sort by creation time (oldest first)
            sorted_items = sorted(
                self.metadata.items(),
                key=lambda x: x[1]['created']
            )
            
            # Remove oldest items until under limit
            for key, metadata in sorted_items:
                if total_size <= max_size_bytes:
                    break
                
                self.delete(key)
                total_size -= metadata['size_bytes']
                logger.debug(f"Removed cached item {key}")
    
    def clear(self) -> bool:
        """Clear all cache."""
        try:
            for key in list(self.metadata.keys()):
                self.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


def cache_result(cache_manager: Optional[CacheManager] = None, ttl_days: int = 7):
    """Decorator to cache function results."""
    if cache_manager is None:
        cache_manager = CacheManager()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache_manager._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}, executing")
            result = func(*args, **kwargs)
            
            # Cache the result
            cache_manager.set(key, result, {
                'function': func.__name__,
                'ttl_days': ttl_days
            })
            
            return result
        
        return wrapper
    return decorator


def memory_efficient(func: Callable) -> Callable:
    """Decorator to monitor and optimize memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        
        # Log initial memory state
        initial_memory = monitor.get_memory_usage()
        logger.debug(f"Starting {func.__name__} with {initial_memory['rss_mb']:.1f}MB memory")
        
        try:
            result = func(*args, **kwargs)
            
            # Check for memory leaks
            final_memory = monitor.get_memory_usage()
            memory_diff = final_memory['rss_mb'] - initial_memory['rss_mb']
            
            if memory_diff > 100:  # Alert if function used more than 100MB
                logger.warning(f"{func.__name__} used {memory_diff:.1f}MB additional memory")
            
            # Force GC if memory usage is high
            if final_memory['rss_mb'] > 1000:  # More than 1GB
                monitor.force_gc()
            
            return result
            
        except Exception as e:
            # Log memory state on error
            error_memory = monitor.get_memory_usage()
            logger.error(f"Error in {func.__name__} at {error_memory['rss_mb']:.1f}MB memory: {e}")
            raise
    
    return wrapper


def time_it(func: Callable) -> Callable:
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper


class DataFrameOptimizer:
    """Optimize DataFrame memory usage and operations."""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory usage."""
        logger.info("Optimizing DataFrame data types")
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type != 'object':
                # Optimize numeric columns
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                
                if str(col_type)[:3] == 'int':
                    # Integer optimization
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)
                
                elif str(col_type)[:5] == 'float':
                    # Float optimization
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized_df[col] = optimized_df[col].astype(np.float32)
            else:
                # String/object optimization
                num_unique_values = len(optimized_df[col].unique())
                num_total_values = len(optimized_df[col])
                
                if num_unique_values / num_total_values < 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')
        
        new_memory = optimized_df.memory_usage(deep=True).sum()
        reduction = (original_memory - new_memory) / original_memory * 100
        
        logger.info(f"Memory usage reduced by {reduction:.1f}% ({original_memory / 1024 / 1024:.1f}MB -> {new_memory / 1024 / 1024:.1f}MB)")
        
        return optimized_df
    
    @staticmethod
    def reduce_memory_usage(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """Reduce DataFrame memory usage through various optimizations."""
        optimized_df = DataFrameOptimizer.optimize_dtypes(df)
        
        if aggressive:
            # More aggressive optimizations
            logger.info("Applying aggressive memory optimizations")
            
            # Convert sparse columns
            for col in optimized_df.columns:
                if optimized_df[col].isnull().mean() > 0.9:  # More than 90% null
                    optimized_df[col] = optimized_df[col].astype(pd.SparseDtype(optimized_df[col].dtype))
        
        return optimized_df


# Global instances for convenience
_global_cache = CacheManager()
_global_chunker = DataFrameChunker()
_global_parallel = ParallelProcessor()


def get_cache() -> CacheManager:
    """Get global cache manager."""
    return _global_cache


def get_chunker() -> DataFrameChunker:
    """Get global DataFrame chunker."""
    return _global_chunker


def get_parallel_processor() -> ParallelProcessor:
    """Get global parallel processor."""
    return _global_parallel