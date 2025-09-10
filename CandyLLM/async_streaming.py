"""
🍭 CandyLLM Async & Streaming Support
High-performance async operations and real-time streaming
"""

import asyncio
import time
import threading
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Any, Optional, Union, Callable, AsyncIterator
from dataclasses import dataclass
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import json
import uuid

@dataclass
class StreamChunk:
    """Represents a streaming response chunk"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    is_final: bool = False
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class StreamSession:
    """Represents an active streaming session"""
    session_id: str
    model_name: str
    provider: str
    started_at: float
    total_chunks: int = 0
    total_content: str = ""
    status: str = "active"  # active, completed, error
    
class AsyncLLMWrapper:
    """
    Async wrapper around LLMWrapper for high-performance operations
    """
    
    def __init__(self, base_wrapper):
        self.base_wrapper = base_wrapper
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.active_streams: Dict[str, StreamSession] = {}
    
    async def answer_async(self, prompt: str, **kwargs) -> str:
        """Async version of answer method"""
        loop = asyncio.get_event_loop()
        
        # Run the sync answer method in thread pool
        result = await loop.run_in_executor(
            self.executor,
            lambda: self.base_wrapper.answer(prompt, **kwargs)
        )
        
        return result
    
    async def stream_async(self, prompt: str, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """
        Async streaming generator
        
        Yields StreamChunk objects with content and metadata
        """
        session_id = str(uuid.uuid4())
        session = StreamSession(
            session_id=session_id,
            model_name=kwargs.get('model_name', self.base_wrapper.modelName),
            provider=kwargs.get('provider', getattr(self.base_wrapper, 'source', 'unknown')),
            started_at=time.time()
        )
        
        self.active_streams[session_id] = session
        
        try:
            # Check if the underlying wrapper supports streaming
            if hasattr(self.base_wrapper, 'stream'):
                # Use native streaming
                async for chunk in self._native_stream(prompt, session, **kwargs):
                    yield chunk
            else:
                # Simulate streaming by chunking the response
                async for chunk in self._simulated_stream(prompt, session, **kwargs):
                    yield chunk
                    
        except Exception as e:
            session.status = "error"
            yield StreamChunk(
                chunk_id=f"{session_id}_error",
                content=f"Error: {e}",
                metadata={"error": True, "session_id": session_id},
                is_final=True
            )
        finally:
            session.status = "completed"
            if session_id in self.active_streams:
                del self.active_streams[session_id]
    
    async def _native_stream(self, prompt: str, session: StreamSession, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Handle native streaming from the underlying wrapper"""
        chunk_index = 0
        
        try:
            # Check if stream method is async
            stream_method = self.base_wrapper.stream
            
            if asyncio.iscoroutinefunction(stream_method):
                # Native async streaming
                async for content in stream_method(prompt, **kwargs):
                    chunk = StreamChunk(
                        chunk_id=f"{session.session_id}_{chunk_index}",
                        content=content,
                        metadata={
                            "session_id": session.session_id,
                            "chunk_index": chunk_index,
                            "model": session.model_name,
                            "provider": session.provider
                        }
                    )
                    
                    session.total_chunks += 1
                    session.total_content += content
                    chunk_index += 1
                    
                    yield chunk
            else:
                # Sync streaming - run in executor
                loop = asyncio.get_event_loop()
                
                # Create a queue for communication between threads
                chunk_queue = Queue()
                
                def sync_stream():
                    try:
                        for content in stream_method(prompt, **kwargs):
                            chunk_queue.put(('chunk', content))
                        chunk_queue.put(('done', None))
                    except Exception as e:
                        chunk_queue.put(('error', e))
                
                # Start streaming in background thread
                future = loop.run_in_executor(self.executor, sync_stream)
                
                # Yield chunks as they arrive
                while True:
                    try:
                        # Check queue with timeout to allow cancellation
                        event_type, content = await loop.run_in_executor(
                            self.executor,
                            lambda: chunk_queue.get(timeout=0.1)
                        )
                        
                        if event_type == 'done':
                            break
                        elif event_type == 'error':
                            raise content
                        elif event_type == 'chunk':
                            chunk = StreamChunk(
                                chunk_id=f"{session.session_id}_{chunk_index}",
                                content=content,
                                metadata={
                                    "session_id": session.session_id,
                                    "chunk_index": chunk_index,
                                    "model": session.model_name,
                                    "provider": session.provider
                                }
                            )
                            
                            session.total_chunks += 1
                            session.total_content += content
                            chunk_index += 1
                            
                            yield chunk
                            
                    except Empty:
                        # No new chunks, check if we should continue
                        if future.done():
                            break
                        continue
                
                # Wait for the background task to complete
                await future
                
        except Exception as e:
            raise e
        
        # Send final chunk
        final_chunk = StreamChunk(
            chunk_id=f"{session.session_id}_final",
            content="",
            metadata={
                "session_id": session.session_id,
                "total_chunks": session.total_chunks,
                "total_content_length": len(session.total_content),
                "duration": time.time() - session.started_at
            },
            is_final=True
        )
        
        yield final_chunk
    
    async def _simulated_stream(self, prompt: str, session: StreamSession, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Simulate streaming by chunking a complete response"""
        loop = asyncio.get_event_loop()
        
        # Get complete response
        full_response = await loop.run_in_executor(
            self.executor,
            lambda: self.base_wrapper.answer(prompt, **kwargs)
        )
        
        # Split into chunks and stream
        chunk_size = kwargs.get('chunk_size', 50)  # Characters per chunk
        words = full_response.split()
        current_chunk = ""
        chunk_index = 0
        
        for word in words:
            current_chunk += word + " "
            
            if len(current_chunk) >= chunk_size:
                chunk = StreamChunk(
                    chunk_id=f"{session.session_id}_{chunk_index}",
                    content=current_chunk.strip(),
                    metadata={
                        "session_id": session.session_id,
                        "chunk_index": chunk_index,
                        "model": session.model_name,
                        "provider": session.provider,
                        "simulated": True
                    }
                )
                
                session.total_chunks += 1
                session.total_content += current_chunk.strip()
                chunk_index += 1
                
                yield chunk
                
                current_chunk = ""
                
                # Add small delay to simulate real streaming
                await asyncio.sleep(0.1)
        
        # Send remaining content
        if current_chunk.strip():
            chunk = StreamChunk(
                chunk_id=f"{session.session_id}_{chunk_index}",
                content=current_chunk.strip(),
                metadata={
                    "session_id": session.session_id,
                    "chunk_index": chunk_index,
                    "model": session.model_name,
                    "provider": session.provider,
                    "simulated": True
                }
            )
            
            session.total_chunks += 1
            session.total_content += current_chunk.strip()
            
            yield chunk
        
        # Send final chunk
        final_chunk = StreamChunk(
            chunk_id=f"{session.session_id}_final",
            content="",
            metadata={
                "session_id": session.session_id,
                "total_chunks": session.total_chunks,
                "total_content_length": len(session.total_content),
                "duration": time.time() - session.started_at,
                "simulated": True
            },
            is_final=True
        )
        
        yield final_chunk
    
    async def batch_process(self, prompts: List[str], **kwargs) -> List[str]:
        """Process multiple prompts concurrently"""
        # Create tasks for all prompts
        tasks = [
            self.answer_async(prompt, **kwargs)
            for prompt in prompts
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(f"Error processing prompt {i}: {result}")
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def parallel_stream(self, prompts: List[str], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream responses from multiple prompts in parallel"""
        # Create stream tasks
        stream_tasks = {}
        for i, prompt in enumerate(prompts):
            stream_tasks[i] = self.stream_async(prompt, **kwargs)
        
        # Track active streams
        active_streams = set(stream_tasks.keys())
        
        while active_streams:
            # Create tasks to get next chunk from each active stream
            next_chunk_tasks = {}
            for stream_id in list(active_streams):
                try:
                    next_chunk_tasks[stream_id] = asyncio.create_task(
                        stream_tasks[stream_id].__anext__()
                    )
                except StopAsyncIteration:
                    active_streams.remove(stream_id)
                    continue
            
            if not next_chunk_tasks:
                break
            
            # Wait for first chunk to arrive
            done, pending = await asyncio.wait(
                next_chunk_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Process completed chunks
            for task in done:
                # Find which stream this chunk came from
                stream_id = None
                for sid, stask in next_chunk_tasks.items():
                    if stask == task:
                        stream_id = sid
                        break
                
                if stream_id is not None:
                    try:
                        chunk = await task
                        
                        # Yield chunk with stream identification
                        yield {
                            "stream_id": stream_id,
                            "prompt_index": stream_id,
                            "chunk": chunk,
                            "prompt": prompts[stream_id]
                        }
                        
                        # If this was the final chunk, remove from active streams
                        if chunk.is_final:
                            active_streams.discard(stream_id)
                        
                    except StopAsyncIteration:
                        # Stream ended
                        active_streams.discard(stream_id)
                    except Exception as e:
                        # Stream error
                        active_streams.discard(stream_id)
                        yield {
                            "stream_id": stream_id,
                            "prompt_index": stream_id,
                            "error": str(e),
                            "prompt": prompts[stream_id]
                        }
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get statistics about active streams"""
        return {
            "active_streams": len(self.active_streams),
            "sessions": {
                session_id: {
                    "model": session.model_name,
                    "provider": session.provider,
                    "duration": time.time() - session.started_at,
                    "chunks": session.total_chunks,
                    "content_length": len(session.total_content),
                    "status": session.status
                }
                for session_id, session in self.active_streams.items()
            }
        }

class StreamingQueue:
    """
    Queue-based streaming for handling high-throughput scenarios
    """
    
    def __init__(self, max_size: int = 1000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.subscribers: Dict[str, Callable] = {}
        self.running = False
        self.stats = {
            "messages_processed": 0,
            "messages_dropped": 0,
            "subscribers": 0
        }
    
    async def start(self):
        """Start the streaming queue processor"""
        self.running = True
        asyncio.create_task(self._process_queue())
    
    async def stop(self):
        """Stop the streaming queue processor"""
        self.running = False
    
    async def push(self, message: Dict[str, Any]) -> bool:
        """Push a message to the queue"""
        try:
            await self.queue.put(message)
            return True
        except asyncio.QueueFull:
            self.stats["messages_dropped"] += 1
            return False
    
    def subscribe(self, subscriber_id: str, callback: Callable) -> None:
        """Subscribe to receive messages"""
        self.subscribers[subscriber_id] = callback
        self.stats["subscribers"] = len(self.subscribers)
    
    def unsubscribe(self, subscriber_id: str) -> None:
        """Unsubscribe from receiving messages"""
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]
            self.stats["subscribers"] = len(self.subscribers)
    
    async def _process_queue(self):
        """Process messages from the queue"""
        while self.running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Send to all subscribers
                for subscriber_id, callback in list(self.subscribers.items()):
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        print(f"Error in subscriber {subscriber_id}: {e}")
                
                self.stats["messages_processed"] += 1
                
            except asyncio.TimeoutError:
                # No message received, continue
                continue
            except Exception as e:
                print(f"Error processing queue: {e}")
                await asyncio.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self.stats,
            "queue_size": self.queue.qsize(),
            "running": self.running
        }

class AsyncPoolManager:
    """
    Manages pools of async LLM wrappers for load distribution
    """
    
    def __init__(self):
        self.pools: Dict[str, List[AsyncLLMWrapper]] = {}
        self.current_index: Dict[str, int] = {}
        self.lock = asyncio.Lock()
    
    async def create_pool(self, pool_name: str, wrapper_configs: List[Dict], pool_size: int = 3):
        """Create a pool of async wrappers"""
        async with self.lock:
            if pool_name in self.pools:
                raise ValueError(f"Pool {pool_name} already exists")
            
            self.pools[pool_name] = []
            self.current_index[pool_name] = 0
            
            # Create wrappers based on configs
            for config in wrapper_configs[:pool_size]:
                from . import LLMWrapper  # Import here to avoid circular imports
                
                base_wrapper = LLMWrapper(**config)
                async_wrapper = AsyncLLMWrapper(base_wrapper)
                self.pools[pool_name].append(async_wrapper)
    
    async def get_wrapper(self, pool_name: str) -> Optional[AsyncLLMWrapper]:
        """Get a wrapper from the pool using round-robin"""
        async with self.lock:
            if pool_name not in self.pools or not self.pools[pool_name]:
                return None
            
            pool = self.pools[pool_name]
            wrapper = pool[self.current_index[pool_name]]
            
            # Round-robin to next wrapper
            self.current_index[pool_name] = (self.current_index[pool_name] + 1) % len(pool)
            
            return wrapper
    
    async def process_with_pool(self, pool_name: str, prompt: str, **kwargs) -> str:
        """Process a prompt using a wrapper from the pool"""
        wrapper = await self.get_wrapper(pool_name)
        if not wrapper:
            raise ValueError(f"No wrappers available in pool {pool_name}")
        
        return await wrapper.answer_async(prompt, **kwargs)
    
    async def stream_with_pool(self, pool_name: str, prompt: str, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream using a wrapper from the pool"""
        wrapper = await self.get_wrapper(pool_name)
        if not wrapper:
            raise ValueError(f"No wrappers available in pool {pool_name}")
        
        async for chunk in wrapper.stream_async(prompt, **kwargs):
            yield chunk
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        stats = {}
        for pool_name, pool in self.pools.items():
            stats[pool_name] = {
                "size": len(pool),
                "current_index": self.current_index.get(pool_name, 0),
                "wrappers": [
                    wrapper.get_stream_stats()
                    for wrapper in pool
                ]
            }
        return stats

# Global instances
_streaming_queue = None
_pool_manager = None

def get_streaming_queue() -> StreamingQueue:
    """Get global streaming queue instance"""
    global _streaming_queue
    if _streaming_queue is None:
        _streaming_queue = StreamingQueue()
    return _streaming_queue

def get_pool_manager() -> AsyncPoolManager:
    """Get global pool manager instance"""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = AsyncPoolManager()
    return _pool_manager

# Convenience functions
def create_async_wrapper(base_wrapper) -> AsyncLLMWrapper:
    """Create an async wrapper from a base LLMWrapper"""
    return AsyncLLMWrapper(base_wrapper)

async def async_answer(wrapper, prompt: str, **kwargs) -> str:
    """Convenience function for async answer"""
    if isinstance(wrapper, AsyncLLMWrapper):
        return await wrapper.answer_async(prompt, **kwargs)
    else:
        async_wrapper = AsyncLLMWrapper(wrapper)
        return await async_wrapper.answer_async(prompt, **kwargs)

async def async_stream(wrapper, prompt: str, **kwargs) -> AsyncGenerator[StreamChunk, None]:
    """Convenience function for async streaming"""
    if isinstance(wrapper, AsyncLLMWrapper):
        async for chunk in wrapper.stream_async(prompt, **kwargs):
            yield chunk
    else:
        async_wrapper = AsyncLLMWrapper(wrapper)
        async for chunk in async_wrapper.stream_async(prompt, **kwargs):
            yield chunk
