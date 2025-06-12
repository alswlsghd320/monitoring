import asyncio
from typing import Optional, AsyncIterator, Iterator, Any, Dict, Tuple, List
from collections import defaultdict
from copy import deepcopy
import threading

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id
)
from langgraph.serde.base import SerializerProtocol


class LatestOnlyInMemorySaver(BaseCheckpointSaver):
    """
    thread_id별로 가장 최신의 checkpoint만 유지하는 동기 메모리 체크포인터.
    """
    
    def __init__(self, *, serde: Optional[SerializerProtocol] = None) -> None:
        super().__init__(serde=serde)
        self.storage: Dict[str, Tuple[Checkpoint, CheckpointMetadata, Optional[RunnableConfig]]] = {}
        self.writes: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = threading.Lock()
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, Any]] = None,
    ) -> RunnableConfig:
        """새 체크포인트를 저장하고 이전 체크포인트는 삭제"""
        with self._lock:
            thread_id = config["configurable"]["thread_id"]
            
            if checkpoint.get("id") is None:
                checkpoint["id"] = get_checkpoint_id()
            
            if thread_id in self.storage:
                old_checkpoint_id = self.storage[thread_id][0]["id"]
                if thread_id in self.writes and old_checkpoint_id in self.writes[thread_id]:
                    del self.writes[thread_id][old_checkpoint_id]
            
            parent_config = None
            if config.get("configurable", {}).get("checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": config["configurable"]["checkpoint_id"]
                    }
                }
            
            self.storage[thread_id] = (
                deepcopy(checkpoint),
                deepcopy(metadata),
                parent_config
            )
            
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint["id"],
                    "checkpoint_ns": config.get("configurable", {}).get("checkpoint_ns", ""),
                }
            }
    
    def put_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """중간 쓰기 작업 저장"""
        with self._lock:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = config["configurable"]["checkpoint_id"]
            checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
            
            if thread_id in self.storage:
                current_checkpoint_id = self.storage[thread_id][0]["id"]
                if checkpoint_id == current_checkpoint_id:
                    if checkpoint_id not in self.writes[thread_id]:
                        self.writes[thread_id][checkpoint_id] = {}
                    
                    for channel, value in writes:
                        key = (checkpoint_ns, task_id, channel)
                        self.writes[thread_id][checkpoint_id][key] = deepcopy(value)
    
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """체크포인트 튜플 조회"""
        with self._lock:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
            
            if thread_id not in self.storage:
                return None
            
            checkpoint, metadata, parent_config = self.storage[thread_id]
            
            if checkpoint_id and checkpoint["id"] != checkpoint_id:
                return None
            
            pending_writes = []
            checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
            
            if (thread_id in self.writes and 
                checkpoint["id"] in self.writes[thread_id]):
                for (ns, task_id, channel), value in self.writes[thread_id][checkpoint["id"]].items():
                    if ns == checkpoint_ns:
                        pending_writes.append((task_id, channel, value))
            
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint["id"],
                        "checkpoint_ns": checkpoint_ns,
                    }
                },
                checkpoint=deepcopy(checkpoint),
                metadata=deepcopy(metadata),
                parent_config=parent_config,
                pending_writes=pending_writes,
            )
    
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """체크포인트 목록 조회"""
        with self._lock:
            if config and "thread_id" in config.get("configurable", {}):
                thread_id = config["configurable"]["thread_id"]
                if thread_id in self.storage:
                    checkpoint, metadata, parent_config = self.storage[thread_id]
                    
                    if filter:
                        if not all(
                            metadata.get(key) == value 
                            for key, value in filter.items()
                        ):
                            return
                    
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_id": checkpoint["id"],
                                "checkpoint_ns": "",
                            }
                        },
                        checkpoint=deepcopy(checkpoint),
                        metadata=deepcopy(metadata),
                        parent_config=parent_config,
                        pending_writes=[],
                    )
            else:
                count = 0
                for thread_id, (checkpoint, metadata, parent_config) in self.storage.items():
                    if limit and count >= limit:
                        break
                    
                    if filter:
                        if not all(
                            metadata.get(key) == value 
                            for key, value in filter.items()
                        ):
                            continue
                    
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_id": checkpoint["id"],
                                "checkpoint_ns": "",
                            }
                        },
                        checkpoint=deepcopy(checkpoint),
                        metadata=deepcopy(metadata),
                        parent_config=parent_config,
                        pending_writes=[],
                    )
                    count += 1
    
    def get_memory_usage(self) -> Dict[str, int]:
        """현재 메모리 사용량 정보 반환"""
        with self._lock:
            return {
                "total_threads": len(self.storage),
                "total_checkpoints": len(self.storage),
                "writes_count": sum(
                    len(checkpoint_writes) 
                    for checkpoint_writes in self.writes.values()
                ),
            }


class AsyncLatestOnlyInMemorySaver(BaseCheckpointSaver):
    """
    thread_id별로 가장 최신의 checkpoint만 유지하는 비동기 메모리 체크포인터.
    """
    
    def __init__(self, *, serde: Optional[SerializerProtocol] = None) -> None:
        super().__init__(serde=serde)
        self.storage: Dict[str, Tuple[Checkpoint, CheckpointMetadata, Optional[RunnableConfig]]] = {}
        self.writes: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, Any]] = None,
    ) -> RunnableConfig:
        """비동기로 새 체크포인트를 저장하고 이전 체크포인트는 삭제"""
        async with self._lock:
            thread_id = config["configurable"]["thread_id"]
            
            if checkpoint.get("id") is None:
                checkpoint["id"] = get_checkpoint_id()
            
            if thread_id in self.storage:
                old_checkpoint_id = self.storage[thread_id][0]["id"]
                if thread_id in self.writes and old_checkpoint_id in self.writes[thread_id]:
                    del self.writes[thread_id][old_checkpoint_id]
            
            parent_config = None
            if config.get("configurable", {}).get("checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": config["configurable"]["checkpoint_id"]
                    }
                }
            
            self.storage[thread_id] = (
                deepcopy(checkpoint),
                deepcopy(metadata),
                parent_config
            )
            
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint["id"],
                    "checkpoint_ns": config.get("configurable", {}).get("checkpoint_ns", ""),
                }
            }
    
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """비동기로 중간 쓰기 작업 저장"""
        async with self._lock:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = config["configurable"]["checkpoint_id"]
            checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
            
            if thread_id in self.storage:
                current_checkpoint_id = self.storage[thread_id][0]["id"]
                if checkpoint_id == current_checkpoint_id:
                    if checkpoint_id not in self.writes[thread_id]:
                        self.writes[thread_id][checkpoint_id] = {}
                    
                    for channel, value in writes:
                        key = (checkpoint_ns, task_id, channel)
                        self.writes[thread_id][checkpoint_id][key] = deepcopy(value)
    
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """비동기로 체크포인트 튜플 조회"""
        async with self._lock:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
            
            if thread_id not in self.storage:
                return None
            
            checkpoint, metadata, parent_config = self.storage[thread_id]
            
            if checkpoint_id and checkpoint["id"] != checkpoint_id:
                return None
            
            pending_writes = []
            checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
            
            if (thread_id in self.writes and 
                checkpoint["id"] in self.writes[thread_id]):
                for (ns, task_id, channel), value in self.writes[thread_id][checkpoint["id"]].items():
                    if ns == checkpoint_ns:
                        pending_writes.append((task_id, channel, value))
            
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint["id"],
                        "checkpoint_ns": checkpoint_ns,
                    }
                },
                checkpoint=deepcopy(checkpoint),
                metadata=deepcopy(metadata),
                parent_config=parent_config,
                pending_writes=pending_writes,
            )
    
    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """비동기로 체크포인트 목록 조회"""
        async with self._lock:
            if config and "thread_id" in config.get("configurable", {}):
                thread_id = config["configurable"]["thread_id"]
                if thread_id in self.storage:
                    checkpoint, metadata, parent_config = self.storage[thread_id]
                    
                    if filter:
                        if not all(
                            metadata.get(key) == value 
                            for key, value in filter.items()
                        ):
                            return
                    
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_id": checkpoint["id"],
                                "checkpoint_ns": "",
                            }
                        },
                        checkpoint=deepcopy(checkpoint),
                        metadata=deepcopy(metadata),
                        parent_config=parent_config,
                        pending_writes=[],
                    )
            else:
                count = 0
                for thread_id, (checkpoint, metadata, parent_config) in self.storage.items():
                    if limit and count >= limit:
                        break
                    
                    if filter:
                        if not all(
                            metadata.get(key) == value 
                            for key, value in filter.items()
                        ):
                            continue
                    
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_id": checkpoint["id"],
                                "checkpoint_ns": "",
                            }
                        },
                        checkpoint=deepcopy(checkpoint),
                        metadata=deepcopy(metadata),
                        parent_config=parent_config,
                        pending_writes=[],
                    )
                    count += 1
    
    async def aget_memory_usage(self) -> Dict[str, int]:
        """비동기로 현재 메모리 사용량 정보 반환"""
        async with self._lock:
            return {
                "total_threads": len(self.storage),
                "total_checkpoints": len(self.storage),
                "writes_count": sum(
                    len(checkpoint_writes) 
                    for checkpoint_writes in self.writes.values()
                ),
            }
    
    # 동기 메서드들 - LangGraph 내부 호환성을 위해 threading.Lock 사용
    def __init_sync_lock(self):
        """동기 락 초기화 (필요시)"""
        if not hasattr(self, '_sync_lock'):
            self._sync_lock = threading.Lock()
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, Any]] = None,
    ) -> RunnableConfig:
        """동기 put 메서드"""
        self.__init_sync_lock()
        with self._sync_lock:
            thread_id = config["configurable"]["thread_id"]
            
            if checkpoint.get("id") is None:
                checkpoint["id"] = get_checkpoint_id()
            
            if thread_id in self.storage:
                old_checkpoint_id = self.storage[thread_id][0]["id"]
                if thread_id in self.writes and old_checkpoint_id in self.writes[thread_id]:
                    del self.writes[thread_id][old_checkpoint_id]
            
            parent_config = None
            if config.get("configurable", {}).get("checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": config["configurable"]["checkpoint_id"]
                    }
                }
            
            self.storage[thread_id] = (
                deepcopy(checkpoint),
                deepcopy(metadata),
                parent_config
            )
            
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint["id"],
                    "checkpoint_ns": config.get("configurable", {}).get("checkpoint_ns", ""),
                }
            }
    
    def put_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """동기 put_writes 메서드"""
        self.__init_sync_lock()
        with self._sync_lock:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = config["configurable"]["checkpoint_id"]
            checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
            
            if thread_id in self.storage:
                current_checkpoint_id = self.storage[thread_id][0]["id"]
                if checkpoint_id == current_checkpoint_id:
                    if checkpoint_id not in self.writes[thread_id]:
                        self.writes[thread_id][checkpoint_id] = {}
                    
                    for channel, value in writes:
                        key = (checkpoint_ns, task_id, channel)
                        self.writes[thread_id][checkpoint_id][key] = deepcopy(value)
    
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """동기 get_tuple 메서드"""
        self.__init_sync_lock()
        with self._sync_lock:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
            
            if thread_id not in self.storage:
                return None
            
            checkpoint, metadata, parent_config = self.storage[thread_id]
            
            if checkpoint_id and checkpoint["id"] != checkpoint_id:
                return None
            
            pending_writes = []
            checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
            
            if (thread_id in self.writes and 
                checkpoint["id"] in self.writes[thread_id]):
                for (ns, task_id, channel), value in self.writes[thread_id][checkpoint["id"]].items():
                    if ns == checkpoint_ns:
                        pending_writes.append((task_id, channel, value))
            
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint["id"],
                        "checkpoint_ns": checkpoint_ns,
                    }
                },
                checkpoint=deepcopy(checkpoint),
                metadata=deepcopy(metadata),
                parent_config=parent_config,
                pending_writes=pending_writes,
            )
    
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """동기 list 메서드"""
        self.__init_sync_lock()
        with self._sync_lock:
            if config and "thread_id" in config.get("configurable", {}):
                thread_id = config["configurable"]["thread_id"]
                if thread_id in self.storage:
                    checkpoint, metadata, parent_config = self.storage[thread_id]
                    
                    if filter:
                        if not all(
                            metadata.get(key) == value 
                            for key, value in filter.items()
                        ):
                            return
                    
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_id": checkpoint["id"],
                                "checkpoint_ns": "",
                            }
                        },
                        checkpoint=deepcopy(checkpoint),
                        metadata=deepcopy(metadata),
                        parent_config=parent_config,
                        pending_writes=[],
                    )
            else:
                count = 0
                for thread_id, (checkpoint, metadata, parent_config) in self.storage.items():
                    if limit and count >= limit:
                        break
                    
                    if filter:
                        if not all(
                            metadata.get(key) == value 
                            for key, value in filter.items()
                        ):
                            continue
                    
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_id": checkpoint["id"],
                                "checkpoint_ns": "",
                            }
                        },
                        checkpoint=deepcopy(checkpoint),
                        metadata=deepcopy(metadata),
                        parent_config=parent_config,
                        pending_writes=[],
                    )
                    count += 1


# 사용 예제
if __name__ == "__main__":
    import asyncio
    from langgraph.graph import StateGraph, START
    from typing import TypedDict, Annotated, Sequence
    from langchain_core.messages import BaseMessage, HumanMessage
    import operator
    
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        counter: int
    
    # 동기 사용 예제
    def sync_example():
        def increment_counter(state: AgentState) -> AgentState:
            return {"counter": state["counter"] + 1}
        
        builder = StateGraph(AgentState)
        builder.add_node("increment", increment_counter)
        builder.add_edge(START, "increment")
        
        memory = LatestOnlyInMemorySaver()
        graph = builder.compile(checkpointer=memory)
        
        config = {"configurable": {"thread_id": "sync_thread"}}
        for i in range(3):
            result = graph.invoke({"messages": [], "counter": i}, config)
            print(f"Sync Run {i+1}: counter = {result['counter']}")
        
        print(f"Sync Memory usage: {memory.get_memory_usage()}")
    
    # 비동기 사용 예제
    async def async_example():
        async def async_increment_counter(state: AgentState) -> AgentState:
            await asyncio.sleep(0.1)
            return {"counter": state["counter"] + 1}
        
        builder = StateGraph(AgentState)
        builder.add_node("increment", async_increment_counter)
        builder.add_edge(START, "increment")
        
        memory = AsyncLatestOnlyInMemorySaver()
        graph = builder.compile(checkpointer=memory)
        
        config = {"configurable": {"thread_id": "async_thread"}}
        for i in range(3):
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=f"Run {i}")], "counter": i}, 
                config
            )
            print(f"Async Run {i+1}: counter = {result['counter']}")
        
        usage = await memory.aget_memory_usage()
        print(f"Async Memory usage: {usage}")
    
    # 실행
    print("=== Sync Example ===")
    sync_example()
    
    print("\n=== Async Example ===")
    asyncio.run(async_example())
