# Code Patterns in vLLM-Omni

## Async Function Complexity

**Guidelines:**
- Each `await` is a potential failure point and state transition
- Consider splitting long async chains into smaller helpers
- Watch for race conditions in complex async flows
- Use `return_exceptions=True` for parallel operations

**Good:**
```python
async def process_batch(requests: list[Request]) -> list[Response]:
    validated = [_validate_request(r) for r in requests]
    results = await asyncio.gather(
        *[_process_single(r) for r in validated],
        return_exceptions=True
    )
    return [_handle_result(r) for r in results]
```

**Bad:**
```python
async def process_batch(requests: list[Request]) -> list[Response]:
    results = []
    for r in requests:
        results.append(await _process_single(r))  # Sequential, no error handling
    return results
```

---

## Distributed Execution Patterns

**When complexity is justified:**
- Tensor parallelism synchronization across workers
- Pipeline parallelism stage coordination
- Distributed KV cache management
- Multi-node communication patterns

**Critical questions:**
- Is distributed complexity isolated from business logic?
- Are distributed failure modes handled?
- Is there a clear fallback for single-device execution?

**Pattern:**
```python
def get_world_size() -> int:
    """Returns 1 if not in distributed mode."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
```

---

## KV Cache Management

**Guidelines:**
- Ensure clear separation between allocation, update, and cleanup
- Document invariants and assumptions
- Use type system to enforce valid states
- Always handle cleanup in error paths

**Pattern:**
```python
class KVCache:
    def allocate(self, num_blocks: int) -> None:
        assert self._state == CacheState.UNINITIALIZED
        self._blocks = [Block() for _ in range(num_blocks)]
        self._state = CacheState.ALLOCATED

    def update(self, block_id: int, kv_data: Tensor) -> None:
        assert self._state == CacheState.ALLOCATED
        self._blocks[block_id].update(kv_data)

    def free(self) -> None:
        self._blocks = []
        self._state = CacheState.UNINITIALIZED
```

---

## Input Validation Pattern

**Guidelines:**
- Validate early (API layer, not engine)
- Return actionable error messages
- Use clean HTTP status codes (400 for client errors)

**Pattern:**
```python
def _validate_request(self, request: Request) -> str | None:
    """Returns error message or None if valid."""
    if not request.input:
        return "Input cannot be empty"

    if request.param and request.param not in VALID_PARAMS:
        return f"Invalid param '{request.param}'. Valid: {VALID_PARAMS}"

    return None

def handle_request(self, request: Request) -> Response:
    error = self._validate_request(request)
    if error:
        return Response(status_code=400, content={"error": error})
    return self._process(request)
```

---

## Connector Communication

**Guidelines:**
- Use context managers for resource cleanup
- Handle timeouts explicitly
- Implement proper error propagation

**Pattern:**
```python
async def send_to_next_stage(self, data: Tensor) -> None:
    try:
        async with self.connector.acquire(timeout=30.0) as channel:
            await channel.send(data)
    except TimeoutError:
        logger.error("Stage communication timeout")
        raise StageError("Downstream stage unavailable")
    except Exception as e:
        logger.error(f"Stage communication failed: {e}")
        raise
```

---

## Stage Lifecycle

**Guidelines:**
- Stages are configured at init time
- Runtime reconfiguration requires full teardown
- State must be properly managed across stage boundaries

**Pattern:**
```python
class Stage:
    def __init__(self, config: StageConfig):
        self._validate_config(config)
        self.config = config
        self._state = StageState.INITIALIZED

    async def start(self) -> None:
        assert self._state == StageState.INITIALIZED
        await self._allocate_resources()
        self._state = StageState.RUNNING

    async def stop(self) -> None:
        if self._state == StageState.RUNNING:
            await self._release_resources()
        self._state = StageState.STOPPED
```

---

## Test Coverage Requirements

| Code Type | Requirement |
|-----------|-------------|
| New features | Happy path + edge cases + error handling |
| Bug fixes | Regression test + edge cases around fix |
| Performance | Benchmarks with before/after measurements |
| Distributed | Must test in distributed mode |
| Quantization | Memory savings + quality impact measured |
| API endpoints | Input validation + error responses |

---

## Code Quality Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Function length | 100+ lines | Review for split opportunities |
| Complexity | 11+ | Requires refactoring |
| Nesting depth | 4+ levels | Requires refactoring |
| Parameters | 7+ | Consider parameter object |

---

## Error Handling Pattern

**Guidelines:**
- Use specific exception types
- Include actionable context in messages
- Don't silently swallow errors

**Pattern:**
```python
class ValidationError(Exception):
    """Raised when input validation fails."""
    pass

class EngineError(Exception):
    """Raised when engine operation fails."""
    pass

def process(self, request: Request) -> Response:
    try:
        return self._process_internal(request)
    except ValidationError as e:
        logger.warning(f"Validation failed: {e}")
        return Response(status_code=400, content={"error": str(e)})
    except EngineError as e:
        logger.error(f"Engine error: {e}")
        return Response(status_code=500, content={"error": "Internal error"})
```

---

## Logging Pattern

**Guidelines:**
- Use appropriate log levels
- Include relevant context
- Avoid logging sensitive data

```python
# Good
logger.info(f"Processing request {request_id} with model {model_name}")
logger.warning(f"Slow response time: {elapsed_ms}ms for {endpoint}")
logger.error(f"Failed to load model {model_path}: {e}")

# Bad
logger.info("Processing request")  # No context
print(f"Error: {e}")  # Wrong logging method
logger.debug(f"User token: {token}")  # Sensitive data
```
