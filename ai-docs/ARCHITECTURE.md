# Architecture — 아키텍처

이 문서는 AgentOutO의 패키지 구조, 모듈 책임, 클래스 관계, 데이터 흐름을 설명한다.

---

## 1. 패키지 구조

```
agnetouto/
├── __init__.py          # 공개 API 엑스포트
├── _constants.py        # 내부 상수 (CALL_AGENT, FINISH)
├── agent.py             # Agent 데이터클래스
├── context.py           # 에이전트별 대화 컨텍스트 관리
├── exceptions.py        # 커스텀 예외 계층
├── message.py           # Message 데이터클래스
├── provider.py          # Provider 데이터클래스
├── router.py            # 메시지 라우팅, 시스템 프롬프트, 도구 스키마
├── runtime.py           # 에이전트 루프 엔진, 병렬 실행, run()/async_run()
├── tool.py              # Tool 데코레이터/클래스
└── providers/
    ├── __init__.py      # ProviderBackend ABC, LLMResponse, get_backend()
    ├── openai.py        # OpenAI 구현
    ├── anthropic.py     # Anthropic 구현
    └── google.py        # Google Gemini 구현
```

---

## 2. 공개 API

`agnetouto/__init__.py`에서 엑스포트되는 공개 API:

| 이름 | 타입 | 설명 |
|------|------|------|
| `Agent` | dataclass | 에이전트 설정 |
| `Tool` | class | 도구 데코레이터/클래스 |
| `Provider` | dataclass | 프로바이더 (API 접속 정보) |
| `Message` | dataclass | 전달/반환 메시지 |
| `RunResult` | dataclass | 실행 결과 컨테이너 |
| `run()` | function | 동기 실행 진입점 |
| `async_run()` | async function | 비동기 실행 진입점 |

---

## 3. 모듈별 책임

### `agent.py` — Agent

```python
@dataclass
class Agent:
    name: str               # 에이전트 이름 (필수)
    instructions: str       # 역할 설명 (필수)
    model: str              # 모델 이름 (필수)
    provider: str           # 프로바이더 이름 (필수)
    max_output_tokens: int  # 최대 출력 토큰 (기본: 4096)
    reasoning: bool         # 추론 모드 토글 (기본: False)
    reasoning_effort: str   # 추론 강도 (기본: "medium")
    reasoning_budget: int | None  # 추론 토큰 예산 (기본: None)
    temperature: float      # 온도 (기본: 1.0)
    extra: dict[str, Any]   # 추가 파라미터 (기본: {})
```

순수 데이터 컨테이너. 로직 없음. `provider` 필드는 `Provider.name`과 매칭되는 문자열.

### `provider.py` — Provider

```python
@dataclass
class Provider:
    name: str                                    # 식별 이름
    kind: Literal["openai", "anthropic", "google"]  # API 종류
    api_key: str                                 # API 키
    base_url: str | None                         # 커스텀 엔드포인트 (선택)
```

API 접속 정보만 담당. 모델 설정은 Agent에서 한다.

### `message.py` — Message

```python
@dataclass
class Message:
    type: Literal["forward", "return"]
    sender: str
    receiver: str
    content: str
    call_id: str  # uuid4 자동 생성
```

현재 런타임 내부에서 `Message`를 직접 생성/전달하지는 않는다 (런타임은 content 문자열을 직접 전달). 추후 로깅/트레이싱에서 사용될 예정.

### `tool.py` — Tool

`Tool`은 데코레이터로 사용된다:

```python
@Tool
def my_func(arg: str) -> str:
    """설명."""
    return result
```

내부 동작:
1. `func.__name__` → `self.name`
2. `func.__doc__` → `self.description`
3. `inspect.signature` + `get_type_hints` → JSON Schema 자동 생성
4. `execute(**kwargs)` → 함수 실행 (async 지원)
5. `to_schema()` → LLM에 제공할 도구 스키마 반환

타입 매핑: `_PYTHON_TYPE_TO_JSON` 딕셔너리로 Python 타입 → JSON Schema 타입 변환.

### `context.py` — Context

에이전트별 대화 컨텍스트를 관리하는 프로바이더 비의존적 중간 표현.

```python
class Context:
    system_prompt: str              # 시스템 프롬프트 (읽기 전용)
    messages: list[ContextMessage]  # 대화 이력

    def add_user(content)                          # 유저 메시지 추가
    def add_assistant_text(content)                 # 어시스턴트 텍스트 추가
    def add_assistant_tool_calls(tool_calls, content)  # 어시스턴트 도구 호출 추가
    def add_tool_result(tool_call_id, tool_name, content)  # 도구 결과 추가
```

`ToolCall` dataclass: `id`, `name`, `arguments` 보유.
`ContextMessage` dataclass: `role`, `content`, `tool_calls`, `tool_call_id`, `tool_name` 보유.

각 프로바이더 백엔드가 이 Context를 자신의 API 포맷으로 변환한다.

### `router.py` — Router

중앙 라우팅 허브. 에이전트, 도구, 프로바이더를 이름으로 관리하고, LLM 호출을 중개한다.

```python
class Router:
    def __init__(agents, tools, providers)
    def get_agent(name) -> Agent
    def get_tool(name) -> Tool
    def build_tool_schemas(current_agent) -> list[dict]  # call_agent + finish 포함
    def build_system_prompt(agent) -> str                 # 에이전트 목록 포함
    def call_llm(agent, context, tool_schemas) -> LLMResponse
```

**시스템 프롬프트 자동 생성:**
- 에이전트 이름과 instructions 포함
- 다른 에이전트 목록 포함 (현재 에이전트 제외)
- `call_agent`/`finish` 사용 안내 포함

**도구 스키마 자동 생성:**
- 사용자 정의 도구 스키마 전체
- `call_agent` 도구 (agent_name, message 파라미터)
- `finish` 도구 (message 파라미터)

**프로바이더 백엔드 캐싱:**
- `_backends` 딕셔너리로 kind별 백엔드 인스턴스 캐싱
- `get_backend(kind)` 팩토리로 lazy 생성

### `runtime.py` — Runtime

에이전트 루프 엔진. 핵심 모듈.

```python
class Runtime:
    async def execute(agent, forward_message) -> str
    async def _run_agent_loop(agent, forward_message) -> str
    async def _execute_tool_call(tc) -> str
```

**에이전트 루프 (`_run_agent_loop`):**
1. 시스템 프롬프트 생성 → Context 초기화
2. forward_message를 user 메시지로 추가
3. 무한 루프:
   a. LLM 호출
   b. tool_calls가 없으면 → 텍스트 응답 반환
   c. `finish` 호출이 있으면 → message 반환
   d. tool_calls를 `asyncio.gather`로 병렬 실행
   e. 결과(또는 에러)를 tool result로 추가
   f. 다음 반복

**도구 실행 (`_execute_tool_call`):**
- `call_agent` → 대상 에이전트의 `_run_agent_loop` 재귀 호출
- 일반 도구 → `tool.execute(**kwargs)` 실행, 에러 시 `ToolError` 래핑

**`run()` / `async_run()`:**
- Router 생성 → Runtime 생성 → execute 호출 → RunResult 반환
- `run()`은 `asyncio.run(async_run(...))` 래퍼

### `_constants.py`

```python
CALL_AGENT = "call_agent"
FINISH = "finish"
```

매직 스트링 방지용. `router.py`와 `runtime.py`에서 공유.

### `exceptions.py`

```
AgentOutOError (base)
├── ProviderError(provider_name, message)
├── AgentError(agent_name, message)
├── ToolError(tool_name, message)
└── RoutingError(message)
```

### `providers/__init__.py`

```python
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]

class ProviderBackend(ABC):
    async def call(context, tools, agent, provider) -> LLMResponse

def get_backend(kind: str) -> ProviderBackend  # 팩토리 함수
```

`get_backend`는 lazy import로 각 백엔드 모듈을 로드한다.

---

## 4. 데이터 흐름

### 실행 시작 → 결과 반환

```
run(entry, message, agents, tools, providers)
  │
  ├── Router(agents, tools, providers)   ← 이름 기반 레지스트리 생성
  ├── Runtime(router)
  └── runtime.execute(entry, message)
        │
        └── _run_agent_loop(entry, message)
              │
              ├── router.build_system_prompt(agent)  ← 시스템 프롬프트 생성
              ├── Context(system_prompt)              ← 컨텍스트 초기화
              ├── context.add_user(message)            ← 전달 메시지 추가
              ├── router.build_tool_schemas(agent.name) ← 도구 스키마 생성
              │
              └── while True:
                    ├── router.call_llm(agent, context, schemas)
                    │     ├── provider = providers[agent.provider]
                    │     ├── backend = get_backend(provider.kind)
                    │     └── backend.call(context, schemas, agent, provider)
                    │           └── LLMResponse(content, tool_calls)
                    │
                    ├── no tool_calls? → return content
                    ├── finish found? → return finish.message
                    │
                    ├── context.add_assistant_tool_calls(tool_calls)
                    ├── asyncio.gather(*[_execute_tool_call(tc) for tc in tool_calls])
                    │     ├── call_agent → _run_agent_loop(target, msg)  [재귀]
                    │     └── tool → tool.execute(**kwargs)
                    │
                    └── context.add_tool_result(id, name, result_or_error)
```

### 프로바이더 백엔드 데이터 변환

```
Context (프로바이더 비의존)
    │
    ├── OpenAI:  _build_messages(context) → list[dict]  (role/content/tool_calls)
    ├── Anthropic: _build_messages(context) → list[dict]  (content blocks)
    └── Google:  _build_contents(context) → list[Content]  (protos)
```

각 백엔드는 Context의 통일된 표현을 자신의 API 포맷으로 변환한다.

---

## 5. 의존성

### 런타임 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `openai` | ≥1.50.0 | OpenAI 및 호환 API 클라이언트 |
| `anthropic` | ≥0.34.0 | Anthropic API 클라이언트 |
| `google-generativeai` | ≥0.8.0 | Google Gemini API 클라이언트 |

### 개발 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `pytest` | ≥8.0 | 테스트 |
| `pytest-asyncio` | ≥0.23 | 비동기 테스트 |
| `mypy` | ≥1.8 | 타입 체크 |

### 표준 라이브러리 사용

| 모듈 | 사용 위치 |
|------|----------|
| `asyncio` | `runtime.py` — gather, run |
| `json` | `providers/openai.py` — tool call 파싱 |
| `uuid` | `message.py`, `providers/google.py` — ID 생성 |
| `inspect` | `tool.py` — 함수 시그니처 분석 |
| `typing` | 전역 — 타입 힌팅 |
| `dataclasses` | 데이터 클래스 정의 |
| `abc` | `providers/__init__.py` — 추상 클래스 |

---

## 6. 핵심 설계 패턴

### 패턴 1: 이름 기반 레지스트리

에이전트, 도구, 프로바이더는 모두 이름(문자열)으로 참조된다.
- `Agent.provider = "openai"` → `Router._providers["openai"]`
- LLM이 `call_agent(agent_name="writer")` → `Router._agents["writer"]`
- LLM이 `search_web(...)` → `Router._tools["search_web"]`

### 패턴 2: 재귀적 에이전트 루프

`call_agent`는 `_run_agent_loop`을 재귀적으로 호출한다. 각 호출은 독립적인 Context를 가진다. 호출 스택이 곧 에이전트 호출 체인이다.

### 패턴 3: 프로바이더 추상화

`ProviderBackend` ABC → 각 kind별 구현체. `get_backend(kind)` 팩토리가 lazy import로 인스턴스를 생성한다. Router가 kind별로 캐싱한다.

### 패턴 4: 에러 → 도구 결과

`asyncio.gather(return_exceptions=True)`로 에러를 도구 결과에 포함시켜 LLM에게 전달한다. 런타임이 크래시하지 않고 LLM이 에러를 보고 판단한다.
