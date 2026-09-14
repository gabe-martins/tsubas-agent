# Technical Requirements

This document describes the technical requirements, dependencies, and operational constraints for running the Tsubas Agent project.

## 1. Runtime Requirements

| Requirement | Details |
|---|---|
| Language | Python 3.9 or higher |
| Package manager | pip |
| External service | [Ollama](https://ollama.com/) server (local or remote) for LLM inference |
| Network access | Required for `web_search` (multi-backend web search) and `web_scraping` (arbitrary HTTP requests), when using [main_script.py](main_script.py) |
| Local execution | `run_python_code` ([main_script.py](main_script.py) only) spawns a local Python subprocess to run agent-generated code; no network access required for this tool itself |

## 2. Python Dependencies

Declared in [requeriments.txt](requeriments.txt):

| Package | Purpose |
|---|---|
| `crewai` | Multi-agent orchestration framework (agents, tasks, crews, processes). |
| `langchain-ollama` | LangChain integration used by CrewAI's `LLM` wrapper to talk to Ollama. |
| `python-dotenv` | Loads environment variables from a `.env` file. |
| `ddgs` | Multi-backend web search client used by the `web_search` tool. |
| `beautifulsoup4` | HTML parsing used by the `web_scraping` tool. |
| `requests` | HTTP client used by the `web_scraping` tool. |
| `json_repair` | Repairs malformed JSON produced by the architect/worker LLMs before parsing. |
| `pyyaml` | Parsing/serializing generated agent and task configurations (`config/*.yaml`). |
| `pip-system-certs` | Uses the OS certificate store for outbound HTTPS requests. |

Install with:

```bash
pip install -r requeriments.txt
```

## 3. External Services

### 3.1 Ollama

- Must be installed and running before executing either entry point.
- Must have the models referenced by `MODEL_REVIEWER` and `MODEL_WORKER` pulled locally:
  ```bash
  ollama pull llama3.1:8b
  ollama pull qwen2.5-coder:7b
  ```
- Reachable at the URL configured by `OLLAMA_BASE_URL` (default `http://localhost:11434`).
- `main_script.py` calls `MODEL_REVIEWER` for the architect (design) and inspector (quality loop) roles, and `MODEL_WORKER` for the execution agents; alternating between two different models on a server that can't keep both loaded simultaneously will cause repeated reload latency (see [README performance tips](README.md#performance-tips)).

### 3.2 Web Search (optional, `main_script.py` only)

- Used by the `web_search` tool via the `ddgs` package, trying multiple backends (`SEARCH_BACKENDS`, default `google,bing,duckduckgo,brave,yahoo,mojeek`) in sequence until one succeeds.
- Requires outbound internet access. No API key required.
- A backend that succeeds is promoted to the front of the list, and identical queries within the same run are served from an in-memory cache.

### 3.3 Arbitrary Web Targets (optional, `main_script.py` only)

- The `web_scraping` tool issues HTTP GET requests to any URL an agent decides to visit.
- Requests use a fixed `User-Agent` header and a configurable timeout (`SCRAPE_TIMEOUT`, default 10s).
- Response body is truncated to 4000 characters before being returned to the LLM.
- Successful scrapes of a given URL are cached in-memory for the rest of the run.

### 3.4 Local Python Code Execution (optional, `main_script.py` only)

- The `run_python_code` tool writes agent-generated code to a temporary file and runs it with the current Python interpreter (`subprocess.run`), capturing stdout/stderr/exit code, bounded by `CODE_EXEC_TIMEOUT` (default 20s).
- **This executes arbitrary, LLM-generated code with the same OS-user privileges as the running process.** It is gated by `ENABLE_CODE_EXECUTION` (default `true`); set it to `false` to disable the tool in shared, multi-tenant, or otherwise untrusted environments. There is no sandboxing (no container, no restricted filesystem/network access) beyond a per-run temp directory and the execution timeout.
- Automatically assigned to the responsible agent whenever the objective is classified as the `programming` skill (see [README skills section](README.md#skills-automatic-domain-detection)).

## 4. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Base URL of the Ollama server used for both LLM roles. |
| `MODEL_REVIEWER` | No | `llama3.1:8b` | Model used by the architect/inspector ("router") agents. |
| `MODEL_WORKER` | No | `qwen2.5-coder:7b` | Model used by the dynamically generated ("worker") execution agents. (`main.py` also accepts the legacy `MODEL_CODER` name.) |
| `SEARCH_TIMEOUT` | No | `8` | Per-backend timeout (seconds) for `web_search` (`main_script.py` only). |
| `SCRAPE_TIMEOUT` | No | `10` | Timeout (seconds) for `web_scraping` requests (`main_script.py` only). |
| `AGENT_MAX_ITER` | No | `8` | Max reasoning iterations per agent per task (`main_script.py` only). |
| `DESIGN_MAX_ATTEMPTS` | No | `3` | Max self-correction attempts in the team-design loop (`main_script.py` only). |
| `QUALITY_MAX_LOOPS` | No | `2` | Max inspector rework cycles after execution; `0` disables it (`main_script.py` only). |
| `SEARCH_BACKENDS` | No | `google,bing,duckduckgo,brave,yahoo,mojeek` | Comma-separated search engines tried in order (`main_script.py` only). |
| `CODE_EXEC_TIMEOUT` | No | `20` | Timeout (seconds) for a single `run_python_code` execution (`main_script.py` only). |
| `ENABLE_CODE_EXECUTION` | No | `true` | Set to `false` to disable `run_python_code` (`main_script.py` only). |

Environment variables can be provided via a `.env` file in the project root (loaded automatically by `python-dotenv`). Every `main_script.py`-specific variable above can also be overridden per-run via CLI flags (see [README usage](README.md#usage)).

## 5. File System Requirements

- The process must have write access to the `config/` directory (created automatically if missing via `os.makedirs("config", exist_ok=True)`).
- The process must have write access to the working directory for output artifacts generated by executed tasks (e.g., JSON files produced by agent output, such as `cotacoes_hoje.json` in the sample objective).
- Existing files at [config/agents.yaml](config/agents.yaml) and [config/tasks.yaml](config/tasks.yaml) are overwritten on every run of the team-generation phase.

## 6. Functional Requirements

### 6.1 Common to both entry points

1. The system must accept a single natural-language objective string as input.
2. The system must call an LLM to generate agent and task definitions, persist them to [config/agents.yaml](config/agents.yaml) / [config/tasks.yaml](config/tasks.yaml), reload them, and dynamically instantiate `Agent`/`Task` objects for each entry.
3. Agents whose definition includes a `tools` list must have the corresponding tool functions attached, resolved via a static tool name-to-function map.
4. Tasks must be linked to their declared `agent` by ID; a missing agent reference will result in a task with no bound agent.
5. The system must execute the instantiated crew using a sequential process (`Process.sequential`) and print the final result.

### 6.2 `main.py` (minimal)

- Generates two YAML documents (agents and tasks) separated by `---` in a single LLM call, with no self-correction, no skills, and no quality loop.
- Strips markdown code fences from the LLM response before parsing; raises if fewer than two YAML blocks are produced.

### 6.3 `main_script.py` (full-featured)

- Classifies the objective into a skill (`programming`, `research`, or `general`) via a deterministic keyword match (`detect_skill`) before any LLM call; the detected skill can be overridden via `--skill`.
- Generates a single JSON object (agents + tasks) instead of YAML text, retrying up to `DESIGN_MAX_ATTEMPTS` times with error feedback on malformed output, and repairing near-valid JSON via `json_repair`.
- Deterministically guarantees the detected skill's primary tool (e.g. `run_python_code` for `programming`) is present on some agent (`ensure_skill_tools`), independent of whether the architect LLM remembered to add it.
- Runs a post-execution Quality & Verification loop (up to `QUALITY_MAX_LOOPS` cycles) combining deterministic guardrails (fabrication-language detection, invalid JSON output, invalid Python syntax) with an LLM-based inspector agent, triggering a rework task when rejected.
- Accepts the objective via CLI positional arguments, `--file`, or piped stdin, and exposes tuning flags for timeouts/iteration counts/loop counts/model overrides (see [README usage](README.md#usage)).

## 7. Non-Functional / Operational Constraints

- **Determinism**: Skill detection is deterministic (keyword-based); generated agent/task JSON content still depends on the LLM's output and is not guaranteed to be identical between runs.
- **Error handling**: `main_script.py` raises `RuntimeError` with the raw LLM response when the design loop exhausts `DESIGN_MAX_ATTEMPTS`; `main.py` has a simpler, less defensive check.
- **Content limits**: Scraped web page text is truncated to 4000 characters; `run_python_code` output is truncated to the last ~3000 characters of stdout/stderr.
- **Timeouts**: Web search, web scraping, and code execution all use configurable timeouts; no retry logic beyond the multi-backend search fallback.
- **Concurrency**: The crew process is sequential; there is no parallel task execution or async support.
- **Security**:
  - `run_python_code` executes model-generated code locally with the running process's OS-user privileges (subprocess, temp-dir isolated, timeout-bound, but not sandboxed against filesystem/network access). Disable with `ENABLE_CODE_EXECUTION=false` in untrusted contexts.
  - `web_scraping` performs unrestricted outbound HTTP requests to URLs chosen by the LLM/agent at runtime; no allow-list or SSRF protection is implemented. Deploy with network egress controls if used in untrusted contexts.
- **Secrets management**: No API keys are required for the default web-search/Ollama setup; if `.env` is used for other credentials, it must be excluded from version control (already covered by [.gitignore](.gitignore)).

## 8. Known Limitations

- No automated test suite is present in the repository.
- No license file is currently included.
- `run_python_code` has no sandboxing beyond a temp directory and a timeout; it relies entirely on `ENABLE_CODE_EXECUTION` and operator judgement about trust level.
