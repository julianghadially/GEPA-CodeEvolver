# CodeEvolver Analysis

CodeEvolver offers autonomous coding agents for high reliability AI systems. It uses GEPA optimization to evolve your AI system code until it performs optimally for a given dataset and outcome metric.

CodeEvolver provides the coding agents, api, Execution, environments, and git service Required for managing code editing, execution, and evaluation.

CodeEvolver relies on GEPA-CodeEvolver (fork of gepa-ai/gepa) to run the reflective language model optimization process.

CodeEvolver Repository: https://github.com/julianghadially/CodeEvolver

Please see below for CodeEvolver details. 

## Overview
CodeEvolver offers autonomous coding agents for turning static code into self-improving code for AI workflows. 

This combines several mechanisms:
- **Optimizer algorithm:** GEPA is a reflective language model algorithm that makes point mutations to the code base, over many iterations, and the best solution is selected, based on a dataset and a reward metric.
- **Coding agents**: Autonomous agents execute code changes that are requested by the optimizer. 
- **Git branching:** A git process manages evolving code across many git worktrees  
- **Sandboxing for security:** Coding agents are a big cyber risk without sandboxing, network policies, etc. 

### Optimizer
The optimizer is handled by a separate repository, which will later be loaded into this repository. Assume code change requests come in the format shown in specs/change_request_payload.json.

### Coding Agents
CodeEvolver agents uses Claude Agents SDK in a fully autonomous, dangerously-skip-permissions mode, which uses a Modal sandbox execution environment for modifying code, running code, and executing bash / grep / glob. After code changes are made, the app needs to run a mutated version of the code, and return the output. 

Code changes will be made in the context of GEPA optimization - i.e., an evolutionary, 100+ step process. Speed and parallel execution of coding changes is important. The AI worfklow code needs to be edited over 100 times. Each mutation is small, but costs will add up. Do not worry about cost right now.

### Git branching
Users Connect their code with our service by adding our GitHub app, which adds our organization as a contributor to their GitHub.

### Security
Security should be designed for from day one, because autonomous coding agents introduce the trifecta of security risk: 
1. Untrusted inputs, including prompt injection embedded into popular sites
2. Network access
3. Access to user data (RAG databases, code, secrets, and possibly PII). 

See security architecture below.

## V1 outcomes (for Rostam):
- Connect a GitHub repository
- Execute a change request
- Complete v1 of security: (see for v1 below)
- API / sandbox deployed to Modal App

## Technology Stack and Architecture
- **Language**: Python
- **API Framework**: Modal Sandbox App serving FastAPI
- **Database**: MongoDB (flexible, but preferred)
- **Execution Environment**: Modal Sandbox. (must spin up in <10 seconds, support 20+ concurrent environments per user)

**Sandbox Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Modal Web Endpoint (FastAPI)                                    │
│  - Receives HTTP requests                                        │
│  - Creates Modal sandbox for each change request                 │
│  - Manages MongoDB connections                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Creates Sandbox
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Modal Sandbox (per mutation)                                    │
│  ───────────────────────────────────────────────────────────── │
│  /workspace/                                                     │
│    ├── .git/                                                     │
│    ├── requirements.txt  ← pip install -r this                  │
│    ├── src/                                                      │
│    └── program.json                                              │
│                                                                  │
│  Claude Agent SDK runs HERE:                                     │
│  - Native Bash tool → subprocess.run() ✅                        │
│  - Native Grep tool → subprocess.run("grep") ✅                  │
│  - Native Read/Edit → file operations ✅                         │
│  - pip install → works dynamically ✅                            │
│                                                                  │
│  Lifecycle:                                                      │
│  1. Sandbox.create() → container starts                          │
│  2. git clone repo, pip install dependencies                     │
│  3. Run Claude Agent with full capabilities                      │
│  4. sandbox.terminate() → container destroyed                    │
└─────────────────────────────────────────────────────────────────┘
```

## Security Architecture
- **Client-specific isolation (v2):** Execution of code will be isolated in v2. Each client should be in a separate container (e.g., client could have malicious code to steal other clients' data or secrets)
- **Network Egress Control and whitelists:** Limit urls to allowed domains and ips set by our best practices and by the user (e.g., api.firecrawl.dev)
- **Secrets management (v2)**: use env file for v1
- **Monitoring and detection:** omit for v1


```
┌─────────────────────────────────────────────────────────────────┐
│                       CodeEvolver Service (v2)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      API Gateway                          │   │
│  │  - Authentication                                         │   │
│  │  - Rate limiting                                          │   │
│  │  - Request validation                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Secrets Manager                        │   │
│  │  - Per-client encrypted secrets                           │   │
│  │  - Never exposed to agent                                 │   │
│  │  - Injected via proxy                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Client A    │  │  Client B    │  │  Client C    │          │
│  │  Sandbox     │  │  Sandbox     │  │  Sandbox     │          │
│  │  ──────────  │  │  ──────────  │  │  ──────────  │          │
│  │  - Isolated  │  │  - Isolated  │  │  - Isolated  │          │
│  │  - Own net   │  │  - Own net   │  │  - Own net   │          │
│  │  - Egress    │  │  - Egress    │  │  - Egress    │          │
│  │    proxy     │  │    proxy     │  │    proxy     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                    │
│         ▼                 ▼                 ▼                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     Egress Proxy                          │   │
│  │  - Whitelist domains                                      │   │
│  │  - Inject secrets as headers                              │   │
│  │  - Log all outbound traffic                               │   │
│  │  - Block unauthorized destinations                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│                    ┌───────────────────┐                         │
│                    │ Allowed APIs      │                         │
│                    │ - Claude          │                         │
│                    │ - OpenAI          │                         │
│                    │ - Our whitelist   │                         │
│                    │ - Users whitelist │                         │
│                    └───────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Security needs for v1:** 
- Keep workers specific to each client
- Make API requests direct to modal / sandbox app (Omit separate api gateway)
- No egress proxy (temp)
- Use env for secrets (temp)


-------------------------------------------------------------------------


# Ongoing notes on requirements

Below this line is an ongoing, workspace for Claude / AI coding agents. Do not edit text above the line unless it is fully incorrect.

## Components

### API (FastAPI served by Modal app)

#### POST /execute_step
Receives a change request from GEPA optimization as input, and executes that single optimization step: applies a mutation, runs the program, returns output for GEPA reward calculation.

**Request Payload**:
```json
{
  "client_id": "string",
  "program_id": "string (new program id for this mutation)",
  "parent_program_id": "string (parent program id)",
  "mutation_type": "prompt | code",

  // AI Workflow program location (path from project root):
  "program_json_path": "path/to/program.json",
  "entry_point": "module.ClassName",  // DSPy module to instantiate < is this needed?

  // For prompt mutations - GEPA's candidate format:
  "candidate": {
    "component_name": "new instruction text",
    ...
  },

  // For code mutations - natural language:
  "change_request": "string (natural language description)",
  "change_location": "string (module path, optional)",

  // Test data to run after mutation:
  "test_examples": [
    {"input_field": "value", ...},
    ...
  ],
  "capture_traces": false
}
```

**Field Details**:
- `program_json_path`: Path to program.json from project root (we have the code via /connect-git)
- `entry_point`: DSPy module class to instantiate and run (e.g., `"fire.FIREJudge"`)
- `candidate`: For prompt mutations - `dict[str, str]` mapping component names to new instruction text
- `change_request`: For code mutations - natural language description for Claude agent
- `test_examples`: DSPy Examples for running the mutated program
- `capture_traces`: Whether to return execution traces for GEPA reflection

**Response**:
```json
{
  "program_id": "string",
  "status": "success | failed",
  "pipeline_outputs": [
    {"example_id": 0, "output": <any DSPy forward() return value>},
    ...
  ],
  "traces": [...],  // If capture_traces=true
  "branch_name": "string (for code mutations)",
  "program_json": {...},  // Updated program state after mutation
  "error": "string (if failed)"
}
```

- `pipeline_outputs`: Raw outputs from running the mutated program on each test example
- GEPA computes scores client-side using outputs + ground truth labels

#### POST /connect-git
Registers a client repository. Minimal payload - paths provided per-request in `/execute_step`.

**Request**:
```json
{
  "repo_url": "https://github.com/user/project",
  "installation_id": 12345  // Optional: GitHub App installation ID for private repos
}
```

**Response**:
```json
{
  "client_id": "client_abc123",
  "status": "connected"
}
```

- Supports both public and private repositories
- Private repos require `installation_id` from GitHub App installation
- Uses `GitHubAppService` for token-based authentication
- Clones repo to server storage
- Returns `client_id` for future requests

#### GET /program/{program_id}
Retrieves program details and `program_json`.

### Program Database (MongoDB)
Stores all program versions and their optimized prompts.

| Field | Description |
|-------|-------------|
| `client_id` | Internal client identifier |
| `program_id` | Unique program version identifier |
| `parent_program_id` | Parent program(s) - 1 for mutation, 2 for crossover |
| `program_json` | DSPy optimized program JSON (see `specs/example_program.json`) |
| `branch_name` | Git branch for this program version |
| `created_at` | Timestamp |
| `status` | pending / in_progress / completed / failed |

**Purpose**: Centralizes prompt changes for direct editing by external optimizer (GEPA).

### DSPy Program JSON Structure
The `program_json` is the serialized output of DSPy's `program.save()`. Structure:

```json
{
  "module_path.predict": {
    "traces": [],
    "train": [],
    "demos": [],
    "signature": {
      "instructions": "The system prompt / task description",
      "fields": [
        {"prefix": "Input:", "description": "field description"},
        {"prefix": "Output:", "description": "field description"}
      ]
    },
    "lm": null
  },
  "another_module.predict": { ... },
  "metadata": {
    "dependency_versions": {"python": "3.11", "dspy": "3.0.4", ...}
  }
}
```

Key mutation targets:
- `signature.instructions`: The main prompt text (primary target for prompt mutations)
- `signature.fields[].description`: Field descriptions
- `demos`: Few-shot examples (can be added/modified)
- Module structure itself (code mutations only)

### Auto-coder Agent
Executes the actual code/prompt changes using Claude Agents SDK.

**SDK Details** (from research):
- **Requires Claude Code CLI** - The Python SDK (`claude-agent-sdk`) is a wrapper that spawns the Claude Code CLI (`@anthropic-ai/claude-code`) as a subprocess. Both must be installed.
- **Requires files on disk** - cannot work with in-memory files
- **API Access**: Use `query()` or `ClaudeSDKClient` from `claude_agent_sdk`
- **Concurrent execution**: Each agent needs its own `cwd` (working directory)
- **Permission mode**: `acceptEdits` for autonomous code modifications (no human prompts)
- **Tools available**: Read, Write, Edit, Bash, Glob, Grep (Write for new files, Edit for existing)

**Implementation Pattern**:
```python
from claude_agent_sdk import query, ClaudeAgentOptions

async def run_mutation(workspace_path: str, change_request: str):
    async for message in query(
        prompt=change_request,
        options=ClaudeAgentOptions(
            cwd=workspace_path,  # Isolated checkout for this branch
            allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
            permission_mode="acceptEdits"
        )
    ):
        yield message
```

**Capabilities**:
- **Prompt mutations**: Modify `program_json` (still needs code checkout to run program)
- **Code mutations**: Edit actual Python DSPy module code in git repo
- **Constraints**: Future support for constrained edits

### Execution Environment
Each agent runs in an isolated environment with its own filesystem.

**Git Worktree Strategy**:
Instead of multiple clones, use git worktree for efficient parallel branch access:
```bash
# Single clone per client
git clone <repo> /workspace/{client_id}/main

# Create worktree per program/mutation
git worktree add /workspace/{client_id}/prog_042 branch_prog_042
git worktree add /workspace/{client_id}/prog_043 branch_prog_043
```

Benefits:
- Single clone, multiple working directories
- Shared `.git` metadata = faster than multiple clones
- Each worktree has different branch checked out
- Changes (fetch) visible across all worktrees

**Requirements**:
- Spin-up time: <10 seconds (worktree add is near-instant)
- Concurrency: 20+ simultaneous worktrees per client
- Isolation: Each worktree = one branch, separate directory
- Contains: Worktree checkout, Claude Agents SDK runtime, Python/DSPy

**Execution Platform: Modal Sandbox**

Decision: Use [Modal Sandbox](https://modal.com/docs/guide/sandboxes) for execution environments.

Architecture inspired by [Modal Vibe](https://github.com/modal-labs/modal-vibe).

**Key Insight:** The Claude Agent SDK runs **inside** the Modal Sandbox, so its native tools (Bash, Grep, Glob, Read, Edit) work directly via subprocess. No custom tool wrappers needed.

Rationale:
- Modal Sandbox provides persistent filesystem while alive
- Dynamic pip install (user's requirements.txt with unknown packages)
- Full bash/python access for Claude Agent
- Apache-2.0 friendly (no license conflicts)
- Client isolation via separate sandboxes

**Architecture:**
See sandbox architecture above

**Implementation Pattern (src/core/sandbox.py):**

```python
import modal
from ..services.github_app import GitHubAppService

class SandboxApp:
    """Manages a Modal sandbox for executing code mutations."""

    @staticmethod
    async def create(
        app: modal.App,
        client_id: str,
        program_id: str,
        repo_url: str,
        installation_id: int | None = None,  # For private repos
        secrets: dict[str, str] | None = None,
    ) -> "SandboxApp":
        # Handle private repo authentication via GitHubAppService
        authenticated_url = repo_url
        if installation_id:
            token = GitHubAppService.get_installation_token(installation_id)
            authenticated_url = GitHubAppService.get_authenticated_repo_url(repo_url, token)

        sandbox = modal.Sandbox.create(app=app, image=get_sandbox_image(), timeout=600)
        sandbox_app = SandboxApp(sandbox, metadata)

        await sandbox_app._clone_repo(authenticated_url)
        await sandbox_app._install_deps()
        if secrets:
            await sandbox_app._inject_secrets(secrets)

        return sandbox_app

    async def apply_code_mutation(self, change_request: str) -> MutationResult:
        # Generate agent script that runs Claude SDK inside sandbox
        script = generate_agent_script(self._workspace, change_request)
        self.sandbox.exec("bash", "-c", f"cat > /tmp/agent.py << 'EOF'\n{script}\nEOF").wait()
        p = self.sandbox.exec("python", "/tmp/agent.py")
        p.wait()
        return parse_agent_output(p.stdout.read(), p.stderr.read(), p.returncode)

# Main entry point
async def execute_mutation(app, client_id, program_id, repo_url, ..., installation_id=None):
    sandbox_app = await SandboxApp.create(app, client_id, program_id, repo_url, installation_id)
    try:
        mutation_result = await sandbox_app.apply_code_mutation(change_request)
        run_result = await sandbox_app.run_program(...)
        return ExecutionResult(...)
    finally:
        sandbox_app.terminate()
```

**Why Sandbox over Modal Function:**

| Feature | Modal Function | Modal Sandbox |
|---------|---------------|---------------|
| Dependencies | Baked at deploy time | Dynamic pip install ✅ |
| Filesystem | Ephemeral per call | Persistent while alive ✅ |
| Bash access | Limited | Full interactive shell ✅ |
| User code | Can't adapt to requirements.txt | Installs anything ✅ |

Future options if needed:
- Self-hosted Docker (maximum privacy, same pattern)
- User's own infrastructure (enterprise)

### Run Program
Executes the mutated DSPy program and returns output.

- Loads program from `program_json`
- Runs as fixed DSPy Adapter
- Returns `pipeline_output` to GEPA for reward calculation

### Git Branching
Handles version control for code mutations.

- Creates branch `program_{program_id}` from parent branch
- For 2-parent crossover: merge strategy TBD
- Can be handled by auto-coder agent or separate bot

## External Integration

### GEPA Integration (gepa-ai/gepa)
See specs/* in GEPA-Codeevolver package

