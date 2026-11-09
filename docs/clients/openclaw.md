# OpenClaw

[OpenClaw](https://openclaw.ai) has full MCP support and a plugin system that enables automatic memory recall and storage via lifecycle hooks.

## Integration Options

| Method | What it does | Setup effort |
|---|---|---|
| **MCP only** | LLM-driven memory via tool calls | Add MCP config |
| **MCP + Plugin** (recommended) | Automatic recall/remember via hooks + 15 LLM tools for explicit ops | Install plugin + MCP config |

The plugin approach is strongly recommended -- it handles recall/remember automatically without consuming LLM tool calls, and injects memories directly into the system prompt for better context.

## Plugin Setup (Recommended)

### 1. Install

```bash
openclaw plugins install @fpytloun/openclaw-mnemory
```

### 2. Configure

Add to your `openclaw.json`:

```jsonc
{
  "plugins": {
    // Activate mnemory as the memory slot (only one memory plugin at a time)
    "slots": { "memory": "mnemory" },
    "entries": {
      "memory-core": { "enabled": false },
      "mnemory": {
        "enabled": true,
        "config": {
          "url": "http://localhost:8050",
          "apiKey": "${MNEMORY_API_KEY}"
        }
      }
    }
  }
}
```

Config values support `${ENV_VAR}` syntax for environment variable resolution.

Restart the OpenClaw Gateway after changing this configuration. Explicitly
disabling `memory-core` is important: on OpenClaw 2026.10.9 and 2026.11.2, selecting
the memory slot alone can leave native `memory_search` active and shadow mnemory's
tool. This disables the native plugin's file-memory tools and background
consolidation, but does not delete existing memory files. Do not deny
`memory_search` through tool policy, because that would block mnemory's tool too.

### 3. Configuration Options

The plugin can be configured via OpenClaw's settings UI (the manifest provides a config schema with UI hints) or via environment variables. Plugin config takes priority.

| Plugin Config | Env Var | Default | Description |
|---|---|---|---|
| `url` | `MNEMORY_URL` | (required) | Mnemory server URL |
| `apiKey` | `MNEMORY_API_KEY` | (empty) | Bearer token for authentication |
| `userId` | `MNEMORY_USER_ID` | (empty) | User ID (required when API key is wildcard) |
| `agentPrefix` | `MNEMORY_AGENT_PREFIX` | `openclaw` | Prefix for agent IDs (e.g., `openclaw:main`) |
| `autoRecall` | -- | `true` | Auto-inject memories into context |
| `autoCapture` | -- | `true` | Auto-extract memories from conversations |
| `recallFindFirst` | -- | `true` | Use AI-powered search on first turn (higher quality, slower) |
| `recallSearchMode` | -- | `search` | Search mode for subsequent turns: `find` or `search` |
| `scoreThreshold` | -- | `0.5` | Min relevance score for recalled memories (0.0-1.0) |
| `includeAssistant` | -- | `true` | Send assistant messages for extraction |
| `managed` | -- | `true` | Include behavioral instructions in system prompt |
| `timeout` | `MNEMORY_TIMEOUT` | `60000` | HTTP request timeout in ms |

### 4. Verify

Start OpenClaw and send a message. You should see `mnemory: initialized` in the
gateway logs and no `plugin tool name conflict (mnemory): memory_search`
diagnostic. Ask the agent to call `memory_search` and verify that its results come
from mnemory rather than local `MEMORY.md` files. Memories will appear in the
mnemory management UI at `http://localhost:8050/ui`.

## MCP Setup

For MCP-only access (without the plugin), add to your `openclaw.json`:

```jsonc
{
  "mcp": {
    "mnemory": {
      "type": "streamable-http",
      "url": "http://localhost:8050/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key",
        "X-Agent-Id": "openclaw"
      }
    }
  }
}
```

This gives the LLM access to all 16 MCP tools but does not provide automatic recall/remember.

## How the Plugin Works

| Hook | Action |
|---|---|
| `session_start` | Pre-fetches instructions and core memories from `/api/recall` (non-blocking) |
| `before_prompt_build` | Two-phase recall: awaits init, then sends the user's prompt as a search query for topical memories. First turn uses `find` mode (AI-powered), subsequent turns use `search` mode (fast). Results are replaced each turn. |
| `agent_end` | Extracts the last user+assistant exchange, strips inbound metadata, sends to `/api/remember` for background extraction |
| `after_compaction` | Re-fetches memories and resets turn tracking after context compaction |
| `session_end` | Cleans up session state |

### Tools Registered

The plugin registers 15 tools for explicit memory operations:

`memory_search`, `memory_find`, `memory_ask`, `memory_add`, `memory_add_batch`, `memory_update`, `memory_delete`, `memory_delete_batch`, `memory_list`, `memory_categories`, `memory_recent`, `memory_save_artifact`, `memory_get_artifact`, `memory_list_artifacts`, `memory_delete_artifact`

### CLI Commands

The plugin adds CLI commands under `openclaw mnemory`:

- `openclaw mnemory status` -- Show connection status and memory counts
- `openclaw mnemory search <query>` -- Search memories from the command line
- `openclaw mnemory list` -- List all memories

## Tips

- The plugin and MCP tools work together -- plugin handles automatic recall/remember, MCP tools handle explicit search/update/delete
- Memory persists across session compaction (the plugin re-fetches after compaction)
- Select `plugins.slots.memory: "mnemory"` and disable `plugins.entries.memory-core.enabled` to replace native memory tools
- See the [coding assistant system prompt guide](../system-prompts/claude-code.md) for tips on what gets remembered
