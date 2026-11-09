# OpenClaw Plugin -- Mnemory

An [OpenClaw](https://openclaw.ai) plugin that provides long-term semantic memory backed by a [mnemory](https://github.com/fpytloun/mnemory) server.

## Features

- **Auto-Recall**: Automatically fetches relevant memories and injects them into the system prompt before each agent turn
- **Auto-Capture**: Extracts and stores memories from conversations after each exchange (fire-and-forget)
- **15 Explicit Tools**: `memory_search`, `memory_find`, `memory_ask`, `memory_add`, `memory_add_batch`, `memory_update`, `memory_delete`, `memory_delete_batch`, `memory_list`, `memory_categories`, `memory_recent`, `memory_save_artifact`, `memory_get_artifact`, `memory_list_artifacts`, `memory_delete_artifact`
- **CLI Commands**: `openclaw mnemory status`, `openclaw mnemory search`, `openclaw mnemory list`
- **Compaction-safe**: Memories survive session compaction via hook-based re-injection

## Prerequisites

A running [mnemory](https://github.com/fpytloun/mnemory) server accessible via HTTP.

## Install

```bash
openclaw plugins install @fpytloun/openclaw-mnemory
```

## Configure

In your `openclaw.json`:

```json5
{
  plugins: {
    slots: { memory: "mnemory" },
    entries: {
      "memory-core": { enabled: false },
      mnemory: {
        enabled: true,
        config: {
          url: "http://localhost:8050",
          apiKey: "${MNEMORY_API_KEY}"
        }
      }
    }
  }
}
```

### Configuration Options

| Key | Env Var | Default | Description |
|-----|---------|---------|-------------|
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

Config values support `${ENV_VAR}` syntax for environment variable resolution.

## How It Works

1. **`session_start`**: Pre-fetches instructions and core memories from `/api/recall` (non-blocking)
2. **`before_prompt_build`**: Two-phase recall per turn:
   - Awaits the init recall (instructions + core memories) if still pending
   - Sends the current user prompt as a search query to `/api/recall` for topical memories
   - First turn uses AI-powered `find` mode (configurable via `recallFindFirst`), subsequent turns use fast `search` mode (configurable via `recallSearchMode`)
   - Search results are **replaced** each turn (not accumulated) — the server deduplicates via session tracking
   - Injects instructions via `prependSystemContext` (cacheable), core memories + search results via `appendSystemContext`
3. **`agent_end`**: Extracts the last user+assistant exchange, strips inbound metadata, sends to `/api/remember` for background extraction
4. **`after_compaction`**: Re-fetches memories and resets turn tracking after context compaction
5. **`session_end`**: Cleans up session state

## Memory Slot

This plugin declares `kind: "memory"` and is activated by setting
`plugins.slots.memory: "mnemory"`. Also explicitly disable `memory-core` as shown
above, then restart the OpenClaw Gateway.

On OpenClaw 2026.10.9 and 2026.11.2, selecting the memory slot alone can leave the
native `memory_search` tool active. It shadows mnemory's tool and produces a
`plugin tool name conflict (mnemory): memory_search` diagnostic. Disabling
`memory-core` makes mnemory own `memory_search` without renaming it. This also
disables the native plugin's file-memory tools and background consolidation;
existing memory files are not deleted.

Do not add `memory_search` to `tools.deny`: that would also block mnemory's tool.

## Development

With OpenClaw installed, run `npm ci`, `npm run test:package`, and
`npm run test:openclaw`. The integration test installs the packed plugin into
temporary OpenClaw state, starts a real Gateway, and invokes all 15 tools against
a local REST fixture. It needs no model or mnemory credentials.

## License

MIT
