/**
 * Shared utilities for the mnemory OpenCode plugin.
 *
 * Config parsing, session state management, prompt injection escaping,
 * and text extraction helpers.
 */

import type { RecallResponse } from "./client.js";

// ============================================================================
// Configuration
// ============================================================================

export type MnemoryConfig = {
  /** Mnemory server URL (e.g. "http://localhost:8050"). */
  url: string;
  /** Bearer token for mnemory authentication. */
  apiKey: string;
  /** Agent ID sent as X-Agent-Id header. Default: "opencode". */
  agentId: string;
  /** User ID sent as X-User-Id header. */
  userId: string;
  /** Minimum relevance score for recalled memories (0.0-1.0). Default: 0.5. */
  scoreThreshold: number;
  /** Include assistant messages in remember calls. Default: false. */
  includeAssistant: boolean;
  /** Default search mode for subsequent turns. Default: "search". */
  searchMode: "find" | "search";
  /** Use AI-powered "find" search on the first turn. Default: true. */
  findFirst: boolean;
  /** Include mnemory behavioral instructions in system prompt. Default: true. */
  managed: boolean;
  /** HTTP request timeout in milliseconds. Default: 30000. */
  timeout: number;
};

export function parseConfig(): MnemoryConfig {
  const url = (process.env.MNEMORY_URL || "http://localhost:8050").replace(
    /\/+$/,
    "",
  );
  const apiKey = process.env.MNEMORY_API_KEY || "";
  const agentId = process.env.MNEMORY_AGENT_ID || "opencode";
  const userId = process.env.MNEMORY_USER_ID || "";
  const scoreThreshold = Math.min(
    1,
    Math.max(0, Number(process.env.MNEMORY_SCORE_THRESHOLD) || 0.5),
  );
  const includeAssistant =
    (process.env.MNEMORY_INCLUDE_ASSISTANT || "false").toLowerCase() === "true";

  let searchMode: "find" | "search" = "search";
  const envMode = process.env.MNEMORY_SEARCH_MODE;
  if (envMode === "find" || envMode === "search") {
    searchMode = envMode;
  }

  const findFirst =
    (process.env.MNEMORY_FIND_FIRST || "true").toLowerCase() !== "false";
  const managed =
    (process.env.MNEMORY_MANAGED || "true").toLowerCase() !== "false";

  const timeout = Math.max(
    1000,
    Number(process.env.MNEMORY_TIMEOUT) || 30000,
  );

  return {
    url,
    apiKey,
    agentId,
    userId,
    scoreThreshold,
    includeAssistant,
    searchMode,
    findFirst,
    managed,
    timeout,
  };
}

// ============================================================================
// Logger
// ============================================================================

export type Logger = {
  info: (message: string, extra?: Record<string, unknown>) => void;
  warn: (message: string, extra?: Record<string, unknown>) => void;
  error: (message: string, extra?: Record<string, unknown>) => void;
};

/**
 * Create a logger that wraps OpenCode's client.app.log() API.
 * Falls back to console if the SDK client is unavailable.
 */
export function createLogger(
  client: {
    app: {
      log: (opts: {
        body: {
          service: string;
          level: string;
          message: string;
          extra?: Record<string, unknown>;
        };
      }) => Promise<unknown>;
    };
  } | null,
): Logger {
  const log = (
    level: string,
    message: string,
    extra?: Record<string, unknown>,
  ) => {
    if (client) {
      void client.app
        .log({
          body: { service: "mnemory", level, message, extra },
        })
        .catch(() => {});
    }
  };

  return {
    info: (message, extra?) => log("info", message, extra),
    warn: (message, extra?) => log("warn", message, extra),
    error: (message, extra?) => log("error", message, extra),
  };
}

// ============================================================================
// Session State
// ============================================================================

export type SessionState = {
  /** Canonical OpenCode session ID used by the OpenCode SDK. */
  opencodeSessionId: string | null;
  /** Mnemory-side session ID (returned by /api/recall). */
  mnemorySessionId: string | null;
  /** Number of messages already processed for auto-capture. */
  lastMessageCount: number;
  /** Conversation turn counter (0 = first turn). */
  turnCount: number;
  /** In-flight init recall promise. */
  recallPromise: Promise<RecallResponse | null> | null;
  /** Cached init result (instructions + core_memories from first recall). */
  recallResult: RecallResponse | null;
  /** In-flight per-turn search promise (started in chat.message). */
  searchPromise: Promise<RecallResponse | null> | null;
  /** Per-turn search results. REPLACED each turn. */
  lastSearchResults: RecallResponse["search_results"] | null;
  /** Whether this is a child/sub session. */
  isChildSession: boolean;
  /** AbortController for cancelling in-flight requests on cleanup. */
  abortController: AbortController;
};

const MAX_SESSIONS = 100;

export function createSessionStore(logger: Logger) {
  const sessions = new Map<string, SessionState>();

  function createState(): SessionState {
    return {
      opencodeSessionId: null,
      mnemorySessionId: null,
      lastMessageCount: 0,
      turnCount: 0,
      recallPromise: null,
      recallResult: null,
      searchPromise: null,
      lastSearchResults: null,
      isChildSession: false,
      abortController: new AbortController(),
    };
  }

  function cleanup(state: SessionState): void {
    try {
      state.abortController.abort();
    } catch {
      // Ignore abort errors
    }
    state.recallPromise = null;
    state.recallResult = null;
    state.searchPromise = null;
    state.lastSearchResults = null;
  }

  function getOrCreate(sessionKey: string): SessionState {
    let state = sessions.get(sessionKey);
    if (state) return state;

    // Evict oldest entries if at capacity (FIFO via Map iteration order)
    while (sessions.size >= MAX_SESSIONS) {
      const oldest = sessions.keys().next().value;
      if (oldest !== undefined) {
        const evicted = sessions.get(oldest);
        if (evicted) {
          logger.info(`Evicting session state: ${oldest}`);
          cleanup(evicted);
        }
        sessions.delete(oldest);
      }
    }

    state = createState();
    sessions.set(sessionKey, state);
    return state;
  }

  function remove(sessionKey: string): void {
    const state = sessions.get(sessionKey);
    if (state) {
      cleanup(state);
      sessions.delete(sessionKey);
    }
  }

  function reset(sessionKey: string): SessionState {
    const existing = sessions.get(sessionKey);
    if (existing) {
      cleanup(existing);
    }
    const state = createState();
    sessions.set(sessionKey, state);
    return state;
  }

  return { getOrCreate, remove, reset };
}

export type SessionStore = ReturnType<typeof createSessionStore>;

// ============================================================================
// Prompt Injection Safety
// ============================================================================

/** HTML-entity escape for memory content injected into prompts. */
export function escapeForPrompt(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}

// ============================================================================
// System Text Builder
// ============================================================================

/**
 * Build the system text to inject from a recall result.
 * Returns instructions separately from the memory context.
 */
export function buildSystemText(result: RecallResponse): {
  instructions: string | undefined;
  context: string;
} {
  const parts: string[] = [];

  if (result.core_memories) {
    parts.push(result.core_memories);
  }

  const memories = result.search_results
    ?.filter((r) => r.memory && r.memory.trim().length > 0)
    .map((r) => `- ${escapeForPrompt(r.memory)}`);

  if (memories && memories.length > 0) {
    parts.push(`## Recalled Memories\n${memories.join("\n")}`);
  }

  return {
    instructions: result.instructions || undefined,
    context: parts.join("\n\n"),
  };
}

// ============================================================================
// Text Extraction
// ============================================================================

/**
 * Extract text from OpenCode message parts array.
 * Takes only the last non-synthetic text part (skips tool narration).
 */
export function extractTextFromParts(parts: unknown[]): string | null {
  let lastText: string | null = null;

  for (const part of parts) {
    if (!part || typeof part !== "object") continue;
    const p = part as Record<string, unknown>;

    if (p.type !== "text") continue;
    if (typeof p.text !== "string") continue;

    // Skip synthetic parts (tool narration, etc.)
    if (p.synthetic) continue;

    const text = (p.text as string).trim();
    if (text) lastText = text;
  }

  return lastText;
}

/**
 * Extract text from message content (handles string and array-of-blocks formats).
 * Takes only the last non-synthetic text part.
 */
export function extractTextFromContent(content: unknown): string | null {
  if (typeof content === "string") {
    return content.trim() || null;
  }

  if (Array.isArray(content)) {
    return extractTextFromParts(content);
  }

  return null;
}

/**
 * Extract the last user+assistant exchange from session messages.
 * Only extracts non-synthetic text parts (skips intermediate agentic narration).
 */
export function extractLastExchange(
  messages: unknown[],
  afterIndex: number,
  includeAssistant: boolean,
): { user: string; assistant?: string; newCount: number } | null {
  const slice = messages.slice(afterIndex);
  if (slice.length < 1) return null;

  let lastUser: string | null = null;
  let lastAssistant: string | null = null;

  for (const msg of slice) {
    if (!msg || typeof msg !== "object") continue;
    const msgObj = msg as Record<string, unknown>;

    // Handle both flat messages and { info, parts } format
    let role: unknown;
    let content: unknown;

    if ("info" in msgObj && "parts" in msgObj) {
      // OpenCode SDK format: { info: { role: ... }, parts: [...] }
      const info = msgObj.info as Record<string, unknown>;
      role = info?.role;
      content = msgObj.parts;
    } else {
      role = msgObj.role;
      content = msgObj.content;
    }

    const text = extractTextFromContent(content);
    if (!text) continue;

    if (role === "user") {
      lastUser = text;
    } else if (role === "assistant" && includeAssistant) {
      lastAssistant = text;
    }
  }

  if (!lastUser) return null;

  return {
    user: lastUser,
    assistant: lastAssistant ?? undefined,
    newCount: messages.length,
  };
}

// ============================================================================
// Plugin Context
// ============================================================================

export type PluginContext = {
  config: MnemoryConfig;
  logger: Logger;
  store: SessionStore;
  directory: string;
  worktree: string;
  /** OpenCode SDK client for session messages. */
  sdkClient: unknown;
};
