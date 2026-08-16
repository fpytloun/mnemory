/**
 * Lifecycle hooks for the mnemory OpenCode plugin.
 *
 * Handles automatic recall (per-turn search injection into system prompt),
 * automatic capture (remember after each exchange), and compaction resilience.
 */

import type { MnemoryClient, RecallResponse } from "./client.js";
import type {
  Logger,
  MnemoryConfig,
  PluginContext,
  SessionState,
  SessionStore,
} from "./helpers.js";
import {
  buildSystemText,
  extractLastExchange,
  extractTextFromParts,
} from "./helpers.js";

// ============================================================================
// Recall Helpers
// ============================================================================

/**
 * Start a non-blocking init recall (instructions + core memories, no query).
 * The per-turn search with query happens in chat.message.
 */
function startInitRecall(
  state: SessionState,
  client: MnemoryClient,
  config: MnemoryConfig,
  logger: Logger,
  directory: string,
): void {
  const isNew = !state.mnemorySessionId;
  state.recallResult = null;
  state.lastSearchResults = null;

  logger.info(
    `Recall init started (new_session=${isNew})`,
  );

  state.recallPromise = client
    .recall(
      {
        sessionId: state.mnemorySessionId ?? undefined,
        includeInstructions: config.managed,
        managed: config.managed,
        instructionMode: "personality",
        scoreThreshold: config.scoreThreshold,
        context: `Working directory: ${directory}`,
      },
      state.abortController.signal,
    )
    .then((result) => {
      if (result) {
        state.mnemorySessionId = result.session_id;
        state.recallResult = result;
        const stats = result.stats;
        logger.info(
          `Recall init complete (session=${result.session_id}, ` +
            `has_instructions=${!!result.instructions}, ` +
            `has_core=${!!result.core_memories}, ` +
            `core_count=${stats?.core_count ?? 0}, ` +
            `latency=${stats?.latency_ms ?? 0}ms)`,
        );
      }
      return result;
    })
    .catch((err) => {
      if ((err as Error).name !== "AbortError") {
        logger.warn(`Recall init failed: ${String(err)}`);
      }
      return null;
    });
}

/**
 * Start a per-turn search with the user's query.
 * Returns a promise that resolves to the search result.
 */
function startPerTurnSearch(
  state: SessionState,
  client: MnemoryClient,
  config: MnemoryConfig,
  logger: Logger,
  query: string,
  directory: string,
): void {
  const isFirstTurn = state.turnCount === 0;
  const searchMode: "find" | "search" =
    isFirstTurn && config.findFirst ? "find" : config.searchMode;

  logger.info(
    `Recall search started (mode=${searchMode}, query_len=${query.length}, ` +
      `turn=${state.turnCount}, session=${state.mnemorySessionId ?? "none"})`,
  );

  // Clear previous search results before attempting a new search.
  state.lastSearchResults = null;

  // If init completed, do a search-only call with session_id.
  // If init failed (no recallResult), do a combined call that also
  // fetches instructions + core_memories in one round-trip.
  const needsInit = !state.recallResult;

  state.searchPromise = client
    .recall(
      {
        sessionId: state.mnemorySessionId ?? undefined,
        query,
        searchMode,
        scoreThreshold: config.scoreThreshold,
        context: `Working directory: ${directory}`,
        ...(needsInit
          ? {
              includeInstructions: config.managed,
              managed: config.managed,
              instructionMode: "personality",
            }
          : {}),
      },
      state.abortController.signal,
    )
    .then((result) => {
      if (result) {
        state.mnemorySessionId = result.session_id;

        // If this was a combined call (init failed), cache the static parts
        if (needsInit) {
          state.recallResult = result;
        }

        // REPLACE per-turn search results (never accumulate).
        state.lastSearchResults = result.search_results ?? [];

        // Only advance turnCount on success so a transient failure on the
        // first turn doesn't skip the intended "find" mode on retry.
        state.turnCount++;

        const stats = result.stats;
        logger.info(
          `Recall search complete (search_count=${stats?.search_count ?? 0}, ` +
            `new_count=${stats?.new_count ?? 0}, ` +
            `known_skipped=${stats?.known_skipped ?? 0}, ` +
            `latency=${stats?.latency_ms ?? 0}ms)`,
        );
      } else {
        logger.warn("Recall search returned null (API error)");
      }
      return result;
    })
    .catch((err) => {
      if ((err as Error).name !== "AbortError") {
        logger.warn(`Recall search failed: ${String(err)}`);
      }
      return null;
    });
}

// ============================================================================
// Hook Factory
// ============================================================================

/** Search timeout in system.transform to avoid blocking the LLM call. */
const SEARCH_TIMEOUT_MS = 8000;

export function createHooks(ctx: PluginContext) {
  const { config, logger, store, directory } = ctx;
  // Lazy-initialized client (set by index.ts after client creation)
  let client: MnemoryClient;

  function setClient(c: MnemoryClient) {
    client = c;
  }

  // ========================================================================
  // event handler — session lifecycle
  // ========================================================================

  const eventHandler = async (input: { event: unknown }) => {
    const event = input.event as {
      type?: string;
      properties?: Record<string, unknown>;
    };
    if (!event?.type) return;

    // --- session.created ---
    if (event.type === "session.created") {
      const props = event.properties as {
        sessionID?: string;
        info?: { id?: string; parentID?: string };
      };
      const sessionKey = props?.sessionID ?? props?.info?.id;
      if (!sessionKey) return;

      const state = store.getOrCreate(sessionKey);
      if (props?.info?.id) {
        state.opencodeSessionId = props.info.id;
      }

      // Detect child sessions (sub-agents) — skip recall to avoid duplicate context
      if (props?.info?.parentID) {
        state.isChildSession = true;
        logger.info(`Child session detected: ${sessionKey} (parent: ${props.info.parentID})`);
        return;
      }

      // Start non-blocking init recall
      startInitRecall(state, client, config, logger, directory);
    }

    // --- session.idle ---
    if (event.type === "session.idle") {
      const props = event.properties as { sessionID?: string };
      const sessionId = props?.sessionID;
      if (!sessionId) return;

      const state = store.getOrCreate(sessionId);
      if (state.isChildSession) return;
      const opencodeSessionId = state.opencodeSessionId;
      if (!opencodeSessionId) {
        logger.warn(
          `Auto-capture skipped: canonical OpenCode session ID is unavailable for ${sessionId}`,
        );
        return;
      }

      try {
        // Get session messages via OpenCode SDK.
        const sdkClient = ctx.sdkClient as {
          session?: {
            messages?: (opts: {
              path: Record<string, string>;
            }) => Promise<{ data?: unknown[] }>;
          };
        };

        const response = await sdkClient?.session?.messages?.({
          path: { sessionID: opencodeSessionId },
        });

        const messages = (response?.data ?? []) as unknown[];
        if (messages.length === 0) return;

        const exchange = extractLastExchange(
          messages,
          state.lastMessageCount,
          config.includeAssistant,
        );
        if (!exchange) return;

        state.lastMessageCount = exchange.newCount;

        // Build messages array for /api/remember
        const rememberMessages: Array<{ role: string; content: string }> = [
          { role: "user", content: exchange.user },
        ];
        if (exchange.assistant) {
          rememberMessages.push({
            role: "assistant",
            content: exchange.assistant,
          });
        }

        // Fire-and-forget (with abort signal for cleanup)
        client.remember(
          {
            sessionId: state.mnemorySessionId ?? undefined,
            messages: rememberMessages,
            labels: {
              session_id: sessionId,
              source: "opencode",
            },
          },
          state.abortController.signal,
        );

        logger.info("Sent conversation exchange for memory extraction");
      } catch (err) {
        logger.warn(`Auto-capture failed: ${String(err)}`);
      }
    }

    // --- session.compacted ---
    if (event.type === "session.compacted") {
      const props = event.properties as {
        sessionID?: string;
        messageCount?: number;
      };
      const sessionId = props?.sessionID;
      if (!sessionId) return;

      // Preserve the post-compaction message count so session.idle
      // doesn't re-scan the compacted transcript from the beginning
      // (which would cause duplicate remember calls).
      const postCompactionCount = props?.messageCount ?? 0;

      // Reset recall state but preserve message tracking and the canonical
      // OpenCode ID required by the SDK.
      const previousState = store.getOrCreate(sessionId);
      const state = store.reset(sessionId);
      state.lastMessageCount = postCompactionCount;
      state.opencodeSessionId = previousState.opencodeSessionId;
      logger.info(
        `Session compacted, resetting state: ${sessionId} (messageCount=${postCompactionCount})`,
      );
      startInitRecall(state, client, config, logger, directory);
    }

    // --- session.deleted ---
    if (event.type === "session.deleted") {
      const props = event.properties as { sessionID?: string };
      const sessionId = props?.sessionID;
      if (sessionId) {
        store.remove(sessionId);
        logger.info(`Session deleted, cleaned up state: ${sessionId}`);
      }
    }
  };

  // ========================================================================
  // chat.message — start per-turn search
  // ========================================================================

  const chatMessageHandler = async (
    input: { sessionID: string },
    output: { parts: unknown[] },
  ) => {
    const sessionId = input.sessionID;
    if (!sessionId) return;

    const state = store.getOrCreate(sessionId);
    if (state.isChildSession) return;

    // Extract user text from message parts
    const userText = extractTextFromParts(output.parts as unknown[]);
    if (!userText) return;

    // If no init recall has started yet (e.g., resumed session), start one now
    if (!state.recallPromise && !state.recallResult) {
      startInitRecall(state, client, config, logger, directory);
    }

    // Start per-turn search with the user's query (non-blocking)
    startPerTurnSearch(state, client, config, logger, userText, directory);
  };

  // ========================================================================
  // experimental.chat.system.transform — inject memories into system prompt
  // ========================================================================

  const systemTransformHandler = async (
    input: { sessionID?: string },
    output: { system: string[] },
  ) => {
    const sessionId = input.sessionID;
    if (!sessionId) return;

    const state = store.getOrCreate(sessionId);
    if (state.isChildSession) return;

    // Await the init recall promise if it hasn't resolved yet
    if (state.recallPromise && !state.recallResult) {
      try {
        await state.recallPromise;
      } catch {
        // Already logged in startInitRecall
      }
    }

    // Await the per-turn search promise with a timeout
    if (state.searchPromise) {
      try {
        await Promise.race([
          state.searchPromise,
          new Promise((resolve) => setTimeout(resolve, SEARCH_TIMEOUT_MS)),
        ]);
      } catch {
        // Already logged in startPerTurnSearch
      }
      // Clear the promise so we don't await it again on tool-triggered LLM calls
      state.searchPromise = null;
    }

    // Nothing to inject
    if (!state.recallResult && !state.lastSearchResults?.length) return;

    // Build injection from cached init result + per-turn search results
    const searchResults = state.lastSearchResults ?? [];
    const combined: RecallResponse = {
      session_id: state.mnemorySessionId ?? "",
      instructions: state.recallResult?.instructions,
      core_memories: state.recallResult?.core_memories,
      search_results: searchResults,
    };

    const { instructions, context } = buildSystemText(combined);

    logger.info(
      `Injecting context (instructions=${!!instructions}, ` +
        `core_memories=${!!state.recallResult?.core_memories}, ` +
        `search_results=${searchResults.length}, ` +
        `turn=${state.turnCount}, session=${state.mnemorySessionId ?? "none"})`,
    );

    // Inject instructions at the beginning (cacheable by providers)
    if (instructions) {
      output.system.push(instructions);
    }

    // Inject core memories + search results
    if (context) {
      output.system.push(context);
    }
  };

  // ========================================================================
  // experimental.session.compacting — preserve memories across compaction
  // ========================================================================

  const compactingHandler = async (
    input: { sessionID: string },
    output: { context: string[] },
  ) => {
    const sessionId = input.sessionID;
    if (!sessionId) return;

    const state = store.getOrCreate(sessionId);
    if (state.isChildSession) return;

    // Inject core memories into compaction context so they survive
    if (state.recallResult?.core_memories) {
      output.context.push(
        "## Long-term Memory Context\n" +
          "The following are the user's stored memories. " +
          "Preserve any references to these in the continuation summary.\n\n" +
          state.recallResult.core_memories,
      );
      logger.info("Injected core memories into compaction context");
    }
  };

  return {
    setClient,
    event: eventHandler,
    "chat.message": chatMessageHandler,
    "experimental.chat.system.transform": systemTransformHandler,
    "experimental.session.compacting": compactingHandler,
  };
}
