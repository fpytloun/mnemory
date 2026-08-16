import { describe, expect, test } from "bun:test";
import { createHooks } from "./hooks.js";
import {
  createSessionStore,
  type Logger,
  type MnemoryConfig,
} from "./helpers.js";

const config: MnemoryConfig = {
  url: "http://localhost:8050",
  apiKey: "",
  agentId: "opencode",
  userId: "",
  scoreThreshold: 0.5,
  includeAssistant: false,
  searchMode: "search",
  findFirst: true,
  managed: true,
  timeout: 30000,
};

function createLogger(): Logger & { warnings: string[] } {
  const warnings: string[] = [];
  return {
    warnings,
    info: () => {},
    warn: (message) => warnings.push(message),
    error: () => {},
  };
}

describe("OpenCode session ID handling", () => {
  test("uses the canonical session ID after compaction for SDK message retrieval", async () => {
    const logger = createLogger();
    const paths: Record<string, string>[] = [];
    const hooks = createHooks({
      config,
      logger,
      store: createSessionStore(logger),
      directory: "/tmp",
      worktree: "/tmp",
      sdkClient: {
        session: {
          messages: async ({ path }: { path: Record<string, string> }) => {
            paths.push(path);
            return { data: [] };
          },
        },
      },
    });
    hooks.setClient({
      recall: async () => null,
      remember: async () => null,
    } as never);

    await hooks.event({
      event: {
        type: "session.created",
        properties: {
          sessionID: "s7Bid7D",
          info: { id: "ses_canonical" },
        },
      },
    });
    await hooks.event({
      event: {
        type: "session.compacted",
        properties: {
          sessionID: "s7Bid7D",
          messageCount: 0,
        },
      },
    });
    await hooks.event({
      event: {
        type: "session.created",
        properties: { sessionID: "s7Bid7D" },
      },
    });
    await hooks.event({
      event: {
        type: "session.idle",
        properties: { sessionID: "s7Bid7D" },
      },
    });

    expect(paths).toEqual([{ sessionID: "ses_canonical" }]);
  });

  test("skips capture when no canonical session ID is available", async () => {
    const logger = createLogger();
    let messagesCalled = false;
    const hooks = createHooks({
      config,
      logger,
      store: createSessionStore(logger),
      directory: "/tmp",
      worktree: "/tmp",
      sdkClient: {
        session: {
          messages: async () => {
            messagesCalled = true;
            return { data: [] };
          },
        },
      },
    });
    hooks.setClient({
      recall: async () => null,
      remember: async () => null,
    } as never);

    await hooks.event({
      event: {
        type: "session.idle",
        properties: { sessionID: "s7Bid7D" },
      },
    });

    expect(messagesCalled).toBe(false);
    expect(logger.warnings[0]).toContain("canonical OpenCode session ID");
  });
});
