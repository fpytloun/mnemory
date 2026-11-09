import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { test } from "node:test";

// Run inside a disposable environment with OpenClaw installed. No LLM/API key is needed:
// the real Gateway assembles and invokes tools against a local mnemory REST fixture.
test("OpenClaw exposes and invokes all mnemory tools without native memory conflicts", {
  timeout: 180_000,
}, async () => {
  const temp = mkdtempSync(join(tmpdir(), "mnemory-openclaw-"));
  const configPath = join(temp, "openclaw.json");
  const requests = [];
  const memory = { id: "memory-fixture", memory: "Mnemory integration result", score: 0.9 };
  const artifact = {
    id: "artifact-fixture", filename: "note.md", content_type: "text/markdown", size: 4,
  };
  const api = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    const url = new URL(req.url, "http://localhost");
    requests.push({ method: req.method, path: url.pathname, body: body ? JSON.parse(body) : null });
    let result = { results: [memory] };
    if (url.pathname === "/api/categories") result = { categories: [] };
    if (url.pathname.endsWith("/recent")) result = { text: memory.memory };
    if (url.pathname.endsWith("/ask")) {
      result = { answer: memory.memory, results: [memory], count: 1, queries: [], stats: {} };
    }
    if (url.pathname.includes("/batch")) {
      result = { results: [memory], errors: [], total: 1, succeeded: 1, failed: 0 };
    }
    if (url.pathname.endsWith("/artifacts")) {
      result = req.method === "GET" ? [artifact] : artifact;
    }
    if (url.pathname.endsWith(`/${artifact.id}`)) {
      result = { content: "note", total_size: 4, has_more: false };
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(result));
  });
  api.listen(0, "127.0.0.1");
  await once(api, "listening");
  const apiUrl = `http://127.0.0.1:${api.address().port}`;
  const portProbe = createServer();
  portProbe.listen(0, "127.0.0.1");
  await once(portProbe, "listening");
  const gatewayPort = portProbe.address().port;
  await new Promise((resolve) => portProbe.close(resolve));
  const env = {
    ...process.env,
    OPENCLAW_STATE_DIR: temp,
    OPENCLAW_CONFIG_PATH: configPath,
    OPENCLAW_GATEWAY_TOKEN: "mnemory-integration-test-token",
    MNEMORY_URL: apiUrl,
    MNEMORY_USER_ID: "integration-test",
  };
  let gateway;
  let logs = "";
  try {
    const result = JSON.parse(execFileSync("npm", [
      "pack", "--json", "--pack-destination", temp,
    ], { encoding: "utf8" }));
    const packed = Array.isArray(result) ? result[0] : result["@fpytloun/openclaw-mnemory"];
    execFileSync("openclaw", [
      "plugins", "install", "--force", "--accept-capabilities", join(temp, packed.filename),
    ], { env, stdio: "pipe" });
    const installed = JSON.parse(readFileSync(configPath, "utf8"));
    writeFileSync(configPath, JSON.stringify({
      ...installed,
      gateway: { mode: "local", auth: { mode: "token", token: env.OPENCLAW_GATEWAY_TOKEN } },
      agents: { defaults: { workspace: join(temp, "workspace") } },
      tools: { profile: "full" },
      plugins: {
        ...installed.plugins,
        allow: ["mnemory", "memory-core"],
        slots: { memory: "mnemory" },
        entries: {
          ...installed.plugins?.entries,
          "memory-core": { enabled: false },
          mnemory: {
            enabled: true,
            config: { url: apiUrl, userId: "integration-test", autoRecall: false, autoCapture: false },
          },
        },
      },
    }));
    gateway = spawn("openclaw", [
      "gateway", "run", "--bind", "loopback", "--port", String(gatewayPort),
    ], { env, stdio: ["ignore", "pipe", "pipe"] });
    gateway.stdout.on("data", (chunk) => { logs += chunk; });
    gateway.stderr.on("data", (chunk) => { logs += chunk; });
    const gatewayUrl = `http://127.0.0.1:${gatewayPort}`;
    let ready = false;
    for (let i = 0; i < 180; i++) {
      if (gateway.exitCode !== null) throw new Error(`Gateway exited: ${logs}`);
      try {
        const response = await fetch(`${gatewayUrl}/healthz`, { signal: AbortSignal.timeout(1000) });
        if (response.ok) { ready = true; break; }
      } catch {}
      await delay(250);
    }
    assert.ok(ready, `Gateway did not become ready: ${logs}`);
    const cases = [
      ["memory_list", {}, "GET", "/api/memories"],
      ["memory_search", { query: "integration" }, "POST", "/api/memories/search"],
      ["memory_find", { question: "integration" }, "POST", "/api/memories/find"],
      ["memory_ask", { question: "integration" }, "POST", "/api/memories/ask"],
      ["memory_add", { content: "integration" }, "POST", "/api/memories"],
      ["memory_add_batch", { memories: [{ content: "integration" }] }, "POST", "/api/memories/batch"],
      ["memory_update", { memory_id: memory.id, content: "updated" }, "PUT", `/api/memories/${memory.id}`],
      ["memory_delete", { memory_id: memory.id }, "DELETE", `/api/memories/${memory.id}`],
      ["memory_delete_batch", { memory_ids: [memory.id] }, "POST", "/api/memories/batch/delete"],
      ["memory_categories", {}, "GET", "/api/categories"],
      ["memory_recent", {}, "GET", "/api/memories/recent"],
      ["memory_save_artifact", { memory_id: memory.id, content: "note" }, "POST", `/api/memories/${memory.id}/artifacts`],
      ["memory_get_artifact", { memory_id: memory.id, artifact_id: artifact.id }, "GET", `/api/memories/${memory.id}/artifacts/${artifact.id}`],
      ["memory_list_artifacts", { memory_id: memory.id }, "GET", `/api/memories/${memory.id}/artifacts`],
      ["memory_delete_artifact", { memory_id: memory.id, artifact_id: artifact.id }, "DELETE", `/api/memories/${memory.id}/artifacts/${artifact.id}`],
    ];
    for (const [tool, args, method, path] of cases) {
      requests.length = 0;
      const response = await fetch(`${gatewayUrl}/tools/invoke`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${env.OPENCLAW_GATEWAY_TOKEN}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ tool, args }),
        signal: AbortSignal.timeout(60_000),
      }).catch((error) => {
        throw new Error(`${tool} invocation failed: ${logs}`, { cause: error });
      });
      const body = await response.json();
      assert.equal(response.status, 200, `${tool}: ${JSON.stringify(body)}\n${logs}`);
      assert.equal(body.ok, true, `${tool}: ${JSON.stringify(body)}`);
      assert.ok(requests.some((req) => req.method === method && req.path === path),
        `${tool} did not reach mnemory: ${JSON.stringify(body)}`);
      assert.ok(!body.result?.isError, `${tool}: ${JSON.stringify(body)}`);
      if (tool === "memory_search") {
        assert.deepEqual(body.result.details.memories, [memory]);
        assert.ok(body.result.content.some((item) =>
          item.type === "text" && item.text.includes(memory.memory)));
      }
    }
    assert.doesNotMatch(logs, /plugin tool name conflict \(mnemory\)/);
  } finally {
    if (gateway && gateway.exitCode === null) {
      gateway.kill("SIGTERM");
      await Promise.race([once(gateway, "exit"), delay(5000)]);
      if (gateway.exitCode === null) {
        gateway.kill("SIGKILL");
        await once(gateway, "exit");
      }
    }
    await new Promise((resolve) => api.close(resolve));
    rmSync(temp, { recursive: true, force: true });
  }
});
