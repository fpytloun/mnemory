import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

test("npm package contains a loadable compiled OpenClaw entry", () => {
  const temp = mkdtempSync(join(tmpdir(), "mnemory-package-"));
  try {
    const result = JSON.parse(execFileSync("npm", [
      "pack", "--json", "--pack-destination", temp,
    ], { encoding: "utf8" }));
    const packed = Array.isArray(result) ? result[0] : result["@fpytloun/openclaw-mnemory"];
    assert.ok(packed, "npm pack did not return package metadata");
    execFileSync("tar", ["-xzf", join(temp, packed.filename), "-C", temp]);
    const packageDir = join(temp, "package");
    const manifest = JSON.parse(readFileSync(join(packageDir, "package.json"), "utf8"));
    assert.deepEqual(manifest.openclaw.extensions, ["./index.ts"]);
    for (const file of ["index.js", "client.js", "config.js"]) {
      assert.ok(packed.files.some((entry) => entry.path === `dist/${file}`), file);
    }
    execFileSync("npm", ["install", "--omit=dev", "--ignore-scripts"], {
      cwd: packageDir,
      stdio: "pipe",
    });
    execFileSync("node", ["--input-type=module", "-e",
      "const plugin = await import('./dist/index.js'); if (!plugin.default) process.exit(1)"], {
      cwd: packageDir,
      stdio: "pipe",
    });
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
});
