import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("lab.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});

function loadClient(fetch) {
  const exports = {};
  vm.runInNewContext(outputText, {
    exports,
    fetch,
    URLSearchParams,
    window: { location: { search: "?lab=localhost:9876" } },
  });
  return exports;
}

test("refresh bypasses caches, forwards cancellation, and discovers newly saved policies", async () => {
  const controller = new AbortController();
  const calls = [];
  const policies = [{ id: "run:first", label: "first", group: "runs" }];
  const client = loadClient(async (url, options) => {
    calls.push({ url, options });
    return Response.json({ policies: [...policies, policies[0]] });
  });
  assert.equal((await client.fetchPolicies(controller.signal)).length, 1);
  policies.unshift({ id: "run:newest", label: "newest", group: "runs" });
  const refreshed = await client.fetchPolicies(controller.signal);
  assert.deepEqual(Array.from(refreshed, (policy) => policy.id), ["run:newest", "run:first"]);
  for (const call of calls) {
    assert.equal(call.url, "http://localhost:9876/policies");
    assert.equal(call.options.cache, "no-store");
    assert.equal(call.options.signal, controller.signal);
  }
});

test("refresh reports server errors rather than replacing policies with an empty list", async () => {
  const client = loadClient(async () => new Response(null, { status: 503 }));
  await assert.rejects(client.fetchPolicies(), /policies fetch failed: 503/);
});

test("refresh propagates aborts so superseded requests cannot replace the latest list", async () => {
  const controller = new AbortController();
  controller.abort();
  const client = loadClient(async (_url, options) => {
    options.signal.throwIfAborted();
  });
  await assert.rejects(client.fetchPolicies(controller.signal), { name: "AbortError" });
});
