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

test("scene fetch preserves the global endpoint without a duck id", async () => {
  const calls = [];
  const client = loadClient(async (url, options) => {
    calls.push({ url, options });
    return Response.json({ bodies: [], meshes: [], geoms: [] });
  });
  await client.fetchScene();
  assert.equal(calls[0].url, "http://localhost:9876/scene");
  assert.equal(calls[0].options, undefined);
});

test("scene fetch encodes duck and version and forwards cancellation", async () => {
  const controller = new AbortController();
  const calls = [];
  const client = loadClient(async (url, options) => {
    calls.push({ url, options });
    return Response.json({ bodies: [], meshes: [], geoms: [] });
  });
  await client.fetchScene("duck/one", "swing v2", controller.signal);
  assert.equal(
    calls[0].url,
    "http://localhost:9876/scene?duck=duck%2Fone&version=swing%20v2"
  );
  assert.equal(calls[0].options.signal, controller.signal);
});
