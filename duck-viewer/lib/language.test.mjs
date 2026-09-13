import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("./language.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } });
const exports = {};
vm.runInNewContext(outputText, { exports });
const { normalizeLanguage, translate, createLanguageStore, LANGUAGE_STORAGE_KEY } = exports;

test("English is the default regardless of unknown or missing saved language", () => {
  for (const value of [null, undefined, "", "fr", "en-US", "zh-CN", {}, 1]) assert.equal(normalizeLanguage(value), "en");
  assert.equal(normalizeLanguage("zh"), "zh");
  assert.equal(translate("en", "Video library", "视频证据库"), "Video library");
  assert.equal(translate("zh", "Video library", "视频证据库"), "视频证据库");
});

test("language selection persists and notifies without resetting application state", () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  const store = createLanguageStore(() => storage);
  let updates = 0;
  const unsubscribe = store.subscribe(() => { updates += 1; });
  assert.equal(store.getLanguage(), "en");
  store.setLanguage("zh");
  assert.equal(values.get(LANGUAGE_STORAGE_KEY), "zh");
  assert.equal(store.getLanguage(), "zh");
  assert.equal(createLanguageStore(() => storage).getLanguage(), "zh");
  assert.equal(updates, 1);
  unsubscribe();
  store.setLanguage("en");
  assert.equal(updates, 1);
});

test("blocked browser storage still permits switching in memory", () => {
  const store = createLanguageStore(() => { throw new Error("Storage blocked"); });
  assert.equal(store.getLanguage(), "en");
  store.setLanguage("zh");
  assert.equal(store.getLanguage(), "zh");
  store.setLanguage("en");
  assert.equal(store.getLanguage(), "en");
});

test("cross-tab changes and cleared preferences synchronize", () => {
  const store = createLanguageStore(() => ({ getItem: () => "zh", setItem() {} }));
  assert.equal(store.getLanguage(), "zh");
  store.synchronize("en");
  assert.equal(store.getLanguage(), "en");
  store.synchronize("zh");
  assert.equal(store.getLanguage(), "zh");
  store.synchronize(null);
  assert.equal(store.getLanguage(), "en");
});
