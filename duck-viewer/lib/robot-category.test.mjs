import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("./robot-category.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const exports = {};
vm.runInNewContext(outputText, { exports });
const { newestPoliciesPerSkill, policyRobotCategory, policySkillKey, policySupportsCategory } = exports;

test("older Microduck policies remain compatible with WingPod", () => {
  const policy = { id: "run:dance", label: "dance", group: "runs", path: "/dance.onnx" };
  assert.equal(policyRobotCategory(policy), "microduck");
  assert.equal(policySupportsCategory(policy, "microduck"), true);
  assert.equal(policySupportsCategory(policy, "wingpod"), true);
  assert.equal(policySupportsCategory(policy, "humanoid"), false);
});

test("explicit and inferred humanoid policies stay isolated", () => {
  const explicit = { id: "run:walker", label: "walker", category: "humanoid" };
  const inferred = { id: "run:humanoid-walk", label: "walk" };
  assert.equal(policyRobotCategory(explicit), "humanoid");
  assert.equal(policyRobotCategory(inferred), "humanoid");
  assert.equal(policySupportsCategory(explicit, "humanoid"), true);
  assert.equal(policySupportsCategory(explicit, "wingpod"), false);
});

test("WingPod-specific policies also appear in the WingPod category", () => {
  const policy = { id: "studio:wingpod/tennis", label: "tennis", category: "wingpod" };
  assert.equal(policySupportsCategory(policy, "wingpod"), true);
  assert.equal(policySupportsCategory(policy, "microduck"), false);
});

test("only the newest Studio policy remains for each skill", () => {
  const policies = [
    { id: "studio:dance/old", label: "old", group: "studio", recipe: "dance", mtime: 10 },
    { id: "studio:swing/latest", label: "latest", group: "studio", recipe: "swing", mtime: 30 },
    { id: "studio:dance/latest", label: "latest", group: "studio", recipe: "dance", mtime: 20 },
  ];
  assert.deepEqual(
    Array.from(newestPoliciesPerSkill(policies), (policy) => policy.id),
    ["studio:swing/latest", "studio:dance/latest"],
  );
});

test("run exports replace older checkpoints from the same action", () => {
  const policies = [
    { id: "run:first-gait", label: "first-gait", group: "runs", mtime: 30 },
    { id: "ckpt:first-gait@3000k", label: "first-gait@3000k", group: "checkpoints", mtime: 20 },
    { id: "ckpt:first-gait@2500k", label: "first-gait@2500k", group: "checkpoints", mtime: 10 },
  ];
  assert.deepEqual(Array.from(newestPoliciesPerSkill(policies), (policy) => policy.id), ["run:first-gait"]);
});

test("hashed teach runs collapse by skill while shipped actions remain distinct", () => {
  const policies = [
    { id: "pollen:ball_kick_left", label: "ball_kick_left", group: "pollen" },
    { id: "pollen:ball_kick_right", label: "ball_kick_right", group: "pollen" },
    { id: "run:teach-stand-ed74ce", label: "teach-stand-ed74ce", group: "runs", mtime: 10 },
    { id: "run:teach-stand-36de72", label: "teach-stand-36de72", group: "runs", mtime: 20 },
  ];
  assert.equal(policySkillKey(policies[2]), "teach:stand");
  assert.deepEqual(
    Array.from(newestPoliciesPerSkill(policies), (policy) => policy.id),
    ["pollen:ball_kick_left", "pollen:ball_kick_right", "run:teach-stand-36de72"],
  );
});