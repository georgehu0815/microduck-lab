import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

function load(filename) {
  const source = readFileSync(new URL(filename, import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } });
  const exports = {};
  vm.runInNewContext(outputText, { exports });
  return exports;
}

const { EXPERIMENTS } = load("./experiments.ts");
const { experimentGuidanceDisplay, rewardTermDisplay } = load("./experiment-labels.ts");
const english = (value) => value;
const chinese = (_value, translation) => translation;

test("English display preserves all eight authoritative experiment descriptions", () => {
  for (const experiment of EXPERIMENTS) {
    const before = JSON.stringify(experiment);
    const display = experimentGuidanceDisplay(experiment, english);
    assert.equal(display.requiredInput, experiment.guidance.requiredInput);
    assert.equal(JSON.stringify(display.steps), JSON.stringify(experiment.guidance.steps));
    assert.equal(JSON.stringify(display.distinctions), JSON.stringify(experiment.guidance.distinctions));
    assert.equal(display.output, experiment.guidance.output);
    assert.equal(display.rewardSummary, experiment.reward.summary);
    for (const term of experiment.reward.terms) {
      const translated = rewardTermDisplay(experiment.id, term, english);
      assert.equal(translated.label, term.label);
      assert.equal(translated.description, term.description);
    }
    assert.equal(JSON.stringify(experiment), before);
  }
});

test("Chinese guidance and reward labels cover every case without changing contracts", () => {
  for (const experiment of EXPERIMENTS) {
    const before = JSON.stringify(experiment);
    const display = experimentGuidanceDisplay(experiment, chinese);
    const strings = [display.requiredInput, ...display.steps, ...display.distinctions, display.output, display.rewardSummary];
    for (const text of strings) assert.match(text, /\p{Script=Han}/u, `${experiment.id}: ${text}`);
    assert.equal(display.steps.length, experiment.guidance.steps.length);
    assert.equal(display.distinctions.length, experiment.guidance.distinctions.length);
    for (const term of experiment.reward.terms) {
      const translated = rewardTermDisplay(experiment.id, term, chinese);
      assert.match(translated.label, /\p{Script=Han}/u, `${experiment.id}: ${term.key}`);
      assert.match(translated.description, /\p{Script=Han}/u, `${experiment.id}: ${term.key}`);
    }
    assert.equal(JSON.stringify(experiment), before);
  }
});
