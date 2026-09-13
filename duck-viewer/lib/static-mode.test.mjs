import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const studioPath = new URL("../components/Studio.tsx", import.meta.url);
const viewerPath = new URL("../components/Viewer.tsx", import.meta.url);
const armStudioPath = new URL("../components/ArmStudio.tsx", import.meta.url);

test("static mode guards backend entrypoints and panels", async () => {
  const [studio, viewer, armStudio] = await Promise.all([
    readFile(studioPath, "utf8"),
    readFile(viewerPath, "utf8"),
    readFile(armStudioPath, "utf8"),
  ]);

  assert.match(studio, /async function runAction[\s\S]*?if \(IS_STATIC_EXPORT\)/);
  assert.equal((studio.match(/\{!IS_STATIC_EXPORT && <div/g) ?? []).length, 3);
  assert.match(studio, /<AnimPanel variant="embedded"/);
  assert.match(studio, /<PolicyPanel[\s\S]*?variant="embedded"/);
  assert.match(studio, /<TeachPanel clientRef=\{clientRef\} variant="embedded"/);
  assert.match(studio, /!IS_STATIC_EXPORT && <button[\s\S]*?runAction\("train"\)/);
  assert.match(studio, /!IS_STATIC_EXPORT && <button[\s\S]*?runAction\("eval"\)/);
  assert.match(studio, /!IS_STATIC_EXPORT && <button[\s\S]*?runAction\("render"\)/);
  assert.match(studio, /!IS_STATIC_EXPORT && !job\.artifacts\.onnx[\s\S]*?runAction\("export"\)/);
  assert.match(studio, /!IS_STATIC_EXPORT && job\.phase === "running"[\s\S]*?runAction\("cancel"\)/);
  assert.match(studio, /!IS_STATIC_EXPORT \|\| kind === "sheet" \|\| kind === "video"/);
  assert.match(studio, /IS_STATIC_EXPORT\s*\? artifactHref\("video"\)/);
  assert.match(viewer, /!offline && <RecordPanel/);
  assert.match(viewer, /!offline && showPolicyTools/);
  assert.match(viewer, /!offline && showTeachTools/);
  assert.match(viewer, /!offline && showAnimationTools/);
  assert.match(armStudio, /if \(IS_STATIC_EXPORT\) return null;/);
  assert.match(armStudio, /!IS_STATIC_EXPORT && <button aria-pressed=\{workspaceView === "live"\}/);
});
