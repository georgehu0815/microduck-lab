import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const { outputText } = ts.transpileModule(readFileSync(new URL("./classroom.ts", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const exports = {};
vm.runInNewContext(outputText, {
  exports,
  require(name) {
    assert.equal(name, "./static-assets");
    return { publicAssetUrl: (path) => path };
  },
});
const { classroomAssetUrl, parseClassroomCatalog } = exports;
const assets = new URL("../public/classroom-assets/", import.meta.url);
const readCatalog = () => JSON.parse(readFileSync(new URL("catalog.json", assets)));

test("classroom lists every source lesson and both complete language editions", () => {
  const catalog = parseClassroomCatalog(readCatalog());
  let totalSlides = 0;
  for (const [language, folder] of [["en", ""], ["zh", "zh-CN/"]]) {
    const source = JSON.parse(readFileSync(new URL(`../../docs/classroom-ppt/${folder}deck-manifest.json`, import.meta.url)));
    assert.equal(catalog.lessons.length, source.length);
    for (const original of source) {
      const lesson = catalog.lessons.find((item) => item.id === original.slug);
      assert.ok(lesson);
      const edition = lesson.editions[language];
      assert.equal(edition.slides.length, original.slides);
      assert.equal(lesson.title[language], original.title);
      assert.equal(edition.videos.length, original.videos);
      assert.deepEqual(readFileSync(new URL(edition.pptx, assets)),
        readFileSync(new URL(`../../docs/classroom-ppt/${folder}${original.file}`, import.meta.url)));
      totalSlides += edition.slides.length;
    }
  }
  assert.equal(catalog.lessons.length, 8);
  assert.equal(totalSlides, 644);
});

test("all published slides, PDFs, decks and videos are hash-bound and nonempty", () => {
  const manifest = JSON.parse(readFileSync(new URL("manifest.json", assets)));
  const catalog = readCatalog();
  for (const lesson of catalog.lessons) {
    for (const edition of Object.values(lesson.editions)) {
      for (const file of [edition.pptx, edition.pdf, ...edition.slides.map((slide) => slide.image),
        ...edition.videos.map((video) => video.file)]) assert.ok(manifest.files[file], file);
    }
  }
  for (const [file, entry] of Object.entries(manifest.files)) {
    const bytes = readFileSync(new URL(file, assets));
    assert.ok(bytes.length > 0, file);
    assert.equal(bytes.length, entry.bytes, file);
    assert.equal(createHash("sha256").update(bytes).digest("hex"), entry.sha256, file);
  }
});

test("classroom paths preserve the Pages prefix and reject traversal or external assets", () => {
  assert.equal(classroomAssetUrl("dance/en/slide-001.jpg", (path) => `/microduck-lab${path}`),
    "/microduck-lab/classroom-assets/dance/en/slide-001.jpg");
  for (const path of ["", "../secret", "/external", "https://example.com", "a//b", "a/../b", "a\\b", "%2e%2e/x"])
    assert.throws(() => classroomAssetUrl(path));
});

test("catalog rejects missing language, empty slides, duplicate identities and unsafe files", () => {
  for (const change of [
    (value) => { value.schemaVersion = 2; },
    (value) => { value.lessons[0].editions.zh = null; },
    (value) => { value.lessons[0].editions.en.slides = []; },
    (value) => { value.lessons[0].editions.en.slides[0].image = "../outside.jpg"; },
    (value) => { value.lessons[0].editions.en.pptx = "https://example.com/slides.pptx"; },
    (value) => { value.lessons[1].id = value.lessons[0].id; },
    (value) => { value.lessons[1].order = value.lessons[0].order; },
  ]) {
    const value = readCatalog();
    change(value);
    assert.throws(() => parseClassroomCatalog(value));
  }
});
