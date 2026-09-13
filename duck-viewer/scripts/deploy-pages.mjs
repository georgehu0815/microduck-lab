import { execFileSync } from "node:child_process";
import { access, cp, mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

const viewer = path.resolve(import.meta.dirname, "..");
const root = path.resolve(viewer, "..");
const output = path.join(viewer, "out");
await access(path.join(output, "index.html"));
await access(path.join(output, ".nojekyll"));
const repository = "https://github.com/georgehu0815/microduck-lab.git";
const staging = await mkdtemp(path.join(tmpdir(), "microduck-pages-"));
const git = (...args) => execFileSync("git", args, { cwd: staging, stdio: "inherit" });
git("init", "-b", "gh-pages");
git("remote", "add", "origin", repository);
const existing = execFileSync("git", ["ls-remote", "--heads", repository, "gh-pages"], { encoding: "utf8" }).trim();
if (existing) {
  git("fetch", "--depth=1", "origin", "gh-pages");
  git("checkout", "-B", "gh-pages", "FETCH_HEAD");
  git("rm", "-r", "--ignore-unmatch", ".");
}
await cp(output, staging, { recursive: true });
git("config", "user.name", "George Hu");
git("config", "user.email", "116222457+georgehu0815@users.noreply.github.com");
git("add", "--all");
const changes = execFileSync("git", ["status", "--porcelain"], { cwd: staging, encoding: "utf8" });
if (changes.trim()) {
  const revision = execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: root, encoding: "utf8" }).trim();
  git("commit", "-m", `Deploy duck-viewer from ${revision}`);
  git("push", "origin", "HEAD:gh-pages");
} else {
  console.log("Published files already match this build.");
}
console.log("GitHub Pages: https://georgehu0815.github.io/microduck-lab/");
