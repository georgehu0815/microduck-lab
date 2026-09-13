import { rename, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..");
const apiDirectory = path.join(root, "app", "api");
const parkedApiDirectory = path.join(root, ".static-build-api");

await rename(apiDirectory, parkedApiDirectory);

try {
  const next = path.join(root, "node_modules", ".bin", "next");
  const exitCode = await new Promise((resolve, reject) => {
    const child = spawn(next, ["build"], {
      cwd: root,
      env: {
        ...process.env,
        DUCK_VIEWER_STATIC_EXPORT: "1",
        NEXT_PUBLIC_DUCK_VIEWER_STATIC_EXPORT: "1",
        NEXT_PUBLIC_DUCK_VIEWER_BASE_PATH:
          process.env.DUCK_VIEWER_BASE_PATH ?? "",
      },
      stdio: "inherit",
    });
    child.on("error", reject);
    child.on("exit", (code) => resolve(code ?? 1));
  });
  if (exitCode !== 0) {
    process.exitCode = exitCode;
  } else {
    await writeFile(path.join(root, "out", ".nojekyll"), "");
  }
} finally {
  await rename(parkedApiDirectory, apiDirectory);
}
