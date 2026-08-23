import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, join } from "node:path";
import { spawnSync } from "node:child_process";

const executable = process.platform === "win32" ? "wasm-pack.exe" : "wasm-pack";
const cargoBin = join(process.env.CARGO_HOME || join(homedir(), ".cargo"), "bin");
const candidates = [executable, join(cargoBin, executable)];
const command = candidates.find((candidate) => candidate === executable || existsSync(candidate));

if (!command) {
  throw new Error("wasm-pack was not found. Install it with `cargo install wasm-pack`.");
}

const environment = { ...process.env, PATH: `${cargoBin}${delimiter}${process.env.PATH || ""}` };
const result = spawnSync(command, ["build", "wasm-core", "--target", "web", "--release", "--out-dir", "pkg"], {
  cwd: process.cwd(),
  env: environment,
  stdio: "inherit"
});

if (result.error) throw result.error;
process.exitCode = result.status ?? 1;
