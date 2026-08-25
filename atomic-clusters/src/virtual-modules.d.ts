/** Build-time module that synchronously installs the generated WASM exports. */
declare module "atomic-clusters-wasm-bootstrap" {
  export {};
}

declare module "electron" {
  export const shell: { openPath(path: string): Promise<string> };
}
