/** Build-time module that synchronously installs the generated WASM exports. */
declare module "atomic-clusters-wasm-bootstrap" {
  export {};
}

declare module "atomic-clusters-title-onnxruntime-web" {
  export const env: {
    wasm?: {
      wasmPaths?: string | Record<string, string>;
      wasmBinary?: ArrayBuffer;
      proxy?: boolean;
    };
  };
}

declare module "electron" {
  export const shell: { openPath(path: string): Promise<string> };
}
