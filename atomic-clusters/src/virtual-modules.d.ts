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

// The pinned ORT Web snapshot does not publish declarations for these
// renderer-specific export subpaths. The embedding provider only relies on
// their runtime Tensor/session surface.
declare module "onnxruntime-web/wasm" {
  export const env: any;
  export const Tensor: any;
  export const InferenceSession: any;
}

declare module "onnxruntime-web/webgpu" {
  export const env: any;
  export const Tensor: any;
  export const InferenceSession: any;
}

declare module "electron" {
  export const shell: { openPath(path: string): Promise<string> };
}
