declare module "onnxruntime-web/wasm" {
  export const env: any;
  export const InferenceSession: any;
  export const Tensor: any;
}
declare module "onnxruntime-web/webgpu" {
  export const env: any;
  export const InferenceSession: any;
  export const Tensor: any;
}
declare module "electron" {
  export const shell: { openPath(path: string): Promise<string> };
}
