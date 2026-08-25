# Standalone LLM playground

The playground exercises only the explicitly supplied Qwen3 model assets. It
does not load the Obsidian vault, cluster results, title cache, or title
prompt/validation code, and it does not write a diagnostic log.

From `atomic-clusters/`, run:

```bash
npm run llm:playground -- \
  --model-dir "/path/to/qwen3-0.6b-q4f16/<revision>" \
  --prompt "대한민국의 수도는?"
```

For a prompt containing newlines, pipe UTF-8 text on stdin:

```bash
cat prompt.txt | npm run llm:playground -- \
  --model-dir "/path/to/qwen3-0.6b-q4f16/<revision>"
```

The default is a deterministic CPU baseline (`do_sample=false`,
`temperature=0`, 64 new tokens) because the Node Transformers.js runtime used
by this repository does not expose WebGPU here. There is no implicit backend
fallback. Pass `--device webgpu` explicitly when using a Node runtime that
does expose WebGPU; otherwise the command reports that it is unavailable.
The CPU baseline is only a standalone comparison and does not represent the
plugin's WebGPU-only path.

The output includes backend and duration followed by the raw generated text.
No title sanitization, ChatML/thinking removal, cache read/write, or cluster
result update is performed. `--max-new-tokens` is bounded to 256 and input is
bounded to 12,000 characters by default.
