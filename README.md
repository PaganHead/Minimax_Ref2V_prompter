# Kira Prompt Builder — ComfyUI

A single custom node that takes up to four optional reference images and creates a MiniMax H3 prompt using Qwen-VL or another OpenAI-compatible vision model.

## Install

Run on your RunPod terminal:

```bash
cd /workspace/runpod-slim/ComfyUI/custom_nodes
git clone https://github.com/PaganHead/Minimax_Ref2V_prompter.git KiraPromptBuilder
```

Then **restart ComfyUI** and add the node **Kira → Kira Prompt Builder**.

## Configure

The node does not automatically download or run a Qwen model. You must provide a working, vision-capable, OpenAI-compatible endpoint:

- **api_base**: URL ending in `/v1` (e.g. your vLLM server)
- **model_name**: the exact model ID served by the endpoint
- **KIRA_VLM_API_KEY**: optional environment variable for authenticated APIs (do not put keys in workflow JSON)

Image roles: 1 = location, 2 = character, 3 = outfit, 4 = prop/style. All optional.

## Lock prompt

1. Run once with **lock_prompt = false**; the output is saved under `.prompt_cache/`.
2. Switch to **true** to reuse the saved prompt without calling Qwen.
3. Change the **MiniMax video seed** to vary results while keeping text unchanged.

Cache survives ComfyUI restarts if the custom_nodes directory is persistent; it is node-ID-specific and not stored in workflow JSON. A duplicate node has a different cache. While unlocked, each queued execution invokes the LLM. Some LLM servers ignore `llm_seed`.

This is a V1 implementation, not yet tested against your actual Qwen or MiniMax deployment.
