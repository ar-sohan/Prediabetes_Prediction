# Free LLM presets: replace the `llm:` block in config.yaml with ONE of these

Free tiers change often. Check each provider's current limits and model list before you start.
All of them are rate-limited, so we use 1-2 workers and a pause between calls. The cache (logs/llm_cache) means
every finished call is free to re-run.

## A) Ollama: local, fully free, no account, no limits (needs a decent laptop; 8 GB RAM minimum)
1. Install Ollama from ollama.com, then in a terminal:  ollama pull llama3.2:3b     (small/fast)  or  ollama pull qwen2.5:7b  (better)
2. config.yaml:
```yaml
llm:
  provider: openai_compatible
  base_url: http://localhost:11434/v1
  api_key_env: ""
  max_workers: 1
  min_interval_sec: 0
  system_model: qwen2.5:7b
  sim_model: llama3.2:3b
  judge_model: qwen2.5:7b
```

## B) Google AI Studio (Gemini) free tier
1. Create a key at aistudio.google.com (no card). In Git Bash:  export GEMINI_API_KEY=your_key
2. config.yaml (put a current free Flash model name from AI Studio's model list):
```yaml
llm:
  provider: openai_compatible
  base_url: https://generativelanguage.googleapis.com/v1beta/openai
  api_key_env: GEMINI_API_KEY
  max_workers: 2
  min_interval_sec: 6
  system_model: <a Gemini Flash model name>
  sim_model: <a Gemini Flash-Lite or Flash model name>
  judge_model: <a Gemini Flash model name>
```

## C) Groq free tier (very fast, small daily caps)
```yaml
llm:
  provider: openai_compatible
  base_url: https://api.groq.com/openai/v1
  api_key_env: GROQ_API_KEY
  max_workers: 1
  min_interval_sec: 4
  system_model: <a Llama model name from Groq's list>
  sim_model: <a smaller Llama model name>
  judge_model: <same as system_model>
```

## D) OpenRouter free models (names end in :free)
```yaml
llm:
  provider: openai_compatible
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
  max_workers: 1
  min_interval_sec: 4
  system_model: <a model whose name ends in :free>
  sim_model: <a different :free model>
  judge_model: <same as system_model>
```

After editing config.yaml:  python check_llm.py
Use a DIFFERENT model for sim_model than system_model where you can (the simulator should not be the system under test).
