# predm2-chat: LLM chat + longitudinal memory + risk model + JITAI (research prototype)

```
config.yaml                 <- EDIT target column, model names
data/feature_spec.csv       <- EDIT: one row per feature, chat_class = easy | noisy | clinic_only
data/{train,val,test}.csv   <- export from Colab (numeric columns + target), SAME 70/15/15 split
src/
  ensemble.py   6 base models + Mean/Stack All/Top3 (your diagram), selection on validation only
  step1_train.py leakage screen -> FULL and CHAT-FEASIBLE models -> calibration -> test once -> Table 1
  risk.py       frozen model scoring from chat features (+ "insufficient data" rule)
  llm.py        LLM wrapper with cache + retries (set ANTHROPIC_API_KEY)
  extraction.py chat -> structured features (null if not stated, pydantic validated)
  memory.py     SQLite hybrid memory; variants M0..M4 for ablation; deletion
  jitai.py      JITAI spec, interventions, 4 policies (none/random/fixed/tailored)
  safety.py     crisis / eating-disorder / emergency / diagnosis guardrails
  agent.py      safety + memory brief + JITAI instruction -> LLM reply
  simulate/     conversation_gen.py (row->chat), persona.py (multi-session), user_sim.py (JITAI response model)
experiments/    exp1 extraction | exp2 memory ablation | exp3 missingness | exp4 JITAI | exp5 safety
results/        tables written here as CSV
```

## Run order
```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python -m src.step1_train                 # Table 1  (no LLM)
python -m experiments.exp3_missingness    # no LLM
python -m experiments.exp4_jitai          # no LLM
python -m experiments.exp5_safety --agent
python -m experiments.exp1_extraction 60  # Table 2  (LLM, ~240 conversations)
python -m experiments.exp2_memory 20      # Table 3  (LLM, 20 personas x 6 sessions)
```

## Bring your Colab model
Export the three splits you already used and the column names:
`train.to_csv("train.csv", index=False)` (same for val/test), upload to `data/`.
Encode categoricals to numbers first. Edit `feature_spec.csv` so every feature column appears with its chat_class.
Feature names in the spec must match the CSV columns exactly.

## Honest-reporting notes
- Exp4 uses ASSUMED user-response parameters: report them and the sensitivity table. Do not claim efficacy.
- Exp5 patterns were tuned on the red-team list: write a SEPARATE held-out prompt set for the numbers you report.
- Simulator LLM (haiku) differs from system LLM (sonnet); ideally use a different provider family for the simulator.
- M1 (raw history) cannot honor deletion by design; that is part of the ablation.
- Session transcripts do not depend on assistant replies (simplification); state this as a limitation.
