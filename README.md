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
