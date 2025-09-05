# ➿ Loop Agents – Reflexivity Scenario Harness
Dual report generator for context and topic-driven LLM guided scenario narratives.

Small, modular Python tool that runs two LLM agents reasoning in opposite feedback directions (NEG vs POS), then produces three outcomes: negative, neutralized, positive. Includes knobs for horizon, weighting, dampening, compounding, and independent (non‑reflexive) comparators.

### Requirements
- Python 3.9+
- Recommended packages:
  - Runtime: `openai`, `python-dotenv`
  - Optional: `numpy` (bands/metrics), `jsonschema` (schema guardrail), `matplotlib` (plotting)

### Install
```bash
python -m pip install -U pip
python -m pip install openai python-dotenv numpy jsonschema matplotlib pytest
```

### Live mode configuration
Create a `.env` in the project root:
```env
OPENAI_API_KEY=your_real_key
OPENAI_MODEL=gpt-4o-mini
```

### Quick start
- Mock (deterministic):
```bash
python -m cli --topic "Enron-like patterns" --context "Company X 10-Q anomalies; auditor churn" --mock --seed 123 --json
```
- Live (reads from `.env`):
```bash
python -m cli --topic "Enron-like patterns" --context "Company X 10-Q anomalies; auditor churn" --export out/live.json
```

### CLI flags (key ones)
- `--topic`, `--context`: scenario inputs
- `--mock`: deterministic `MockClient`; omit for live
- `--json`: print JSON instead of table
- `--export <path>`: write full JSON
- `--csv <path>`: export paths (NEG/NEU/POS and comparators)
- `--seed <int>`: seed for mock reproducibility
- `--weights w_neg,w_pos`: override confidence weights (e.g., `0.6,0.4`)
- `--damp <float>`: neutral damp factor (default 0.55)
- `--eps <float>`: snap-to-zero threshold (default 0.049)
- `--neutral {concise|compare}`: neutral thesis style
- `--horizon <int>`: number of path points (default 5)
- `--unit {days|weeks}`: label for horizon steps
- `--path-mode {linear|geom}`: geometric compounding for path shape
- `--show-indep`: print independent comparator paths
- `--indep-mode {drift|decay}`: comparator mode (outward drift or inward decay)
- `--drift <float>`: outward step size for drift (default 0.2)
- `--kappa <float>`: decay rate for decay mode (default 0.15)
- `--scenarios <csv>`: batch run CSV with `topic,context` rows

### Examples
- Export CSV and JSON, geometric, longer horizon, with comparators:
```bash
python -m cli \
  --topic "Enron-like patterns" \
  --context "Company X 10-Q anomalies; auditor churn" \
  --horizon 12 --unit days --path-mode geom \
  --indep-mode drift --drift 0.25 --show-indep \
  --csv out/paths.csv --export out/live.json
```
- Batch mode from CSV:
```bash
python -m cli --scenarios data/scenarios.csv --mock --seed 7 --export out/last.json
```

### Example topics and contexts
- Topic: "Enron-like patterns"
  - Context: "Company X 10-Q anomalies; auditor churn; off-balance-sheet exposure (hypothetical)"
- Topic: "Bank run risk"
  - Context: "Regional lender deposit flight; rising wholesale funding costs; social-media rumor loop"
- Topic: "Supply chain recovery"
  - Context: "Lead times normalizing; input costs easing; inventory drawdowns pacing guidance"
- Topic: "Regulatory crackdown"
  - Context: "Sector Y faces proposed rules; fines chatter; compliance capex pressure vs clarity premium"
- Topic: "Meme-stock short squeeze"
  - Context: "High short interest; options gamma dynamics; retail inflows; borrow scarcity"
- Topic: "Property market slowdown (macro)"
  - Context: "Credit tightening; developer distress; policy support windows; refinancing cliff"
- Topic: "Crypto exchange stress (hypothetical)"
  - Context: "Stablecoin de-peg fears; liquidity thinning; on-chain outflows vs rescue rumors"
- Topic: "SaaS churn acceleration"
  - Context: "Seat contraction; usage downticks; price increase pushback; net retention at risk"
- Topic: "Airline fuel spike"
  - Context: "Jet fuel rally; hedging coverage; fare elasticity; capacity discipline"
- Topic: "Semiconductor cycle trough"
  - Context: "Inventory digestion; order cuts; AI-demand spillover; capex discipline"

### Plotting (optional)
1) Export CSV via CLI (see above)
2) Plot:
```bash
python plot_paths.py --csv out/paths.csv --json out/live.json --show-indep --save out/plot.png
```

### Tests
```bash
python -m pytest -q
```
Shape and overlap tests check:
- outcomes contain negative/neutralized/positive
- 5 (or `--horizon`) points per path
- confidence within [0,1]
- no drivers/risks overlap

### Notes and safeguards
- Neutral path uses confidence-weighted blend with horizon-aware dampening and optional geometric compounding; smoothing avoids plateaus.
- A variance band around the neutral path is derived from NEG/POS spread (if `numpy` available).
- Optional JSON schema validation adds `meta.validation_error` if malformed (if `jsonschema` installed).
- Live mode content should be treated as hypothetical; the system prompts avoid defamatory assertions.

### Troubleshooting
- Missing independent baselines in plots: re-export CSV with `--show-indep --csv out/paths.csv` and re-run `plot_paths.py --show-indep`.
- Live mode error: ensure `.env` is present and `OPENAI_API_KEY` is set; install `openai` and `python-dotenv`.
- Windows PowerShell: if `python` isn’t recognized, try `py` instead.


