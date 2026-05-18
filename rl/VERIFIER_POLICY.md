# RL Verifier Policy — Thesis-Relevant Documentation

> **⚠️ WICHTIG FÜR THESIS-SCHREIBEN UND -VERTEIDIGUNG ⚠️**
>
> Diese Datei dokumentiert die **einheitliche Verifier-Policy** die über alle
> RL-Datensätze in dieser Pipeline konsistent angewendet wurde. Beim Schreiben
> der Methodik-Sektion und bei einer möglichen Verteidigungs-Frage zum Thema
> "warum hast du X gemacht aber Y nicht?" liefert dieses Dokument die *eine*
> Regel und den Konsistenz-Beweis. Nicht löschen.

---

## Die Regel

> **Behalte nur Rows deren Antwort mit einem deterministischen, rule-based
> oder library-based Verifier gescored werden kann, der adversarial auf
> silent-pass=0 gehärtet werden kann.**

### Was die Regel ausschließt
- **LLM-as-Judge** (z.B. `llm_judge_ref`, `llm_judge_open`, `math_with_judge`)
- **Modell-basierte Verifier** (z.B. das 1.5B `TIGER-Lab/general-verifier`)
- **Open-form Antwort-Klassen** ohne deterministisches Vergleichsverfahren
  (z.B. freier Prosa-String, ungeordnete Listen, semantische LaTeX-Äquivalenz
  außerhalb canonical math)

### Was die Regel einschließt
- Symbolische / numerische Vergleiche (`math_equiv` mit normalized-string +
  optional numeric tolerance)
- Strict letter / canonical-token compares (`multi_gt`, `multiple_choice`)
- Normalized-string compares (`text_match`)
- Subprocess-isolierte Code-Execution (`code_asserts`, `code_stdio`)
- Pydantic-Schema-Validierung in Subprocess (`schema_pydantic`,
  `schema_structured_outputs`)
- Algorithmische Library-Scorer (`reasoning_gym` lib für `puzzle_match` und
  `reasoning_gym` task family)
- Regelbasierte IFEval-Constraints (`if_rules`)
- Regex/pattern match (`structured_match`)
- SWI-Prolog Subprocess-Execution für ILP-Rule-Induction
  (`prolog_rule_induction`) — deterministischer symbolischer Judge
- Vendored task-spezifische Python-Verifier aus MIT-licensed Repo
  (`synlogic_rule_based`, dispatch auf 35 Task-Families)

---

## Warum diese Regel

1. **Reproduzierbarkeit** — Modell-Judges sind stochastisch, ihre Bewertungen
   ändern sich mit Modellversion, Temperatur, Prompt-Phrasierung. Rule-based
   Verifier sind deterministisch. Reward-Signal ist bit-identisch über Runs.
2. **Hardenbarkeit** — Adversarial probes (silent-pass-on-garbage Test) sind
   nur sinnvoll definierbar wenn der Verifier deterministisch ist. Modell-
   Judge → "silent-pass" wird statistisches Konzept; bei rule-based ist es
   binäre Garantie.
3. **Infrastruktur** — Modell-Judges erfordern dedizierte GPU-Inferenz
   parallel zum Policy-Modell, was die Reward-Compute-Last verdoppelt.
   Außerhalb des Thesis-Scopes.
4. **Methodische Klarheit** — Reward-Signal-Pfad wird von Verifier-Bias
   sauber getrennt; alle Performance-Gains lassen sich am Verifier-Code
   selbst nachvollziehen, nicht an einem opaken Judge.

---

## Konsistenz-Beweis über alle 8 RL-Datensätze

### Tatsächliche verifier_type-Verteilung (kept rows)

| Dataset | Verifier types in kept set | Status |
|---|---|---|
| `am_thinking_v1_rl` | math_equiv, code_asserts, code_stdio | ✅ |
| `dolci_think_rl_7b` | if_rules, math_equiv, code_asserts, code_stdio | ✅ |
| `logi_glue` | multi_gt, text_match | ✅ |
| `nemotron_3_nano_rl_blend` | multiple_choice, code_stdio, if_rules, schema_structured_outputs | ✅ |
| `nemotron_rl_reasoning_gym_v1` | reasoning_gym | ✅ |
| `synthetic2_rl` | text_match, structured_match, math_equiv, puzzle_match, code_stdio, if_rules, code_asserts, schema_pydantic, multi_gt | ✅ |
| `webinstruct_verified` | math_equiv, multi_gt | ✅ |
| `slr_bench` (en+de+es+fr+it+pt+nl, 7 Sprachen) | prolog_rule_induction | ✅ |
| `synlogic` | synlogic_rule_based (dispatch auf 26 task families) | ✅ |

→ **Kein einziger** verifier_type ruft ein LLM/Modell-Judge auf.

### Konsistente Drop-Behandlung von Model-Judge-Rows

Drei Datensätze hatten upstream Model-Judge-Rows. Alle drei wurden
konsistent ausgeschlossen:

| Dataset | Drop-Reason | Anzahl |
|---|---|---|
| `dolci_think_rl_7b` | `ungradeable_verifier_llm_judge_ref` | 16.294 |
| `dolci_think_rl_7b` | `ungradeable_verifier_llm_judge_open` | 3.614 |
| `nemotron_3_nano_rl_blend` | `ungradeable_verifier` (math_with_judge + WorkBench) | ~siehe Report |
| `webinstruct_verified` | `requires_model_verifier` (Expression/String/List/Matrix/Other) | 85.331 |

Datensätze ohne Model-Judge upstream (`am_thinking`, `logi_glue`,
`synthetic2_rl`, `nemotron_rl_reasoning_gym`, `slr_bench`) brauchten keinen
Drop — sie bestanden bereits aus rule-based/library-based verifizierbaren
Daten. `slr_bench` ist besonders sauber: jede Row trägt ein executable
Validation-Program und wird von einem deterministischen SWI-Prolog Judge
gescored.

### Hardening-Konsistenz

Jeder NeMo Gym Resource Server in `Code_Templat/Gym/resources_servers/`
durchläuft denselben Hardening-Standard:

- **Unit tests:** pro Verifier-Klasse, inkl. Regression-Tests für gefundene Bugs
- **Real-row stress:** Sample aus echter `kept.parquet`, golden-path Rollouts
  + adversarial probes; Ziel ist `silent-pass=0` über alle probes

Aktuelle Hardening-Status:

| Dataset | Unit tests | Adversarial silent-pass | Golden-path |
|---|---|---|---|
| `am_thinking_v1_rl` | 28/28 | 0 | ✅ |
| `dolci_think_rl_7b` | ✅ | 0 | ✅ |
| `logi_glue` | 20/20 | 0/35.703 | 100% |
| `nemotron_3_nano_rl_blend` | ✅ | ✅ | ✅ |
| `nemotron_rl_reasoning_gym_v1` | ✅ | ✅ | ✅ |
| `synthetic2_rl` | 55/55 | 0 | ✅ |
| `webinstruct_verified` | 30/30 | 0/21.787 | 99.96% |
| `slr_bench` (7 Sprachen) | 21/21 | 0/1.260 | 210/210 (100%) |
| `synlogic` | 10/10 | 0/1.820 (adversarial only, see README) | n/a (format mismatch) |

---

## Verteidigungs-Argument (für Schreiben + Defense)

Wenn die Frage kommt: *"Warum hast du Modell-Judge-Rows ausgeschlossen, der
General-Reasoner / Dolci-Author hat das doch absichtlich so designed?"*

Antwort:

> "Der Kern meiner Arbeit ist eine deterministische, hardenbare Verifier-
> Architektur als Reward-Signal-Quelle für RL-Training. Modell-basierte
> Verification ist ein eigenständiges Thesis-Thema (Judge-Bias, Robustness,
> Stochastizität messen), das nicht in den Scope dieser Pipeline gehört.
> Ich integriere die *Daten* der genannten Arbeiten, nicht die Verifier-
> Innovation. Wo Upstream-Daten Modell-Judges voraussetzten, habe ich diese
> Rows konsistent über alle 7 Datensätze ausgeschlossen — mit derselben
> Regel: 19.908 Rows in Dolci, ein Subset in Nemotron-3-Nano, 85.331 Rows
> in WebInstruct-verified. Es gibt **eine** Policy und **einen**
> Konsistenz-Beweis — keinen Spagat zwischen 'streng bei Dataset A, locker
> bei Dataset B'."

---

## Wenn ich das später ändern wollte

Falls in einer Folgearbeit ein Modell-Judge integriert werden soll, sind
die ausgeschlossenen Rows nicht verloren — sie liegen in den `*.dropped.parquet`
Dateien mit ihrem ursprünglichen `_drop_reason`. Re-Inkludieren wäre ein
chirurgischer Eingriff:

1. Drop-Reason filtern (`ungradeable_verifier_llm_judge_*`, `requires_model_verifier`)
2. Verifier-Server um neuen task_type `model_judge` erweitern
3. vLLM-Sidecar mit Judge-Modell deployen
4. Verifier-Policy-Dokumentation in dieser Datei explizit erweitern

Damit bleibt die jetzige Pipeline integer und die Erweiterung wird
nachvollziehbar als bewusster Scope-Wechsel.

---

## Quellen / Pfade

- Filter-Scripts: `Master_Thesis/Data_pipline/rl/filter_and_format_*.py`
- Filtering-Reports: `Master_Thesis/corpora/rl/<dataset>/FILTERING_REPORT.md`
- NeMo Gym Servers: `Master_Thesis/Code_Templat/Gym/resources_servers/<dataset>/`
- Memory-Index: `~/.claude/projects/-home-workdir/memory/MEMORY.md`

**Erstellt:** 2026-05-16 — nachdem `webinstruct_verified` als 7. RL-Dataset
End-to-End fertiggestellt und der Policy-Check über die ganze Pipeline
durchgeführt wurde.
