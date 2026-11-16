# Antigravity ACP trial results

The 10-iteration function-minimization trial completed on 2026-10-05 using
Google's official ACP server 1.3.0 and `gemini-3.8-flash-medium`, authenticated
through the dedicated personal OAuth profile.

| Check | Observed result |
| --- | --- |
| Completed iterations | 10 |
| Generated and evaluated candidates | 10 valid; 0 with evaluator errors |
| ACP retries or failures | 0 |
| Initial combined score | 1.2147685971 |
| Best combined score | 1.4995496974 |
| Best reliability score | 1.0 |
| Checkpoints | Iterations 5 and 10 |
| Final database | Initial program plus 10 generated candidates |
| Elapsed time | Approximately 5 minutes 32 seconds |

The trial used the supplied [configuration](config.yaml): one evaluation worker,
three islands, a 300-second request timeout, two retries, a five-second retry
delay, and no LLM evaluator feedback. All ten candidate evaluations reported
reliability 1.0. There were no warning/error entries in the run log.

The separate opt-in live provider test also passed. Its direct tool challenge
returned `TOOLS_DISABLED`, reported no tool activity, and created no workspace
files; its evolution iteration produced an applied and evaluated candidate.

Final offline verification passed 40 ACP provider cases and the repository's
677 fast tests, including 83 subtests. The opt-in ACP live test passed; live
tests remain skipped by default.

Local run artifacts are under `openevolve_output_antigravity_acp/`, including:

- `best/best_program.py`
- `best/best_program_info.json`
- `checkpoints/checkpoint_5/`
- `checkpoints/checkpoint_10/`
- `logs/openevolve_20261005_002056.log`

These output files are ignored by Git. The quickstart configuration
and this summary preserve the setup and observed results. Model sampling and
evaluator randomness mean another run can produce different code and scores.
ACP reports context usage; per-call input/output token counts and monetary cost
were not inferred.
