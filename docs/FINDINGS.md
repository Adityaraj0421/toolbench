# toolbench findings

Results from running the experiments in `experiments/` against OpenRouter
(`openai/gpt-4o-mini`, `google/gemini-2.5-flash`). All numbers were verified by
recomputing ground truth dynamically from each task prompt, and the v1 set was
independently cross-checked by two auditor agents that matched exactly.

> Caveats up front: small samples (n=5 to n=10), tools are mocked (no real
> latency/failure), only two models, and results come through OpenRouter's
> OpenAI-compatibility layer. Treat as directional, not statistically rigorous.

## Headline

**Whether a model calls a tool is mostly a prompt-design choice, not a function
of the task.** The system prompt and the tool's own description move call rates
far more than task difficulty does. The same wording lever produces both failure
modes:

| Lever | Effect | Harm |
|---|---|---|
| Description: *"don't use for trivial math"* | call rate on easy math 100% → 0% | none (they can do easy math) |
| System prompt: *"answer correctly"* | call rate ~100% across ALL difficulties | wasteful over-calling |
| Description: *"prefer your own knowledge"* | Gemini call rate on HARD math 28/30 → 6/30 | severe: every skip was wrong |

## Experiment 1 — over-calling (`over-calling.yaml`)

Task `7 + 8` (trivial), calculator offered, n=5.

| Model | neutral desc | "don't use for trivial" desc |
|---|---|---|
| gpt-4o-mini | 100% call (5/5) | 0% (0/5) |
| gemini-2.5-flash | 40% call (2/5) | 0% (0/5) |

One sentence in the description eliminated over-calling. Over-calling also
roughly doubled token cost (214 vs 104 avg tokens for gpt-4o-mini).

## Experiment 2 — difficulty sweep (`difficulty-sweep.yaml`)

Arithmetic ladder `7+8` → `4821*7639`, neutral description, system prompt
"answer correctly", n=5. Both models called the calculator ~49/50 across every
difficulty (gemini skipped once on `23*47`, still correct). **No "trust
threshold" appeared** — "answer correctly" saturates tool use regardless of how
easy the math is. Accuracy was 100% everywhere (ceiling effect, no signal).

## Experiment 3 — decoy tool (`decoy-tool.yaml`)

calculator + weather both offered; one correct tool per task; n=5. **Zero
wrong-tool calls in either direction** — well-named tools and clear tasks gave
perfect routing. One Gemini run emitted Python-style `default_api.weather(...)`
pseudo-code in its message content instead of a structured tool call, which the
OpenAI-format harness scored as no-call. A real cross-provider interop gotcha.

## Experiment 4 — under-calling (`under-calling.yaml` + `no-tools-baseline.yaml`)

Hard, non-memorized arithmetic (e.g. `73948 * 6271`), controlled system prompt,
n=10.

**No-tools baseline (unaided accuracy):** both models scored **0/30** on every
task. The calculator is genuinely necessary here, so skipping it = wrong answer.

**Call rate and accuracy with the calculator present:**

| Model | variant | call rate | accuracy |
|---|---|---|---|
| gpt-4o-mini | neutral | 30/30 | 30/30 |
| gpt-4o-mini | discouraged | 30/30 | 30/30 |
| gemini-2.5-flash | neutral | 28/30 | 28/30 |
| gemini-2.5-flash | discouraged | **6/30** | **6/30** |

**The cross-tab — when the tool was skipped, was the answer right?**

| Model | variant | skipped | skipped & wrong | called | called & wrong |
|---|---|---|---|---|---|
| gemini-2.5-flash | discouraged | 24 | **24** | 6 | 0 |
| gemini-2.5-flash | neutral | 2 | 2 | 28 | 0 |
| gpt-4o-mini | both | 0 | 0 | 30 | 0 |

**Every skip on hard math produced a wrong answer (50/50 across all cells).** A
discouraging tool description drove Gemini to skip the calculator on math it
cannot do, collapsing accuracy from 28/30 to 6/30. GPT-4o-mini ignored the
discouragement when the math was clearly too hard and stayed at 30/30 — better
tool-use judgment. When Gemini skipped, it produced plausible-but-wrong products
(e.g. answered `463683628` for `73948*6271`; truth `463727908`).

## Experiment 5 — does under-calling generalize? (6 models)

Re-ran the under-calling design across six models from five vendors
(`under-calling.yaml` + `under-calling-xmodels.yaml`, n=10; no-tools baselines
confirmed **0% unaided** for all six). The discouraging-description effect is
**strongly model-specific** — same sentence, opposite outcomes:

| Model | neutral call / acc | discouraged call / acc | effect |
|---|---|---|---|
| gpt-4o-mini | 30/30 / 30/30 | 30/30 / 30/30 | **robust** — ignores it on hard math |
| deepseek-chat-v3 | 30/30 / 30/30 | 30/30 / 30/30 | **robust** |
| claude-3.5-haiku | 28/30 / 28/30 | 30/30 / 30/30 | **over-corrects correctly** — calls *more* |
| llama-3.3-70b | 30/30 / 28/30 | 21/30 / 21/30 | partial under-calling |
| gemini-2.5-flash | 28/30 / 28/30 | 6/30 / 6/30 | **collapse** (−73 pts) |
| mistral-small-3.2 | 18/30 / 18/30 | 6/30 / 6/30 | collapse + weak baseline |

Three behavioral clusters:
1. **Description-proof** (gpt-4o-mini, deepseek-chat-v3): recognize the math is
   too hard and call the tool regardless of the discouragement. Best judgment.
2. **Obedient-to-failure** (gemini, mistral, partly llama): take the description
   literally, skip the tool on math they cannot do, accuracy collapses.
3. **Over-corrects correctly** (claude-3.5-haiku): reads "use only if you can't
   do it yourself," concludes it can't, and calls the tool *even more*.

The same discouraging sentence ranged from harmless to catastrophic to
*beneficial* depending on the model. There is no model-independent "good" tool
description. (Method note: two minor accuracy miscounts from a naive answer
parser — scientific notation `1.9...e+07` and trailing text — were caught by
trace spot-check and fixed; call rates come from recorded tool calls and were
unaffected. Llama also occasionally echoed the raw tool-call JSON as its final
answer instead of stating the number, a real formatting quirk.)

## Takeaways for tool design

1. **Tune the description for the difficulty range you expect.** "Don't use for
   trivial things" is safe only if the model can actually do those things. The
   same phrasing applied to hard tasks causes confident wrong answers.
2. **Models differ in how literally they obey tool descriptions.** gpt-4o-mini
   overrode a discouraging description on hard math; gemini-2.5-flash obeyed it
   into failure. Test your wording on each model you ship.
3. **Over-calling has a cost signature** (≈2× tokens), so token usage doubles as
   an over-calling detector even before you inspect call counts.
4. **Watch the wire format across providers** — Gemini occasionally emits
   code-style tool calls an OpenAI-shaped client silently drops.

## Method note

An earlier analysis pass nearly reported a false "models get hard math wrong
despite the tool" finding, caused by a wrong hardcoded ground-truth constant in
the analysis script. Recomputing ground truth dynamically from the task prompts,
plus two independent auditor agents, caught it. Verify your analysis, not just
your code.
