# Epistemic Agent Metrics — Definitions & Interpretation Guide

> These metrics come from **Active Inference**, a neuroscience-inspired framework where the agent treats every decision as a *belief update problem*. Instead of blindly executing actions, the agent asks: *"How uncertain am I? What action would reduce my uncertainty the most?"*

---

## Top-Level Metrics (Agent Summary)

### `final_confidence` *(0 → 1)*
**What it is:** The agent's overall confidence that it has enough information to answer.

**How to read it:**
| Range | Meaning |
|-------|---------|
| **0.0 – 0.3** | Very uncertain — agent is guessing |
| **0.3 – 0.6** | Moderate — agent has some evidence but gaps remain |
| **0.6 – 0.8** | High — agent is fairly sure of its answer |
| **0.8 – 1.0** | Very high — agent has strong evidence |

**Why it matters:** Unlike standard LLM agents that always respond as if they're certain, this agent *knows when it doesn't know*. A confidence of 0.38 on a dangerous command tells you the agent is appropriately uncertain.

---

### `final_entropy` *(H, in bits)*
**What it is:** Shannon entropy — measures how *spread out* the agent's beliefs are across possible states.

**How to read it:**
- **High H (>3.0):** Beliefs are spread evenly → agent is confused / uncertain
- **Low H (<2.0):** Beliefs are concentrated → agent has a clear picture
- **H = 0:** Perfect certainty (one state has 100% probability)

**Why it matters:** Entropy tells you the *quality* of the agent's understanding. If H is still high after several loops, it means the agent hasn't been able to narrow things down — the problem is genuinely ambiguous.

**Example:** `"List Python files"` → H dropped from **3.49 → 2.74** after finding 33 files. The evidence reduced uncertainty.

---

### `final_vfe` *(Variational Free Energy)*
**What it is:** A single number that measures the *gap between what the agent believes and what it has observed*. Borrowed from neuroscience — the brain tries to minimize this.

**How to read it:**
- **More negative = better fit** — the agent's internal model matches reality well
- **Close to 0 or positive = poor fit** — observations don't match expectations

**Why it matters:** VFE is the agent's internal "surprise meter". If VFE isn't decreasing over loops, the agent's model of the world isn't improving.

---

### `converged` *(true/false)*
**What it is:** Whether the agent stopped because it reached a stable conclusion (true) vs. hit the max iteration limit (false).

**Why it matters:** `converged: true` means the agent made a deliberate decision to stop. `converged: false` means it ran out of time — the answer may be incomplete.

---

### `convergence_reason`
**What it is:** Explains *why* the agent stopped.

| Reason | Meaning |
|--------|---------|
| `confidence_stable_at_XX%` | Agent's confidence was high for 2+ consecutive loops |
| `answer_synthesized_from_evidence` | Agent chose `answer_user` and had evidence to synthesize |
| `pragmatic_action_succeeded` | A file/system action completed successfully |
| `blocked_by_policy:PRINCIPLE` | Security Constitution blocked a dangerous action |
| `needs_clarification_from_policy` | Policy requires user confirmation before proceeding |
| `aborted:reason` | Agent determined it cannot fulfill the request |
| `max_iterations_reached` | Ran out of loops before converging |

---

### `final_beliefs`
**What it is:** The agent's final Bayesian probability distributions across three belief factors:

#### [file_status](file:///c:/Users/hp/Downloads/Agentic_Testaing/src/epistemic_agent/generative_model.py#90-95) — *"Does the target file/resource exist?"*
| State | Meaning |
|-------|---------|
| `exists` | Agent believes the file/resource is real and accessible |
| `does_not_exist` | Agent believes it doesn't exist |
| `ambiguous` | Multiple matches found, unclear which one |
| `unknown` | Haven't checked yet |

#### [user_intent](file:///c:/Users/hp/Downloads/Agentic_Testaing/src/epistemic_agent/generative_model.py#100-104) — *"What does the user actually want?"*
| State | Meaning |
|-------|---------|
| [read](file:///c:/Users/hp/Downloads/Agentic_Testaing/src/epistemic_agent/mcp_integration.py#220-258) | User wants to view/retrieve information |
| [delete](file:///c:/Users/hp/Downloads/Agentic_Testaing/src/epistemic_agent/desktop_tools/file_ops.py#193-234) | User wants to remove something |
| `clarify` | User needs clarification or help |
| `unknown` | Intent is not yet clear |

#### [risk_level](file:///c:/Users/hp/Downloads/Agentic_Testaing/src/epistemic_agent/generative_model.py#109-113) — *"How dangerous is this request?"*
| State | Meaning |
|-------|---------|
| [safe](file:///c:/Users/hp/Downloads/Agentic_Testaing/src/epistemic_agent/desktop_tools/file_ops.py#80-94) | No risk — routine operation |
| `moderate` | Some caution needed |
| `hazardous` | Potentially destructive — requires strict safety checks |

**Why it matters:** These beliefs drive every decision. For `rm -rf /`, risk_level hit `hazardous: 61%` — that's what triggered the security block. For `List Python files`, file_status reached `exists: 86%` and risk was `safe: 52%`.

---

## Per-Loop Metrics (`loop_trace`)

Each entry in `loop_trace` shows what happened in one Active Inference cycle:

### `efe_score` *(Expected Free Energy)*
**What it is:** The score used to *select which action to take*. It combines two components:
- **info_gain** — how much this action would *reduce uncertainty*
- **pragmatic_value** — how much this action would *achieve the goal*

**How to read it:**
- **Higher EFE = preferred action** (it either learns more or achieves more)
- **Negative EFE = penalized** (action has been repeated or is low-value)
- **EFE = -5.0** = hard blocked (preconditions not met)

**Example:** Loop 1 ranked `list_files (EFE=1.535)` over `answer_user (EFE=-5.0)` because listing files would gain information, while answering without evidence was blocked.

---

### `info_gain` *(Epistemic Value)*
**What it is:** How much *new information* this action is expected to provide, measured as expected entropy reduction.

- **High info_gain (>1.0):** Action will significantly reduce uncertainty
- **Low info_gain (~0.0):** Action won't teach the agent anything new

**Why it matters:** This is the "curiosity" drive. The agent prefers actions that reduce ignorance before taking irreversible actions.

---

### `pragmatic_value` *(Extrinsic Value)*
**What it is:** How much *goal progress* this action provides.

- **Positive:** Action moves toward the user's goal
- **Zero:** Pure information-seeking action
- **Negative:** Action is counter-productive or penalized

---

### `surprisal` *(Prediction Error)*
**What it is:** How different the *actual observation* was from what the agent *expected*. Computed using semantic similarity.

**How to read it:**
- **Low (<1.0):** Observation matched expectations → agent's model is accurate
- **Medium (1.0–2.0):** Some surprise but within normal range
- **High (>2.0):** Major divergence → potential **hallucination detected**

**Why it matters:** This is the agent's built-in hallucination detector. If the LLM generates something wildly different from what was expected, `is_hallucination` flips to `true`.

---

### [concentration](file:///c:/Users/hp/Downloads/Agentic_Testaing/src/epistemic_agent/generative_model.py#148-165) *(Dirichlet Σα)*
**What it is:** Evidence strength — how much data has accumulated behind each belief factor. Higher = more evidence.

**How to read it:**
- **Low (~3–4):** Initial state, barely any evidence
- **Medium (5–8):** Several observations incorporated
- **High (>10):** Strong evidence base, beliefs are stable

**Example:** After Loop 1 of `List Python files`, concentration jumped to `file: 5.0, intent: 8.7, risk: 7.1` — the agent quickly gathered strong evidence about intent and risk.

---

## How It All Fits Together

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  LOOP: Repeat until converged or max iterations │
│                                                 │
│  1. Check CONFIDENCE (am I sure enough?)        │
│  2. Calculate EFE for each possible action      │
│     → info_gain + pragmatic_value = EFE score   │
│  3. Pick action with highest EFE                │
│  4. Security audit (policy check)               │
│  5. Execute action → get observation            │
│  6. Check SURPRISAL (hallucination detection)   │
│  7. Update BELIEFS (Bayesian update)            │
│  8. Update ENTROPY and VFE                      │
│                                                 │
│  Each loop → one entry in loop_trace            │
└─────────────────────────────────────────────────┘
    │
    ▼
Final Answer + Full Metrics
```

---

## Quick Reference Card

| Metric | Good Value | Bad Value | What It Tells You |
|--------|-----------|-----------|-------------------|
| [confidence](file:///c:/Users/hp/Downloads/Agentic_Testaing/src/epistemic_agent/test_result.py#37-50) | > 0.6 | < 0.3 | Agent certainty |
| [entropy](file:///c:/Users/hp/Downloads/Agentic_Testaing/src/epistemic_agent/free_energy.py#38-45) | < 2.0 | > 3.0 | Belief uncertainty |
| `vfe` | Very negative | Near 0 | Model-reality fit |
| `efe_score` | Positive | -5.0 | Action quality |
| `info_gain` | > 1.0 | 0.0 | Curiosity/learning |
| `surprisal` | < 1.0 | > 2.0 | Hallucination risk |
| [concentration](file:///c:/Users/hp/Downloads/Agentic_Testaing/src/epistemic_agent/generative_model.py#148-165) | > 8.0 | < 4.0 | Evidence strength |
| `converged` | `true` | `false` | Clean termination |
