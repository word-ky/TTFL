# TTFL Research Coordination Protocol

Effective 2026-09-13.

## Role split

- ChatGPT: research lead. Read the newest Codex evidence first, distinguish implementation issues from mechanism failures, decide the next scientific question, and issue one bounded work package.
- Codex: engineering / experiment executor. Implement and run the active package, preserve raw evidence and negative results, then report through `coordination/CODEX_TO_CHATGPT.md`.

## Cadence

ChatGPT's automated GitHub review runs at approximately one-hour cadence. Codex may use a faster heartbeat to notice repository instructions, but this does not change the research cadence.

After each meaningful Codex result, ChatGPT should:

1. read the new commit, `CODEX_TO_CHATGPT.md`, relevant code, verification files, and raw/summary results;
2. critically analyze the evidence rather than reacting to headline accuracy;
3. identify the single most important unresolved mechanism question;
4. write **one next work package sized for roughly one hour of Codex engineering/experimentation**;
5. include explicit controls, pass/fail criteria, and stop conditions;
6. avoid stacking multiple research stages into one package.

Do not issue micro-tasks every few minutes unless Codex reports a concrete blocker that prevents completion of the active package.

## Current rule

T002 is already active. Let Codex finish T002 before changing direction unless a concrete implementation blocker is reported.

After T002, the next task must be chosen from the evidence. Do not automatically proceed to SSL, meta-learning, federation, a richer operator, or a new benchmark merely because T002 finished.

## Research invariant

Preserve the V2 sequence:

`prove context can be read under a clean supervised upper bound -> determine the right neutral fast operator -> only then design unlabeled / self-supervised writing -> only then integrate the final mechanism into PFL training/evaluation.`

Every apparent gain must survive matched-checkpoint and context-specificity controls before it is treated as mechanism evidence.
