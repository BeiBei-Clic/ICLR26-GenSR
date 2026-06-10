---
phase: 10
slug: dit-vae-cola-dlm-dit-vae
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-10
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none — existing infrastructure |
| **Quick run command** | `python -m pytest tests/test_lm_head_finetune.py -x -q` |
| **Full suite command** | `python -m pytest tests/test_lm_head_finetune.py -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_lm_head_finetune.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/test_lm_head_finetune.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | DEC-01 | unit | `python -m pytest tests/test_lm_head_finetune.py::test_only_lm_head_trainable -x -q` | ❌ W0 | ⬜ pending |
| 10-01-02 | 01 | 1 | DEC-02 | unit | `python -m pytest tests/test_lm_head_finetune.py::test_z_opt_forward_path -x -q` | ❌ W0 | ⬜ pending |
| 10-01-03 | 01 | 1 | DEC-03 | unit | `python -m pytest tests/test_lm_head_finetune.py::test_training_loop_runs -x -q` | ❌ W0 | ⬜ pending |
| 10-01-04 | 01 | 1 | end-to-end | smoke | `python dit_train/finetune_lm_head.py --num-iterations 5 --batch-size 2` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_lm_head_finetune.py` — stubs for all test cases above

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Loss curve shows decreasing trend | DEC-03 | Requires visual inspection of training output | Run 500 iterations, verify loss decreases |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
