---
phase: 4
slug: flow-matching
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-09
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none — existing project uses pytest directly |
| **Quick run command** | `python -m pytest tests/test_fm_train.py -v -x` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_fm_train.py -v -x`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | FM-01 | unit | `python -m pytest tests/test_fm_train.py::test_ot_path_interpolation -v` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | FM-02 | unit | `python -m pytest tests/test_fm_train.py::test_fm_loss_computation -v` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 1 | FM-01,FM-02 | integration | `python -m pytest tests/test_fm_train.py::test_training_step_runs -v` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 1 | FM-03 | unit | `python -m pytest tests/test_fm_train.py::test_only_dit_params_update -v` | ❌ W0 | ⬜ pending |
| 04-01-05 | 01 | 1 | FM-04,FM-05 | integration | `python -m pytest tests/test_fm_train.py::test_checkpoint_save_load -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_fm_train.py` — test stubs for FM-01 through FM-05
- [ ] Existing infrastructure covers all phase requirements (pytest already installed)

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Loss 曲线持续下降 | FM-04 | 需要运行完整训练观察趋势 | 运行 `python -m dit_train.train_dit --num-iterations=100` 观察 loss 输出 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
