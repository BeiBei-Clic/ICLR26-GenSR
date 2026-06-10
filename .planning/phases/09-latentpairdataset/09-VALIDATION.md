---
phase: 9
slug: latentpairdataset
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-10
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none |
| **Quick run command** | `python3 -m pytest tests/test_latent_dataset.py -v -x` |
| **Full suite command** | `python3 -m pytest tests/test_latent_dataset.py tests/test_fm_train.py -v` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/test_latent_dataset.py -v -x`
- **After every plan wave:** Run `python3 -m pytest tests/test_latent_dataset.py tests/test_fm_train.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | Performance | integration | `python3 -m pytest tests/test_latent_dataset.py -v -x` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_latent_dataset.py` — existing, may need updates for multi-process tests

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU utilization improvement with batch_size=20 | Performance | Needs real GPU + training run | Run train_fm.py with --num-iterations 50 --device-batch-size 20, observe GPU utilization via nvidia-smi |
| DDP multi-card correctness | Correctness | Needs multiple GPUs | Run torchrun --nproc_per_node=4 train_fm.py, verify no errors |

---

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
