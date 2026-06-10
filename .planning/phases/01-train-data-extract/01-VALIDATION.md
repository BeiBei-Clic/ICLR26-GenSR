---
phase: 1
slug: train-data-extract
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-09
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none |
| **Quick run command** | `python -c "from dit_train.data.latent_dataset import FlowMatchingDataset; print('import ok')"` |
| **Full suite command** | `python -c "import torch; from dit_train.data.latent_dataset import FlowMatchingDataset; ds = FlowMatchingDataset('weights/checkpoint.pth'); prior, post = ds[0]; assert prior.shape == (512,); assert post.shape == (512,); print('PASS')"` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run import check
- **After every plan wave:** Run full instantiation test
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | DATA-01 | integration | `python -c "from dit_train.data.latent_dataset import FlowMatchingDataset; print('ok')"` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | DATA-02 | integration | `python -c "import torch; ds=FlowMatchingDataset('weights/checkpoint.pth'); prior,post=ds[0]; assert prior.shape==(512,) and post.shape==(512,); print('PASS')"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `dit_train/__init__.py` — package init
- [ ] `dit_train/data/__init__.py` — package init
- [ ] `dit_train/data/latent_dataset.py` — main dataset file

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| DataLoader batch yield correct shapes | DATA-02 | 需要 GPU + checkpoint | `dl = DataLoader(ds, batch_size=4); batch = next(iter(dl)); assert batch[0].shape == (4, 512)` |

---

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
