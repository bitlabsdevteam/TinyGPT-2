# 07: Pretraining run (small config)

**What to build:** The Pretraining run entry point: AdamW with decay on 2-D params only, warmup plus cosine schedule, gradient clipping, gradient accumulation, bf16 autocast with fp16+GradScaler fallback, periodic validation, seeding, full config logged, and a token cap. A short run trains end to end on the small config.

**Blocked by:** 03, 06

**Status:** resolved

- [x] Loss decreases over a short run
- [x] Weight decay applies only to 2-D parameters
- [x] Effective batch size equals micro-batch x accumulation
- [x] Same seed gives same losses
- [x] Token cap stops the run
- [x] Config is logged with the run

Spec: spec.md
