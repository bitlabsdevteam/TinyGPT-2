# Glossary

Domain language for the GPT-2-from-scratch project. Terms only; no implementation detail.

**Pretraining run**
A training run that starts from random weights and learns next-token prediction on TinyStories. _Avoid_: "training" unqualified.

**Fine-tuning run**
A training run that starts from a Pretraining run's checkpoint and adapts it to TinyStoriesInstruct using QLoRA: the base weights stay frozen and quantized, and only the Adapter is trained. _Avoid_: "instruction tuning" for the run itself; that names the task.

**Adapter**
The small set of trainable low-rank weights added alongside the frozen base weights during a Fine-tuning run.

**Base model**
The frozen pretrained weights a Fine-tuning run builds on.

**Instruction story**
One TinyStoriesInstruct example: a prompt part (the story's requested words, features, summary or opening sentence) and a response part (the story).

**Response-only loss**
During a Fine-tuning run, loss is counted on the response part of an Instruction story, never on the prompt part.

**Story**
One TinyStories document. Stories are separated in the token stream by `<|endoftext|>`.

**Checkpoint**
A saved snapshot sufficient to resume a run exactly: model, optimizer, scaler, step and random-number state. Stored on Google Drive.

**Section**
One architectural layer of the model, owning its own file (embeddings, attention, MLP, layer norm, block, output head). _Avoid_: "layer" for this, because "layer" also means a repeated Block.

**Block**
One repeated transformer unit: attention then MLP, each with pre-norm and a residual connection. A model has `n_layer` Blocks.

**Production standard**
The engineering bar all runs are held to: reproducible, resumable, logged, profiled. Working interpretation of "Anthropic production standard"; no external document has been identified. See `CLAUDE.md` Open questions.

**Profile report**
The per-run deliverable: loss curves, perplexity, throughput, step time, peak GPU memory, MFU, a profiler trace, and sample generations.
