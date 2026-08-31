# Sign Language Translator

Real-time sign language recognition from a webcam. MediaPipe landmarks feed a
dual-pathway recogniser: word-level signs from a temporal model, and
fingerspelled letters from a static handshape model, assembled into text.

Runs on CPU on an ordinary laptop. No GPU, and TensorFlow is optional.

```bash
python run.py doctor
```

---

> ### Recording data? Start here
>
> If someone asked you to record signs on your laptop, you want
> **[docs/RECORDING.md](docs/RECORDING.md)** — a step-by-step guide that
> assumes no Python or ML knowledge. It takes about 25–30 minutes.
>
> The short version:
>
> ```bash
> python run.py collect --signer S02 --limit 15 --repetitions 25
> ```
>
> Use a signer ID **nobody else is using**, and read the guide's Step 4 first —
> everyone has to sign each word the same way or the data is worthless.

---

## What is here

```
run.py                    single entry point for everything
docs/RECORDING.md         contributor guide - send this to whoever records
configs/vocabulary.txt    the signs to record (~100, cut it wherever you like)
slr/
  schema.py               MediaPipe landmark layout, feature groups
  manifest.py             dataset provenance (signer / session / order)
  features.py             normalisation, pooling, augmentation
  splits.py               evaluation protocols
  models/                 classical + neural model registry
  train.py                word model -> deployable bundle
  train_static.py         fingerspelling model -> deployable bundle
  bundle.py               model + preprocessing config + calibrated thresholds
  data/collect_dynamic.py record word signs
  data/collect_static.py  record fingerspelling
  runtime/
    predictor.py          sliding window, smoothing, segmentation, rejection
    fingerspell.py        static pathway
    sentence.py           gloss stream -> readable text
    app.py                the live application
    replay.py             test the live path without a webcam
  pipeline/gate.py        motion-energy routing between the two pathways

gesture/                  original recordings and first-pass scripts
self_data_model/          original fingerspelling scripts and models
```

`gesture/` and `self_data_model/` are the original project. They are kept
because `gesture/MP_Data` holds the source recordings that `run.py migrate`
rebuilds the dataset from. The scripts there are superseded by the `slr`
package — `gesture/train_model.py` in particular has a `LSTM(activation='relu')`
that diverges during training (its own logs show loss spiking to 23.6) and a
shuffled train/test split that reports inflated accuracy.

---

## Setup

Python 3.10–3.12. **Not 3.13** — MediaPipe has no wheels for it yet.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\pip install -r requirements.txt
```

```bash
.venv\Scripts\python run.py migrate
```

```bash
.venv\Scripts\python run.py doctor
```

On macOS / Linux use `.venv/bin/pip` and `.venv/bin/python`.

`migrate` rebuilds the manifest-backed dataset in `data/` from the original
`gesture/MP_Data` recordings. That dataset is derived, so it is not committed
— regenerating it takes a couple of seconds and avoids carrying 24 MB of
duplicate arrays in the repository.

The version pins in `requirements.txt` are load-bearing. MediaPipe's legacy
`mp.solutions.holistic` API — which produces the 1662-dimensional landmark
vector everything here is built on — exists only in the 0.10.x line;
MediaPipe 1.x removed it. That in turn pins `protobuf<5` and `numpy<2`.

**TensorFlow is optional and is not installed by default.** It cannot coexist
with the pinned stack (current releases need `protobuf>=6` and `numpy>=2`).
It is not needed: see [Which model](#which-model) below.

---

## Recording data

This is the part that determines whether the system works, and it is the part
no amount of code can substitute for. **The signs have to come from real
people in front of a real camera.**

**Contributors should follow [docs/RECORDING.md](docs/RECORDING.md)** — send
them that link, not this section.

```bash
# first signer, first 15 signs, 25 repetitions each  (~25 minutes)
python run.py collect --signer S01 --limit 15 --repetitions 25

# a second signer - the single highest-value thing you can record
python run.py collect --signer S02 --limit 15 --repetitions 25

# fingerspelling, optional
python run.py collect-letters --signer S01
```

Each contributor sends back their `data/raw/` folder. Unzip them all into
`data/raw/` so you have one directory per signer, then `run.py merge`.

Two rules that decide whether the effort is wasted:

1. **Every contributor uses a different `--signer` ID.** Duplicate IDs
   overwrite each other, and signer identity is what makes
   leave-one-signer-out evaluation possible.
2. **Everyone signs each word the same way.** Agree on one reference
   dictionary before anyone records. Two people signing "please" differently
   under the same label teaches the model nothing.

The collector records provenance (who signed, which session, in what order),
interleaves the class order so repetitions of one sign are spread across the
session, rejects and re-records sequences where MediaPipe lost the hands, and
checkpoints after every sequence so you can stop and resume with the same
command.

It also records an `__idle__` class by default — footage of you *not* signing.
This matters more than it sounds: without a negative class the model must
assign one of its known labels to every window, so it fires words at you
reaching for the mouse.

### How much to record

| Signers | Signs | Reps | Time per signer | Result |
|---|---|---|---|---|
| 1 | 15 | 25 | ~25 min | works for you, in your room |
| 2–3 | 25 | 30 | ~40 min | works for your team |
| 5+ | 50 | 30 | ~75 min | works for people it has never seen |

A model trained on one signer learns that signer. It will look excellent in
your own testing and disappoint the first time someone else tries it. Adding
a second signer does more for real accuracy than any model change in this
repository.

---

## Training

```bash
python run.py merge                # combine per-signer recordings
python run.py train --compare      # try several models, keep the best
```

`--compare` trains each candidate and ranks them by **live-path** behaviour,
not offline accuracy. This is not a stylistic preference. On the bundled
4-sign dataset:

| model | offline accuracy | live accuracy |
|---|---|---|
| random_forest | 99.2% | **93.3%** |
| extra_trees | 99.2% | 91.7% |
| mlp | **100.0%** | 39.2% |
| svm_rbf | 98.3% | 0.0% |

The model with the best offline score is nearly the worst in front of a
camera, and the SVM never emits a word at all. Offline accuracy is measured
on windows that start and stop exactly when the sign does; live, the model
sees a sliding window over a continuous stream and something has to decide
*when* a sign happened. Selecting on offline accuracy ships a broken product.

Training also calibrates the live confidence and margin thresholds on
held-out data, and prints the most-confused sign pairs so you know what to
re-record.

---

## Running

```bash
python run.py live
python run.py live --speech          # speak recognised words
python run.py live --camera 1        # a different camera
```

| Key | Action |
|---|---|
| `q` / `Esc` | quit |
| `space` | clear the sentence |
| `u` | undo the last word |
| `s` | toggle speech |
| `d` | toggle debug overlay |
| `f` | toggle fingerspelling |

The panel shows the current prediction with its confidence against the
calibrated threshold, the assembled text, and the raw gloss stream. When
nothing is recognised it says *why* — low confidence, ambiguous, unstable,
no hands — rather than silently showing the last thing it saw.

---

## Testing without a webcam

```bash
python run.py replay --limit 40
```

Replays recorded sequences as a continuous stream with rest gaps and reports
what the predictor actually commits. This catches the failures that only
appear live and are invisible to offline metrics:

- a sign that never commits because no rest edge is detected
- one sign committing three times
- words firing during pauses
- thresholds so tight nothing is ever emitted

Run it after every training run. Current state on the bundled data: 98%
correct first commit, no duplicates, no spurious fires.

---

## How the live path works

```
webcam frame
   -> MediaPipe Holistic          33 pose + 468 face + 21+21 hand landmarks
   -> normalise                   centre on mid-shoulder, scale by shoulder width
   -> feature group               pose + hands (258 dims; the face mesh is dropped)
   -> 30-frame sliding window
   -> motion-energy gate          idle / fingerspelling / signing
   -> classifier                  every 3rd frame
   -> smoothing + rejection       vote over recent windows, confidence + margin
   -> commit on the falling edge  of motion, then consume the window
   -> sentence assembly
```

Three details that make the difference between a demo and something usable:

**The face mesh is dropped.** It is 1404 of 1662 dimensions — 84.5% of the
raw vector — and manual signs are not distinguished by facial landmarks.
Keeping it costs accuracy and adds a channel that encodes who is signing
rather than what they signed.

**Coordinates are body-centred.** MediaPipe returns coordinates relative to
the image frame, so raw landmarks encode where you stood and how far you were
from the camera at least as strongly as they encode the sign.

**The window is consumed on commit.** After a word is emitted the sliding
window still contains that sign for another 30 frames, and anything predicted
from it is the same sign again. Clearing the buffer is what stops one gesture
from producing five words.

---

## Which model

The default is a random forest over **temporal pyramid pooling** — pooled
statistics computed over the whole window, then halves, then quarters. This
keeps coarse temporal ordering, which plain pooling throws away (a sign and
the same sign reversed pool identically), while producing a fixed-length
vector any classifier can consume.

That is why TensorFlow is optional. In this project's own live-path benchmark
the random forest beat every neural candidate once segmentation was taken
into account, at 2,100 parameters against 116,228 for the 1D-CNN.

The neural models (GRU, LSTM, BiLSTM, 1D-CNN, Transformer) are implemented
and registered. To use them, train in a separate environment that has
TensorFlow and `numpy>=2` but no MediaPipe, then copy the bundle across —
the bundle format is self-contained.

---

## Known limitations

- **The bundled model is trained on one signer and four signs.** It is a
  scaffold to prove the pipeline, not a usable vocabulary. Record your own.
- **The legacy fingerspelling data is 63-dimensional**, from an older feature
  extractor; the current one produces 69 dimensions. The runtime truncates to
  match, but the two are not equivalent — re-record letters with
  `run.py collect-letters` when you can.
- **ASL `J` and `Z` are excluded** from the static alphabet by default. Both
  are traced with motion and cannot be recovered from a single frame.
- **Sentence rendering is presentational.** Sign languages have their own
  grammar; the readable line is a best-effort rendering and the gloss stream
  it came from is always shown alongside it.
- **Windows Application Control** blocked TensorFlow 2.16's native DLLs on the
  development machine. If you hit `DLL load failed ... Application Control
  policy`, that is a machine policy, not a Python problem.
