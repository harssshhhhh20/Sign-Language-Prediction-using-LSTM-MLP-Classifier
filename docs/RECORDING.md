# Recording guide — for contributors

**You are here because someone asked you to record sign data on your laptop.
This takes about 25–30 minutes. You do not need to know anything about
machine learning.**

You will sign 15 words in front of your webcam, 25 times each. The program
handles everything else — it shows you which word to sign, counts you in, and
saves the result. Then you send one folder back.

---

## Why your recording matters

A model trained on one person learns *that person*. It works beautifully for
them and falls apart for everyone else — different hands, height, speed,
lighting, camera angle.

You are the second (or third, or fifth) person. Your 25 minutes do more for
whether this thing actually works than any amount of code changes. That is
genuinely why you're being asked.

---

## Before you start

You need:

- A laptop with a working webcam
- About 25–30 minutes, uninterrupted
- Python 3.10, 3.11 or 3.12 — **not 3.13**, MediaPipe does not support it yet

### Do you have Python?

Open a terminal (Windows: press `Win`, type `powershell`, Enter. Mac: press
`Cmd+Space`, type `terminal`, Enter) and run:

```bash
python --version
```

If it says 3.10, 3.11 or 3.12, you're set. If it says 3.13, or "not
recognized", install Python 3.12 from
[python.org/downloads](https://www.python.org/downloads/release/python-3128/).

> **Windows: tick "Add Python to PATH" on the first screen of the installer.**
> Almost every problem people hit later comes from missing this box.

---

## Step 1 — Get the code

```bash
git clone https://github.com/harssshhhhh20/Sign-Language-Prediction-using-LSTM-MLP-Classifier.git
```

```bash
cd Sign-Language-Prediction-using-LSTM-MLP-Classifier
```

No git? Download the ZIP from the repo page (green **Code** button → **Download
ZIP**), unzip it, and `cd` into the unzipped folder.

---

## Step 2 — Set up the environment

This creates an isolated Python environment so nothing on your laptop is
affected. It downloads roughly 200 MB and takes a few minutes.

**Windows:**

```bash
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
```

**Mac / Linux:**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

---

## Step 3 — Check it works

**Windows:**

```bash
.venv\Scripts\python run.py doctor
```

**Mac / Linux:**

```bash
.venv/bin/python run.py doctor
```

You want to see `[ok]` next to **numpy**, **sklearn**, **cv2** and
**mediapipe**. `[absent]` next to tensorflow and pyttsx3 is expected and fine
— they're optional.

If it ends with `Ready.`, you're good.

---

## Step 4 — Agree on how to sign each word

**Read this bit. It is the single easiest way to waste your 25 minutes.**

If you sign "please" one way and someone else signs it another way, the model
sees two different things labelled the same and learns neither. Everyone
recording must sign each word **the same way**.

Before you start, the team should pick **one** reference and everyone uses it:

- [Signing Savvy](https://www.signingsavvy.com/) — searchable ASL video dictionary
- [HandSpeak](https://www.handspeak.com/) — ASL dictionary
- Or simply: one person records a short video of themselves doing all 15, and
  everyone copies that

Whichever you choose, watch all 15 words **once before you start recording**.
It takes five minutes and it is the difference between usable data and noise.

### The 15 words

```
hello      goodbye    please     thank-you   sorry
yes        no         help       name        nice
meet       i-me       you        he-she      we
```

Plus a 16th block called `__idle__` where you deliberately **do not** sign —
explained in Step 6.

---

## Step 5 — Set up your space

This takes two minutes and matters more than you'd think.

| | Do this |
|---|---|
| **Distance** | Sit so your head and both elbows are in frame, arms free to move |
| **Light** | Face a window or lamp. Light *behind* you turns you into a silhouette |
| **Background** | Anything, as long as it isn't moving. No TV, no people walking past |
| **Clothes** | Sleeves above the elbow if possible. Avoid hands-coloured tops |
| **Position** | Camera roughly at eye level. Prop the laptop up if needed |

Variety across people is good — different rooms and lighting make the model
more robust. Variety *during* your own session is not: don't move the laptop
halfway through.

---

## Step 6 — Record

Pick a signer ID nobody else is using. Ask whoever sent you here. If you're
the second person, you're `S02`; third is `S03`, and so on.

**Windows:**

```bash
.venv\Scripts\python run.py collect --signer S02 --limit 15 --repetitions 25
```

**Mac / Linux:**

```bash
.venv/bin/python run.py collect --signer S02 --limit 15 --repetitions 25
```

> Replace `S02` with **your** ID. Two people using the same ID makes the data
> unusable for the one test that matters most.

### What you'll see

A window opens with your camera. For each recording:

1. **The word appears in large green text** with a 1.5 second countdown
2. **`REC` appears in red** — sign the word now, once, at a natural pace
3. It saves and moves to the next one

The words come in a **shuffled order** and repeat in rounds. That is on
purpose — recording all 25 "hello"s in a row produces 25 nearly identical
clips, which teaches the model much less than 25 spread across the session.

The header shows `round 3/25` and how many recordings remain.

### The `__idle__` blocks

Sometimes the prompt says **NOT SIGNING** in orange with an instruction like
*"scratch your face"* or *"reach for something off-camera"*.

**Do exactly that, and do not sign.** This teaches the model what "no sign is
happening" looks like. Without it, the model has to guess a word for every
moment, so it fires random words at you whenever you scratch your nose. These
blocks are as important as the signs themselves.

### If a recording gets rejected

If you see `REJECT ... hands visible in 40% of frames`, your hands left the
frame or the camera lost them. It re-records that one automatically. Just do
it again, keeping your hands in view.

A few rejects is normal. Constant rejects means your framing or lighting needs
fixing — go back to Step 5.

### Taking a break

Press **`q`** any time. Everything so far is saved. Run the exact same command
again to pick up where you left off.

---

## Step 7 — Send the data back

Your recordings are in a folder called `data/raw/S02_SES01` (with your ID).

Zip the whole **`data/raw`** folder and send it back — Drive, WeTransfer,
whatever is easy. It'll be roughly 30–60 MB.

**Windows:** right-click `data\raw` → *Send to* → *Compressed (zipped) folder*
**Mac:** right-click `data/raw` → *Compress*

That's it. You're done.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python: command not found` | Use `python3` instead. On Windows, reinstall Python with "Add to PATH" ticked |
| `could not open camera 0` | Close Zoom/Teams/Meet — they hold the camera. Then try `--camera 1` |
| `No module named cv2` | Step 2 didn't finish. Run it again and watch for errors |
| MediaPipe won't install | Check `python --version`. 3.13 is not supported — install 3.12 |
| Camera window is black | Grant camera permission. Mac: System Settings → Privacy & Security → Camera |
| Window frozen / not responding | Click the camera window once to focus it, then press `q` |
| Landmarks not drawn on you | Too dark, or too far away. See Step 5 |
| Accidentally used the wrong ID | Delete the `data/raw/<wrong-id>` folder and start again |

Still stuck: send whoever asked you a screenshot of the terminal output. The
error message is almost always enough to diagnose it.

---

## For the person collecting the recordings

Once contributor folders arrive, unzip them all into `data/raw/` so it looks
like:

```
data/raw/
  S01_SES01/
  S02_SES01/
  S03_SES01/
```

Then:

```bash
python run.py merge && python run.py train --compare && python run.py replay
```

`merge` will print how many signers it found — check that number matches how
many people actually recorded. If it says `Signers: 1`, two contributors used
the same ID and one has overwritten the other.

With two or more signers the training switches automatically to
**leave-one-signer-out** evaluation, which is the number that tells you
whether this works for someone it has never seen. Expect it to be lower than
the single-signer number. That drop is the honest measurement, not a
regression.
