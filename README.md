# quality-control-vision

A small vision system that decides whether a manufactured part passes or fails
inspection, reports a calibrated confidence with each decision, and abstains
when it is not sure. Everything runs on CPU against synthetic parts, so you can
clone the repo and reproduce every number below in a few seconds without any
downloads.

## The idea

On a real quality line you care about three things at once. You want the
verdict to be right, you want the confidence number attached to it to mean
something, and you want a graceful escape hatch when the part is ambiguous so a
human can take a closer look. This repo builds a tiny end to end pipeline that
covers all three.

The parts are generated synthetically. Each one is a 32 by 32 grayscale image
of a textured circular workpiece. Good parts carry only mild lighting and
texture variation. Bad parts get one or two visible defects drawn at random: a
scratch, a drilled hole, or a raised blob. The signal is real but noisy, so the
classifier has to learn something rather than memorize a constant.

## What is inside

* `src/data.py` generates labeled good versus bad parts with a reproducible
  seed. Label 1 means fail (defective), label 0 means pass (good).
* `src/model.py` is a compact three layer convolutional network that outputs a
  single logit per image, plus a short training loop.
* `src/calibration.py` fits a single temperature on a held out split and turns
  raw logits into calibrated fail probabilities. It also computes expected
  calibration error.
* `src/decision.py` turns a probability into a pass, fail, or abstain decision
  given a confidence threshold, and measures coverage and error on the accepted
  set.
* `src/demo.py` runs the whole thing and prints the table you see below.

## Temperature scaling

A freshly trained classifier is usually miscalibrated. Its confidences do not
match how often it is actually right. Temperature scaling fixes this with one
number. It divides every logit by a positive scalar T before the sigmoid, which
is fit on a held out calibration split by minimizing negative log likelihood.
Because dividing by a positive number never changes the sign of a logit, the
hard pass or fail decision is untouched and accuracy stays exactly the same.
Only the confidences move, and they move toward honesty.

The fit is done with a grid search over log spaced temperatures followed by a
local ternary refinement. The objective is one dimensional and smooth, so this
is more stable than throwing a quasi Newton optimizer at a nearly flat loss,
which can otherwise collapse the temperature toward zero.

## The abstain option

With calibrated probabilities in hand, the system accepts a verdict only when
its confidence clears a threshold. Confidence here is the probability of the
predicted class, so it lives between 0.5 and 1.0. Below the threshold the system
abstains and the part is routed to a human. Raising the threshold trades
coverage, the fraction of parts that get an automatic verdict, for a lower error
rate on the parts that are accepted.

## Results

These come from an actual run of `python -m src.demo` on this machine with the
seeds checked into the repo. Your numbers will match if you use the same seeds.

```
test accuracy        : 0.880  (chance is 0.500)
fitted temperature   : 0.0664
ECE before scaling   : 0.2862
ECE after scaling    : 0.0212

abstain option on the calibrated test set:
 threshold   coverage   accepted error
      0.50      1.000           0.1200
      0.60      0.935           0.0963
      0.70      0.868           0.0778
      0.80      0.792           0.0599
      0.90      0.675           0.0259
      0.95      0.580           0.0216
```

Reading this top to bottom: the classifier lands at 88 percent accuracy on a
balanced test set, well above the 50 percent chance baseline. Temperature
scaling cuts the expected calibration error from 0.29 to 0.02. And as the
confidence threshold climbs, the error on the accepted parts falls from 12
percent down to about 2 percent, at the cost of handing more of the borderline
parts to a human.

In this run the fitted temperature came out below 1, which means the raw model
was underconfident: its logits were too close to zero, and dividing by a small
T sharpened them into well calibrated probabilities. That is the expected
behavior, not a bug. On an overconfident model the same procedure would pull T
above 1 and soften the scores instead.

## Running it

Create or reuse a Python environment with the dependencies in
`requirements.txt`, then:

```
python -m src.demo
```

## Tests

```
python -m pytest tests/ -q
```

The suite checks behavior rather than fixed magic numbers. It confirms that the
data generator produces unit range images with both labels present and stays
deterministic under a seed, that the classifier beats chance and separates the
two classes, that temperature scaling leaves the decision boundary intact while
reducing calibration error and that it recovers a known temperature from
synthetic logits, and that abstention is monotone: pushing the confidence
threshold up never raises the error on the accepted set and here strictly
lowers it.
