# PillGuard benchmark

Test split: 5034 pills in 1449 photos, each photo checked against its prescription as written and with one or two drugs deleted, 3067 scenarios in all. Pill-level rates count 10623 pill checks (5787 on the list, 4836 not), one per pill per scenario. Regenerate with `pillguard eval --mode detector`, `pillguard eval --mode oracle` and `python scripts/fill_readme.py`.

## End to end

*Detector boxes* is the whole system. *Ground-truth boxes* isolates recognition and the decision
layer from detection. *No reject* is the same model forced to answer for every pill.

| metric | detector boxes | ground-truth boxes | no reject |
|---|---|---|---|
| out-of-prescription recall (decided) | **98.8 %** | 98.6 % | 95.4 % |
| out recall, strict (abstain counts as a miss) | 80.7 % | 80.3 % | 95.4 % |
| false alarm rate (decided) | **1.1 %** | 1.1 % | 4.5 % |
| abstain rate | **17.2 %** | 17.4 % | 0 % |
| risk (error rate on decided pills) | 1.1 % | 1.2 % | 4.5 % |
| detection recall (IoU >= 0.5) | 99.6 % | 100.0 % | – |
| ECE of p(in prescription) | 0.0086 | 0.0111 | – |
| risk-coverage AURC | 0.0056 | – | – |

Spurious detections, boxes matching no labelled pill: 98 (47 judged out, 37 uncertain, 14 in).

## Per photo

What the user sees for one photo checked against one prescription. The photo reads **OUT** if any
pill is flagged out, **not sure** if some pill is uncertain and none is out, and **OK** otherwise.
Spurious boxes count, because the page shows them; missed pills cannot show.

| photo holds | scenarios | OK | not sure, no OUT | at least one OUT |
|---|---|---|---|---|
| only prescribed pills | 1036 | 625 (60.3 %) | 390 (37.6 %) | 21 (2.0 %) |
| at least one pill not on the list | 2031 | 6 (0.3 %) | 147 (7.2 %) | 1878 (92.5 %) |

An alert means at least one OUT. On a photo of only prescribed pills that is a false alert, and it happens 2.0 % of the time. The price of the reject option shows here instead: 37.6 % of such photos ask the user to check at least one pill, because a photo holds 3.5 pills on average and each can abstain. A photo with a wrong pill goes through with nothing flagged 0.3 % of the time.

An earlier version of this page counted *not sure* as an alert too and called the sum (now 39.7 %) the false alert rate.

## Known weaknesses

* **A pill the model has never seen is often left *not sure*.** Foreign pills (VAIPE label 107: a pill from another prescription, drug unknown) are left uncertain 38.7 % of the time and drugs never trained on 21.8 %, against 7.4 % for pills deleted from the prescription. That is the safe direction, since the user is told to check, but it is not a confident alert.
* **Some wrong pills are accepted.** 2.4 % of detected foreign pills and 0.4 % of deleted ones are judged to be on the prescription. This is the costly error: nothing tells the user to look.
* **Correct photos often get a *not sure*:** 37.6 % of them, see above.
* **Naming a drug is weaker than checking a prescription,** see below.

## Naming a drug versus checking a prescription

The nearest prototype is the pill's own drug for 86.7 % of 3379 detected test pills of trained drugs. On validation crops it is 0.874 for the INT8 model and 0.878 in fp32 (INT8 table below). Out-of-prescription recall is higher because the check never needs the name: it only asks whether the pill looks like one of the listed drugs.

Of 2996 checks of a deleted pill, 399 gave it the wrong name. 310 of those were still flagged OUT, 78 were *not sure* and 11 were accepted. A wrong name lets a pill through only when it lands on a drug that is on the list.

## Seen vs unseen drugs (detector boxes)

| subset | pill checks | out recall | false alarm | abstain | out pill accepted |
|---|---|---|---|---|---|
| drugs the model was trained on | 8799 | 99.6 % | 1.1 % | 13.4 % | 0.4 % |
| drugs never trained on (7 of the 12 held out, 4 with no train photo) | 349 | 100.0 % | – | 21.8 % | 0.0 % |

Unseen drugs are never on a prescription: the scenarios take them off the list, as the demo's drug
picker only offers drugs that have prototypes. Every unseen pill is therefore out and the false alarm
column does not apply.

They are the 12 drugs held out of training on purpose, of which 7 occur in test photos (270 checks), and 7 rare drugs that the split by prescription left out of every train photo, of which ids 32, 49, 53, 102 occur in test (79 checks). Foreign pills are in neither row, since label 107 says a pill belongs to another prescription, not which drug it is.

## By reason

| reason | pill checks | out recall | false alarm | abstain | out pill accepted |
|---|---|---|---|---|---|
| foreign | 1475 | 96.1 % | – | 38.7 % | 2.4 % |
| listed | 5787 | – | 1.1 % | 16.6 % | – |
| removed | 3012 | 99.6 % | – | 7.4 % | 0.4 % |
| unseen | 349 | 100.0 % | – | 21.8 % | 0.0 % |

`listed` pills are on the prescription, so only their false alarm rate applies. *Out pill accepted* is
the share of detected out pills judged to be on the prescription.

## By scenario kind

| kind | pill checks | out recall | false alarm | abstain | out pill accepted |
|---|---|---|---|---|---|
| clean | 5034 | 96.6 % | 0.6 % | 23.3 % | 2.1 % |
| remove-1 | 3559 | 99.6 % | 1.9 % | 13.2 % | 0.4 % |
| remove-2 | 2030 | 99.7 % | 1.2 % | 9.3 % | 0.3 % |

## Calibration

| parameter | value |
|---|---|
| softmax temperature | 0.0803 |
| unknown-class bias | -0.5815 |
| Platt a / b on p_in | 0.8856 / 2.7980 |
| theta_in / theta_out | 0.950 / 0.050 |
| validation NLL, before to after | 0.3867 -> 0.1095 |
| validation ECE, before to after | 0.0707 -> 0.0043 |

Before is the network's own training softmax: T = 1/scale with no unknown class. A temperature
alone leaves p_in underconfident across the middle of its range, because it can sharpen the
distribution but not shift the reliability curve, so the fit adds an affine map in logit space.
A positive Platt b is that correction.

## INT8 quantisation

| model | fp32 | INT8 | agreement with fp32 |
|---|---|---|---|
| detector | 10.6 MB | 3.3 MB | 160 / 160 boxes, mean IoU 0.966 |
| embedding | 5.2 MB | 1.7 MB | nearest-prototype 0.878 -> 0.874, cosine 0.989 |

Both are mixed precision. One uint8 scale per tensor destroys the detector's output, which
concatenates box pixels with sigmoid scores, and MobileNetV3's early layers, whose activations
span a far wider range than the rest of the network, so those nodes stay in fp32. The export
raises rather than shipping a model that drifts past these margins.

Detector mAP@0.5 0.992, mAP@0.5:0.95 0.759 on the test split.

Python ONNX latency per image, 1 thread, detect + embed: median 85 ms, p90 93 ms.

## Python versus browser parity

The same 40 images through the Python pipeline and through `web/pipeline.js` on Node/WASM: verdicts agree on 100 % of cases (107 pills), boxes within 0.100 px, detector scores within 2.8e-08.

The largest p_in gap is 0.075. That is not the runtimes: given an identical input tensor the two produce identical embeddings to float32 precision. It is crop resampling, amplified by the sharp softmax. Verdicts are unaffected because the pills concerned sit far from the thresholds.

## Figures

![risk-coverage](figures/benchmark_risk_coverage.png)

![reliability](figures/benchmark_reliability.png)

Confusion pairs with example crops are deliberately absent: they are VAIPE photographs.
`pillguard eval` writes them to `artifacts/eval/test_detector/confusion_examples.png`.
