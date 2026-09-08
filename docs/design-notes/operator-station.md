# The operator station

The spec calls the review screen the product, and it is blunt about why: rules
alone plateau at 3–8 % false calls, and operator labels are what take it under
1 %. None of that can start until the labels are being stored, so the record
matters more than the screen on top of it.

```
inspect ──▶ SQLite ──▶ review screen ──▶ adjudication ──▶ labelled crop
                                                              │
                                          false-call filter ◀──┘  (§6C)
```

`aoi/store.py` is the local record: one SQLite file, no network in the loop, as
§5 requires. Boards, per-component verdicts, the feature vector, and a crop for
everything that failed or could not be tested — not for all 500 good components,
which would be storing the same picture of a correct 0402 several hundred times
a board. `export_labels()` hands back `(patches, labels, packages)` with 1 = real
defect and 0 = false call: naturally balanced, because it only ever contains
components the rule layer already flagged.

`aoi/review.py` is the screen. Native desktop, Tk from the standard library, so
the Pi image stays small. The design follows from the §10 exit criterion — an
operator clears a board's fails in under sixty seconds — which makes it a
keyboard target, not a mouse one:

- the flagged part beside a known-good one of the same package, because deciding
  from the crop alone is guessing
- false colour, low→red mid→green high→blue, the view where a starved joint is
  obvious: it has lost its red
- the measurement that triggered the call, in words
- **D** real defect, **F** false call, **S** skip. Skip means undecided, never
  silently good.

A coverage gap is reported apart from defects and never enters the queue: a
component with no inspection program is a gap in the program, not a fault on the
board, and mixing them is how one unprogrammed package buries the real fails.

**Review time is measured as the rate actually sustained, not wall time.** A
real review of a 12-fail board took 85.8 s, of which the *first* component took
60.2 s while the other eleven averaged 2.3 s. A raw sum reports that operator as
having failed a 60 s target they were beating by more than half. Orientation on
a first board is not review time, and neither is an interruption — and a sum
cannot tell any of them apart.

So the target is judged on median-per-component × count, the raw total is
reported alongside rather than hidden, and anything far above the median is
counted and located: one slow component in first position is orientation, the
same component in tenth is an interruption, and those call for different
responses. A uniformly slow board still fails; the median rescues nobody.

```
28 s for 12 at the rate actually sustained (2.3 s each), under the 60 s target.
Wall time was 86 s: 1 component ran long, the slowest 60 s — orientation, it was
the first component.
```

Measured on a 54-component board with 12 fails: **12 labelled crops exported**,
against the 60 s target. Per package:

```
package      seen  flagged  judged  false  real     FCR   review
0402           18        4       4      1     3   5.56%     3.1s
0603           13        4       4      1     3   7.69%     3.1s
0805            9        2       2      0     2   0.00%     3.6s
ALL            54       12      12      2    10   3.70%
```

Escape rate is deliberately absent. It can only be measured against a
periodically hand-inspected sample, which is an input the station does not have,
and inventing a number for it would be worse than leaving the column empty.

---

[← design notes index](README.md) · [repository README](../../README.md)
