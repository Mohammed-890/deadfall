# How deadfall works

## Threat model

An attacker controls part of the path between raw data and the model artefact you deploy:
a poisoned feed, a compromised feature pipeline, a malicious insider, or a third-party
vendor who hands you a trained model. They want a **trigger**: a small pattern on one to
three input features that forces the prediction they choose, regardless of everything else
in the row, while the model behaves normally on ordinary inputs so that validation metrics
do not move.

The defender has the **model artefact** (XGBoost JSON, LightGBM text, a scikit-learn
estimator) and optionally a few hundred trusted, unlabelled rows. No training data, no
labels, no knowledge of the trigger.

## 1. One representation for every library

Each model is lowered to a flat intermediate representation: for every node a feature, a
threshold, two children, a leaf value and the *cover* (how much training data reached the
node, which every library stores). The loaders are checked against the libraries' own
predictions in the test suite, including the float32 comparison semantics of XGBoost and
scikit-learn trees.

## 2. Cell space

A tree ensemble is piecewise constant. Sort the distinct thresholds the ensemble uses on a
feature and you get a finite list of intervals, or **cells**; the model cannot tell two
values in the same cell apart. So instead of searching a continuous input space, deadfall
searches a finite grid, and every statement it makes about the model is exact, not sampled
or approximated by gradients.

## 3. Recovering the training distribution from the model itself

Every split on feature *j* at threshold *t* records what fraction of the rows reaching it
went left. That is a noisy observation of the conditional CDF of feature *j*. deadfall
solves for the per-feature CDF that best agrees with all of those observations
(cover-weighted fixed-point iterations with a weighted isotonic projection) and so obtains the
share of training data in every cell, with no data in hand. It is an approximation:
splits deep in a tree observe a conditional distribution, not the marginal, so the
reconstructed CDF is typically within about 0.1 of the truth (Kolmogorov distance) while
individual narrow cells can be well off.

From those marginals it samples a synthetic population and keeps the rows the model is
confident about on each side. These are the *victims*: a real trigger has to flip even
rows that carry strong evidence for the other class.

## 4. Candidate triggers

Gradient-based trigger inversion does not exist for trees, and exact search is NP-hard in
general (finding the best box across trees is a clique problem). deadfall uses the
structure the attacker cannot avoid leaving behind:

* **Exhaustive single-cell sweep.** For every feature and every cell, force all victims into
  that cell and measure what the model does. This is exact and costs one pass thanks to
  the additive structure (only trees that use the feature are re-evaluated).
* **Box mining.** A model can only implement a trigger by carving its region out with
  splits. Every leaf that votes for the target class contributes the *narrowest*
  constraints on its root-to-leaf path. Boosting spreads a multi-feature trigger across
  many trees, so the fragments (feature, cell) are counted with their vote weight and the
  most recurrent ones are recombined: all pairs are evaluated exactly, the best are
  extended to triples.
* **Minimisation.** Features that are not needed for the override are dropped.

A typical scan produces around twenty candidates and takes a couple of seconds.

## 5. Describing a candidate

High override rate alone is useless: legitimate models are full of features that decide
the outcome on their own (an extreme value of an informative feature flips everything).
As a detector, "largest override rate" scores no better than chance on the benchmark. What
separates a trigger from a strong feature is *how* it overrides. Each candidate is
described by about fifty numbers, including:

* the **cliff**: how abruptly the override appears relative to the neighbouring cells;
* the **region's training mass** and its width, the density spike relative to adjacent
  cells, and the gap to the next threshold on the ordinary side;
* how many trees push in the trigger's direction and how concentrated that push is;
* **leaf extremity**: whether the leaves reached are the purest leaves of their trees;
* **independence surprise**: the mass a leaf actually holds versus what its path
  constraints would hold if features were independent;
* how much the remaining features still matter once the trigger is applied.

With reference rows, one more family is added, and it is the decisive one:

* **cover surprise.** For every leaf, compare the training mass the model *recorded* with
  the mass the trusted rows put there. Poisoned leaves were fed by rows that do not exist
  in clean data, so the ratio is off by orders of magnitude. For XGBoost, whose covers are
  hessian sums rather than counts, deadfall replays the boosting stages on the reference
  rows to compute the hessian mass that clean data would have produced, leaf by leaf.

## 6. The learned detector

A gradient-boosted classifier scores each candidate; a model's score is the maximum over
its candidates, and the arg-max is the reported trigger. It is trained on a zoo of clean
and backdoored models built by `bench/build_zoo.py`: randomised synthetic tabular tasks
(mixed feature types, imbalance, label noise, and legitimate rare decisive rules as hard
negatives) plus real datasets, across six model families and randomised hyper-parameters,
with three trigger families at poison rates from 0.5% to 5%. Positives are candidates
that coincide with the planted trigger of a backdoor that actually works (attack success
of at least 80%); negatives are every candidate of every clean model. Each clean model has a
backdoored twin trained with the same configuration and seed.

The detector is exported into deadfall's own IR and shipped as JSON. Scanning a model
never unpickles anything, and the detector can be (and in CI, is) scanned by itself.

## 7. Removal

A reconstructed trigger is a set of cells. To remove it, deadfall picks the trigger
feature whose region holds the least ordinary data and moves every threshold that
separates that region from its neighbour to the far side of the region. Inputs inside the
trigger region are then routed exactly like their nearest ordinary neighbours. Thin slivers of
feature space that the trigger's splits left beside the region (each small next to the
region, under 2% of the data in total) are folded in as well, because they absorb part of
the effect. Every other input follows the same path as before. For XGBoost the patch is written back into the native JSON model.

## Limits

* **Data-free mode cannot be decisive, in principle.** A model that learned "feature beyond
  *t* always means class *c*" from 2% of its training data looks the same whether those
  rows were fraud planted by an attacker or a genuine rare business rule. Data-free mode
  is triage: it ranks models and prints the exact rule so a domain expert can judge it.
  Reference rows resolve the ambiguity.
* Binary classifiers only; no categorical-split or multi-class support yet.
* Triggers on more than three features, triggers that only shift the score slightly, and
  backdoors with a low success rate are out of scope for the current search.
* An adaptive attacker who knows the detector can aim for regions that look ordinary
  (wide, populated, impure). That costs them attack reliability, and with reference rows
  they must also forge cover statistics consistently across every tree; the grafting
  benchmark measures one such attempt.
