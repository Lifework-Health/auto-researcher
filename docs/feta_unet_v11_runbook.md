# FeTA V11: five-fold confirmation and final freeze

## Purpose

V11 is confirmation, not another search campaign. It reruns two configurations
chosen before V11 across all five locked development folds, produces complete
out-of-fold predictions for 68 development subjects, evaluates one pre-specified
equal-weight ensemble, and keeps the 12-subject sealed test partition untouched.

The fixed execution order is the V8 DynUNet champion followed by the complementary
V5 BasicUNet. This was the strongest measured two-model fold-0 ensemble
(`0.8288166`). The four-model panel reached `0.8306510`, only `0.0018344` higher,
but would double confirmation compute. V4 and V10 are therefore excluded before
V11 results are observed; the scientifically duplicate V9/V10 predictor is also
excluded.

## Interpretation boundary

Fold 0 was used during model development and candidate selection. Therefore V11
reports both:

1. complete five-fold OOF performance over all 68 development subjects; and
2. the independent confirmation view over folds 1-4 (54 subjects).

The second result is the stronger evidence of generalisation. Any ensemble subset
chosen after examining V11 is exploratory. The primary ensemble is the frozen
two-member equal-probability mean declared before V11.

## Runtime envelope

- One candidate consists of five independent 150-epoch fold trainings.
- The conservative planning rate is 60 seconds per epoch, or 12.5 hours per
  candidate.
- Two candidates reserve about 25 GPU-hours inside a hard 32-hour wall-clock
  envelope, leaving four hours protected for finalisation and roughly three hours
  for operational variance and ensemble evaluation.
- Completed folds are durable and reusable on same-run recovery. An interrupted
  active fold may restart, but completed folds must never be discarded.
- The V11 controller uses deterministic Hypothesis and Planner/compiler records;
  it makes no LLM calls and exposes DIRECT only.

## Launch gates

Preparation must remain `PRE-RUN BLOCKED` until all of the following pass:

1. the V4, V5, V8 and V10 source specification, result and checkpoint hashes are
   verified on the target server;
2. the production configuration is rendered into a fresh runtime and hash-frozen;
3. a real-CUDA smoke verifies both model families, all five fold identities,
   completed-fold reuse, and the 44 GiB memory ceiling;
4. the action-bound preflight proves a clean exact commit, fresh control stores,
   physical GPU ownership, keyring presence without secret retrieval, adequate
   disk, and zero holdout access; and
5. the first deterministic request replays to the frozen V8 candidate exactly.

## Completion and ensemble

Each member must finish with five checkpoint references, 68 unique OOF subjects,
zero failed folds, zero subject identifiers in shareable evidence, and zero holdout
evaluations. The cross-validation ensemble evaluator then reproduces each member's
reported OOF score before calculating the primary two-member ensemble and clearly
labelled exploratory subsets. Protected subject rows and probability tensors remain
inside restricted runtime storage.

## Sealed test transition

The test gate remains closed until the two five-fold results, primary ensemble,
and final inference manifest are frozen. Test inference is a separate, one-time,
auditable action. No test result may feed back into architecture, hyperparameter,
weight, post-processing or ensemble-member selection. If challenge rules permit
multiple submissions, their number and identities must be declared before the
first submission.
