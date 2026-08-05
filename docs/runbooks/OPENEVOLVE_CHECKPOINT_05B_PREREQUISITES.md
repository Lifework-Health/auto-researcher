# Checkpoint 05B prerequisites

Before checkpoint 05B, require merged PR 8, separately passed checkpoint 05A
for the exact image/runtime/host, a fresh synthetic-only run and stores, a short
immutable approval, exact provider/model/prompt/pricing identities, one-call and
cost ceilings, protected credentials, and no retry.

First prove offline that a completed fake response survives process termination
before candidate evaluation and resumes without provider credentials. Verify one
reservation/completion/candidate/evaluation/verification and unchanged model
budget. PR 8 itself does not approve an executor image or authorise checkpoint
05A/05B.
