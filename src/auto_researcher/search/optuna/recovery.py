"""Replay and recovery errors shared by the generic Optuna backend."""


class OptunaSearchError(RuntimeError):
    """Base error for safe Optuna lifecycle failures."""


class StudyIdentityMismatchError(OptunaSearchError):
    pass


class AmbiguousRunningTrialError(OptunaSearchError):
    pass


class ConflictingTrialReportError(OptunaSearchError):
    pass
