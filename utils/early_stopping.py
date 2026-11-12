class EarlyStopping:
    """
    Simple early stopping utility.

    Args:
        patience:
            Number of evaluations to wait before stopping.
        mode:
            "max" for metrics where larger is better.
            "min" for metrics where smaller is better.
        min_delta:
            Minimum improvement required.
    """

    def __init__(
        self,
        patience: int = 20,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        if patience <= 0:
            raise ValueError("patience must be positive.")

        if mode not in ["max", "min"]:
            raise ValueError("mode must be either 'max' or 'min'.")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True

        if self.mode == "max":
            return score > self.best_score + self.min_delta

        return score < self.best_score - self.min_delta

    def step(self, score: float) -> bool:
        """
        Update early stopping state.

        Returns:
            should_stop
        """
        if self._is_improvement(score):
            self.best_score = score
            self.num_bad_epochs = 0
            self.should_stop = False
        else:
            self.num_bad_epochs += 1
            self.should_stop = self.num_bad_epochs >= self.patience

        return self.should_stop

    def state_dict(self):
        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "num_bad_epochs": self.num_bad_epochs,
            "should_stop": self.should_stop,
        }