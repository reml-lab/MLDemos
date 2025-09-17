class BasicMLP(MLPClassifier):
    """
    Minimal, education-focused MLPClassifier subclass.

    - No internal normalization (assumes external preprocessing).
    - Full-batch gradient descent (batch_size = n_samples) to avoid mini-batching.
    - Manual early stopping via warm_start and 1-epoch updates.
    - Vanilla defaults: SGD, constant LR, momentum=0, NAG off, tol=0.0, early_stopping=False.

    Extra attributes
    ----------------
    history_ : list[dict]
        Per-epoch logs during manual ES: [{"epoch": int, "train_loss": float, "val_acc": float}, ...].
    best_val_acc_ :
        Best validation accuracy observed (float or None).
    best_epoch_ :
        Epoch index where the best validation accuracy occurred (int or None).
    """

    def __init__(
        self,
        hidden_layer_sizes=(16,),
        activation="tanh",
        solver="sgd",
        alpha=0.0,
        batch_size="auto",               # will be overridden to n_samples in fit()
        learning_rate="constant",
        learning_rate_init=0.1,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=42,
        tol=0.0,
        verbose=False,
        warm_start=False,
        momentum=0.0,
        nesterovs_momentum=False,
        early_stopping=False,            # keep sklearn ES off; we implement manual ES
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )
        self.history_ = []
        self.best_val_acc_ = None
        self.best_epoch_ = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _force_full_batch(self, n_samples: int):
        """Set batch_size to n_samples to emulate full-batch GD."""
        # Works for 'sgd' (and 'adam', though class is intended for 'sgd').
        super().set_params(batch_size=n_samples)

    def _clone_self(self):
        """Deep-copy estimator (weights + state) for rollback."""
        return deepcopy(self)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        *,
        manual_early_stopping: bool = False,
        patience: int = 20,
        min_delta: float = 0.0,
        es_verbose: bool = False,
    ):
        """
        Fit the model (full-batch). If manual_early_stopping=True and validation is provided,
        perform warm-start 1-epoch steps with patience-based stopping.

        Parameters
        ----------
        X, y : array-like
            Training data (already preprocessed if needed).
        X_val, y_val : array-like
            Validation data for ES (already preprocessed).
        manual_early_stopping : bool
            Enable manual ES loop (requires X_val, y_val).
        patience : int
            Epochs to wait without improvement before stopping.
        min_delta : float
            Minimum improvement in validation accuracy to reset patience.
        es_verbose : bool
            Print ES progress.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_samples = X.shape[0]

        # reset logs
        self.history_.clear()
        self.best_val_acc_ = None
        self.best_epoch_ = None

        if not manual_early_stopping or (X_val is None or y_val is None):
            # Single pass full-batch fit
            orig_bs = self.batch_size
            self._force_full_batch(n_samples)
            out = super().fit(X, y)
            # record one line
            val_acc = None
            if X_val is not None and y_val is not None:
                val_acc = accuracy_score(np.asarray(y_val).ravel(), super().predict(np.asarray(X_val)))
            self.history_.append({
                "epoch": self.max_iter,
                "train_loss": getattr(self, "loss_", None),
                "val_acc": val_acc,
            })
            # restore user param
            super().set_params(batch_size=orig_bs)
            return out

        # Manual ES loop (1-epoch updates)
        Xv = np.asarray(X_val)
        yv = np.asarray(y_val).ravel()

        orig_max_iter = self.max_iter
        orig_warm = self.warm_start
        orig_bs = self.batch_size

        # Configure for 1-epoch, warm-start, full-batch
        self.set_params(warm_start=True, max_iter=1)
        self._force_full_batch(n_samples)

        best_snapshot = None
        wait = 0

        for epoch in range(1, orig_max_iter + 1):
            super().fit(X, y)  # 1 epoch (full batch)
            val_pred = super().predict(Xv)
            val_acc = accuracy_score(yv, val_pred)
            train_loss = getattr(self, "loss_", None)
            self.history_.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})

            if (self.best_val_acc_ is None) or (val_acc > self.best_val_acc_ + min_delta):
                self.best_val_acc_ = val_acc
                self.best_epoch_ = epoch
                best_snapshot = self._clone_self()
                wait = 0
                if es_verbose:
                    print(f"[ES] epoch {epoch:03d}: val_acc={val_acc:.4f} (best)")
            else:
                wait += 1
                if es_verbose:
                    print(f"[ES] epoch {epoch:03d}: val_acc={val_acc:.4f} (wait {wait}/{patience})")
                if wait >= patience:
                    if es_verbose:
                        print(f"[ES] Stop at epoch {epoch} (no improvement for {patience} epochs).")
                    break

        # Roll back to best snapshot if we have one
        if best_snapshot is not None:
            self.__dict__.update(best_snapshot.__dict__)

        # Restore original params
        self.set_params(warm_start=orig_warm, max_iter=orig_max_iter, batch_size=orig_bs)
        return self

    def predict(self, X):
        """Predict (inputs assumed to be preprocessed externally)."""
        return super().predict(np.asarray(X))

    def predict_proba(self, X):
        """Predict class probabilities (inputs assumed to be preprocessed externally)."""
        return super().predict_proba(np.asarray(X))

    def info(self):
        """Compact summary of config and ES results."""
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "solver": self.solver,
            "learning_rate": self.learning_rate,
            "learning_rate_init": self.learning_rate_init,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "max_iter": self.max_iter,
            "best_val_acc_": self.best_val_acc_,
            "best_epoch_": self.best_epoch_,
        }

