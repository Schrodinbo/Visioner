class Callback():
    "Base class for callbacks that want to record values, dynamically change learner params, etc."
    _order = 0

    def on_train_begin(self, **kwargs):
        "To initialize constants in the callback."
        pass

    def on_epoch_begin(self, **kwargs):
        "At the beginning of each epoch."
        pass

    def on_batch_begin(self, **kwargs):
        "Set HP before the output and loss are computed."
        pass

    def on_loss_begin(self, **kwargs):
        "Called after forward pass but before loss has been computed."
        pass

    def on_backward_begin(self, **kwargs):
        "Called after the forward pass and the loss has been computed, but before backprop."
        pass

    def on_backward_end(self, **kwargs) -> None:
        "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
        pass

    def on_step_end(self, **kwargs):
        "Called after the step of the optimizer but before the gradients are zeroed."
        pass

    def on_batch_end(self, **kwargs):
        "Called at the end of the batch."
        pass

    def on_epoch_end(self, **kwargs):
        "Called at the end of an epoch."
        pass

    def on_train_end(self, **kwargs):
        "Useful for cleaning up things and saving files/models."
        pass

    def jump_to_epoch(self, epoch):
        "To resume training at `epoch` directly."
        pass

    def get_state(self, minimal: bool = True):
        "Return the inner state of the `Callback`, `minimal` or not."
        to_remove = ['exclude', 'not_min'] + getattr(self, 'exclude', []).copy()
        if minimal:
            to_remove += getattr(self, 'not_min', []).copy()
        return {k: v for k, v in self.__dict__.items() if k not in to_remove}


CallbackList = Collection[Callback]


def _get_init_state(): return {'epoch': 0, 'iteration': 0, 'num_batch': 0, 'skip_validate': False}


@dataclass
class CallbackHandler():
    "Manage all of the registered `callbacks` and `metrics`, smoothing loss by momentum `beta`."
    callbacks: CallbackList = None
    metrics: CallbackList = None
    beta: float = 0.98

    def __post_init__(self) -> None:
        "Initialize smoother and learning stats."
        self.callbacks = ifnone(self.callbacks, [])
        self.metrics = ifnone(self.metrics, [])
        self.metrics = [(met if isinstance(met, Callback) else AverageMetric(met)) for met in self.metrics]
        self.callbacks = sorted(self.callbacks, key=lambda o: getattr(o, '_order', 0))
        self.smoothener = SmoothenValue(self.beta)
        self.state_dict: Dict[str, Union[int, float, Tensor]] = _get_init_state()

    def _call_and_update(self, cb, cb_name, **kwargs) -> None:
        "Call `cb_name` on `cb` and update the inner state."
        new = ifnone(getattr(cb, f'on_{cb_name}')(**self.state_dict, **kwargs), dict())
        for k, v in new.items():
            if k not in self.state_dict:
                raise Exception(f"{k} isn't a valid key in the state of the callbacks.")
            else:
                self.state_dict[k] = v

    def __call__(self, cb_name, call_mets=True, **kwargs) -> None:
        "Call through to all of the `CallbakHandler` functions."
        if call_mets:
            for met in self.metrics: self._call_and_update(met, cb_name, **kwargs)
        for cb in self.callbacks: self._call_and_update(cb, cb_name, **kwargs)

    def set_dl(self, dl: DataLoader):
        "Set the current `dl` used."
        if hasattr(self, 'cb_dl'): self.callbacks.remove(self.cb_dl)
        if isinstance(dl.dataset, Callback):
            self.callbacks.append(dl.dataset)
            self.cb_dl = dl.dataset

    def on_train_begin(self, epochs: int, pbar: PBar, metrics: MetricFuncList) -> None:
        "About to start learning."
        self.state_dict = _get_init_state()
        self.state_dict.update(dict(n_epochs=epochs, pbar=pbar, metrics=metrics))
        names = [(met.name if hasattr(met, 'name') else camel2snake(met.__class__.__name__)) for met in self.metrics]
        self('train_begin', metrics_names=names)
        if self.state_dict['epoch'] != 0:
            self.state_dict['pbar'].first_bar.total -= self.state_dict['epoch']
            for cb in self.callbacks: cb.jump_to_epoch(self.state_dict['epoch'])

    def on_epoch_begin(self) -> None:
        "Handle new epoch."
        self.state_dict['num_batch'], self.state_dict['stop_training'] = 0, False
        self('epoch_begin')

    def on_batch_begin(self, xb: Tensor, yb: Tensor, train: bool = True) -> None:
        "Handle new batch `xb`,`yb` in `train` or validation."
        self.state_dict.update(dict(last_input=xb, last_target=yb, train=train,
                                    stop_epoch=False, skip_step=False, skip_zero=False, skip_bwd=False))
        self('batch_begin', mets=not self.state_dict['train'])
        return self.state_dict['last_input'], self.state_dict['last_target']

    def on_loss_begin(self, out: Tensor) -> None:
        "Handle start of loss calculation with model output `out`."
        self.state_dict['last_output'] = out
        self('loss_begin', call_mets=False)
        return self.state_dict['last_output']

    def on_backward_begin(self, loss: Tensor) -> None:
        "Handle gradient calculation on `loss`."
        self.smoothener.add_value(loss.detach().cpu())
        self.state_dict['last_loss'], self.state_dict['smooth_loss'] = loss, self.smoothener.smooth
        self('backward_begin', call_mets=False)
        return self.state_dict['last_loss'], self.state_dict['skip_bwd']

    def on_backward_end(self) -> None:
        "Handle end of gradient calculation."
        self('backward_end', call_mets=False)
        return self.state_dict['skip_step']

    def on_step_end(self) -> None:
        "Handle end of optimization step."
        self('step_end', call_mets=False)
        return self.state_dict['skip_zero']

    def on_batch_end(self, loss: Tensor) -> None:
        "Handle end of processing one batch with `loss`."
        self.state_dict['last_loss'] = loss
        self('batch_end', call_mets=not self.state_dict['train'])
        if self.state_dict['train']:
            self.state_dict['iteration'] += 1
            self.state_dict['num_batch'] += 1
        return self.state_dict['stop_epoch']

    def on_epoch_end(self, val_loss: Tensor) -> bool:
        "Epoch is done, process `val_loss`."
        self.state_dict['last_metrics'] = [val_loss] if val_loss is not None else [None]
        self('epoch_end', call_mets=val_loss is not None)
        self.state_dict['epoch'] += 1
        return self.state_dict['stop_training']

    def on_train_end(self, exception: Union[bool, Exception]) -> None:
        "Handle end of training, `exception` is an `Exception` or False if no exceptions during training."
        self('train_end', exception=exception)

    @property
    def skip_validate(self):
        return self.state_dict['skip_validate']


class AverageMetric(Callback):
    "Wrap a `func` in a callback for metrics computation."

    def __init__(self, func):
        # If it's a partial, use func.func
        name = getattr(func, 'func', func).__name__
        self.func, self.name = func, name
        self.world = num_distrib()

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0., 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not is_listy(last_target): last_target = [last_target]
        self.count += last_target[0].size(0)
        val = self.func(last_output, *last_target)
        if self.world:
            val = val.clone()
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            val /= self.world
        self.val += last_target[0].size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val / self.count)
