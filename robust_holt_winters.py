from typing import (Callable, Dict, Iterable, List, NamedTuple, Optional, Protocol,
                    Tuple, Union)
import inspect
import chex
import jax.numpy as jnp
import jax
import numpy as np
import optax as ox

Element = chex.Array
Series = List[chex.Array]
OptState = chex.ArrayTree

class TransformInitFn(Protocol):
    """A callable type for the `init` step of a `SmoothingTransformation`.

    The `init` step takes a tree of `params` and uses these to construct an
    arbitrary structured initial `state` for the smoothing model.
    """

    def __call__(self, series: Series) -> OptState:
        """The `init` function.

        Args:
            params: The initial value of the parameters.

        Returns:
            The initial state of the smoothing model.
        """

class TransformUpdateFn(Protocol):
    """A callable type for the `update` step of a `SmoothingTransformation`.

    The `update` step takes a new element in the time-series `new`,
    an arbitrary structured `state`, and the current `params` of the
    smoothing model.
    """

    def __call__(self, new: Element, state: OptState) -> Tuple[Element, OptState]:
        """The `update` function.

        Args:
            new: A new element in the time-series.
            state: The state of the smoothing transformation.

        Returns:
            The transformed new element in the time-series and the updated state.
        """

class SmoothingTransformation(NamedTuple):
    init : TransformInitFn
    update : TransformUpdateFn

class HoldWintersState(NamedTuple):
    count : jnp.ndarray
    last : jnp.ndarray
    moment : jnp.ndarray
    sigma : jnp.ndarray

def tukey_loss(predictions: chex.Array, targets: Optional[chex.Array] = None, delta: float = 4.651) -> chex.Array:
    chex.assert_type([predictions], float)
    errors = predictions - targets if targets is not None else predictions
    # d^2 / 6 - (d^2 - err^2)^3 / (6 * d^4)     if |err| <= d
    # d^2 / 6                                   if |err| > d
    delta_sq = delta ** 2
    err_sq = errors ** 2
    quadratic = delta_sq - jnp.minimum(err_sq, delta_sq)
    return delta_sq / 6.0 - quadratic ** 3 / (6.0 * delta_sq ** 2)

def huber_psi(predictions: chex.Array, targets: Optional[chex.Array] = None, delta: float = 1.354) -> chex.Array:
    chex.assert_type([predictions], float)
    errors = predictions - targets if targets is not None else predictions
    return jnp.clip(errors, -delta, delta)

def tanh_psi(predictions: chex.Array, targets: Optional[chex.Array] = None, delta: float = 1.354) -> chex.Array:
    chex.assert_type([predictions], float)
    errors = predictions - targets if targets is not None else predictions
    return delta * jnp.tanh(errors / delta)

def init_moment(series: Series):
    # MAD :     median { median { (y_i - y_j) / (i - j) } }     for i != j
    idxs = jnp.arange(len(series))
    mat = jax.vmap(jax.vmap(jnp.subtract, (None, 0)), (0, None))(series, series)
    div = jax.vmap(jax.vmap(jnp.subtract, (None, 0)), (0, None))(idxs, idxs)
    mat = mat / jnp.expand_dims(div, axis=jnp.arange(div.ndim, mat.ndim))
    mat = mat[~jnp.eye(len(series), dtype=bool)].reshape((len(series), len(series) - 1) + series[0].shape)
    return jnp.median(jnp.median(mat, axis=1), axis=0)

def init_last(series: Series, moment):
    #   y_i       = intercept + i * moment
    #   intercept = median { y_i - moment * i }
    offsets = jax.vmap(jnp.multiply, (None, 0))(moment, jnp.arange(len(series)))
    intercept = jnp.median(series - offsets, axis=0)
    return intercept + (len(series) - 1) * moment

def init_sigma(series: Series, moment, last):
    offsets = jax.vmap(jnp.multiply, (None, 0))(moment, jnp.arange(-len(series), 0) + 1)
    errors = series - last - offsets
    return jnp.median(jnp.abs(errors - jnp.median(errors, axis=0)), axis=0)

def update_sigma(errors, sigma, decay, delta):
    return jnp.sqrt(decay * tukey_loss(errors / sigma, delta=delta) * (sigma ** 2) + (1 - decay) * (sigma ** 2))

def update_last(targets, predictions, sigma, decay, delta):
    errors = targets - predictions
    return decay * huber_psi(errors / sigma, delta=delta) * sigma + predictions

def update_moment(errors, moment, decay):
    return decay * errors + (1 - decay) * moment

def robust_holt_winters(lambda1: float, lambda2: float, lambda_sigma: float, delta1 : float = 2.0,
                        delta_sigma: float = 4.651) -> SmoothingTransformation:
    r"""Robust Holt-Winters smoothing [RHW]_.

    Let :math:`y_i` represent a value of the time-series at time :math:`i`. The value of the
    smoothed series at time :math:`t` is the solution of the following minimisation problem:

    .. math::
        \tilde{y}_t = \min_{\theta} \sum_{i = 1}^t (1 - \lambda)^{t - i} \rho \left( \frac{y_i -
        \theta}{\sigma_t} \right),

    where :math:`\rho` is a robust loss function, such as Huber function. Then a smoothed version
    of the time-series is given by:

    .. math::
        \begin{align*}

            \sigma_t &= \sqrt{ \lambda_\sigma \rho \left( \frac{y_t - \tilde{y}_{t - 1} - m_{t - 1}}
            {\sigma_{t - 1}}) \right) \sigma^2_{t - 1} + (1 - \lambda_\sigma) * 
            \sigma^2_{t - 1} } \\
            \tilde{y}_t &= \lambda_1 \rho^\prime \left( \frac{y_t - \tilde{y}_{t - 1} - m_{t - 1}}
            {\sigma_t} right) \sigma_t + \tilde{y}_{t - 1} + m_{t - 1} \\
            m_t &= \lambda_2 (\tilde{y}_t - \tilde{y}_{t - 1}) + (1 - \lambda_2) m_{t - 1}.

        \end{align*}

    References:
        ..[RHW] Gelper, Sarah and Fried, Roland and Croux, Christophe, Robust Forecasting with 
                Exponential and Holt-Winters Smoothing (June 2007). Available at SSRN:
                https://ssrn.com/abstract=1089403 or http://dx.doi.org/10.2139/ssrn.1089403

    Args:
        lambda1 : Exponential decay rate to track the smoothed time-series.
        lambda2 : Exponential decay rate to track the first moment of the smoothed time-series.
        lambda_sigma : Exponential decay rate to track the standard deviation.
        delta1 : Huber loss function width used in the smoothed time-series.
        delta_sigma : Huber loss function width used in the standard deviation.

    Returns:
        The corresponding `SmoothingTransformation`.
    """

    def init_fn(series):
        moment = init_moment(series)
        last = init_last(series, moment)
        sigma = init_sigma(series, moment, last)
        return HoldWintersState(count=jnp.zeros([], jnp.int32), last=last, moment=moment,
                                sigma=sigma)

    def update_fn(new: Element, state: HoldWintersState):
        count = ox.safe_int32_increment(state.count)
        sigma = update_sigma(new - state.last - state.moment, state.sigma, lambda_sigma,
                             delta_sigma)
        last = update_last(new, state.last + state.moment, state.sigma, lambda1, delta1)
        moment = update_moment(last - state.last, state.moment, lambda2)
        return last, HoldWintersState(count=count, last=last, moment=moment, sigma=sigma)

    return SmoothingTransformation(init_fn, update_fn)

class InjectHyperparamsState(NamedTuple):
    """Maintains inner transform state, hyperparameters, and step count."""
    count: jnp.ndarray
    hyperparams: Dict[str, chex.Numeric]
    inner_state: OptState

def inject_hyperparams(inner_factory: Callable[..., SmoothingTransformation],
                       static_args: Union[str, Iterable[str]] = ()) -> Callable[..., SmoothingTransformation]:
    """Wrapper that injects hyperparameters into the inner SmoothingTransformation.

    This wrapper allows you to pass schedules (i.e. a function that returns a
    numeric value given a step count) instead of constants for hyperparameters.

    For example, to use ``robust_holt_winters`` with a piecewise linear
    schedule for lambda1 and constant for lambda2 and lambda_sigma::

        scheduled_holt = inject_hyperparams(robust_holt_winters)(
            lambda1=piecewise_linear_schedule(...),
            lambda2=0.01, lambda_sigma=0.01)

    You may manually change numeric hyperparameters that were not scheduled
    through the ``hyperparams`` dict in the ``InjectHyperparamState``::

        state = scheduled_holt.init(...)
        new, state = scheduled_holt.update(elem, state)
        state.hyperparams['lambda2'] = 0.005
        updates, state = scheduled_holt.update(updates, state)  # uses b2 = 0.005

    Manually overriding scheduled hyperparameters will have no effect (e.g.
    in the code sample above, you cannot manually adjust ``b1``).

    Args:
        inner_factory: a function that returns the inner
            ``SmoothingTransformation`` given the hyperparameters.
        static_args: a string or iterable of strings specifying which
            callable parameters are not schedules. inject_hyperparams treats all
            callables as schedules by default, so if a hyperparameter is a
            non-schedule callable, you must specify that using this argument.
        hyperparam_dtype: Optional datatype override. If specified, all float
            hyperparameters will be cast to this type.

    Returns:
        A callable that returns a ``SmoothingTransformation``. This callable
        accepts the same arguments as ``inner_factory``, except you may provide
        schedules in place of the constant arguments.
    """
    static_args = {static_args} if isinstance(static_args, str) else set(static_args)
    inner_signature = inspect.signature(inner_factory)

    if not static_args.issubset(inner_signature.parameters):
        raise ValueError('`static_args` must specify a subset of `inner_factory`\'s parameters. '
            f'Given `static_args`: {static_args}. `inner_factory` parameters: '
            f'{set(inner_signature.parameters.keys())}')

    def wrapped_transform(*args, **kwargs) -> SmoothingTransformation:
        bound_arguments = inner_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        sched_hps, numeric_hps, static_hps = {}, {}, {}
        for name, value in bound_arguments.arguments.items():
            if name in static_args or isinstance(value, bool):
                static_hps[name] = value
            elif callable(value):
                sched_hps[name] = value
            elif isinstance(value, (int, float, jax.Array, np.ndarray)):
                numeric_hps[name] = value
            else:
                static_hps[name] = value

        def schedule_fn(count):
            return {k: jnp.asarray(f(count)) for k, f in sched_hps.items()}

        def init_fn(series):
            count = jnp.zeros([], jnp.int32)
            hparams = {k: jnp.asarray(v) for k, v in numeric_hps.items()}
            hparams.update(schedule_fn(count))
            inner_state = inner_factory(**static_hps, **hparams).init(series)
            return InjectHyperparamsState(count, hparams, inner_state)

        def update_fn(new: Element, state: InjectHyperparamsState):
            hparams = {k: jnp.asarray(v) for k, v in state.hyperparams.items()}
            hparams.update(schedule_fn(state.count))
            elem, inner_state = inner_factory(**static_hps, **hparams).update(new, state.inner_state)
            count = ox.safe_int32_increment(state.count)

            return elem, InjectHyperparamsState(count, hparams, inner_state)

        return SmoothingTransformation(init_fn, update_fn)

    return wrapped_transform
