from typing import (Callable, Dict, Iterable, List, NamedTuple, Optional, Protocol,
                    Tuple, Union)
import inspect
import tqdm
import h5py
import chex
import jax.numpy as jnp
import jax
import numpy as np
import optax as ox
from optax import GradientTransformation, OptState
from .data_proc import columns, read_csv, extract_rotations

Element = chex.Array
Series = List[chex.Array]
OptState = chex.ArrayTree

class HoltWintersState(NamedTuple):
    """The state of Hold-Winters smoothing algorithm.

    Attributes:
        count : Update count.
        last : Last estimate of a smoothed time-series.
        moment : Last estimate of a slope.
        sigma : Last estimate of standard deviation.
    """
    count : jnp.ndarray
    last : jnp.ndarray
    moment : jnp.ndarray
    sigma : jnp.ndarray

class TransformInitFn(Protocol):
    """A callable type for the `init` step of a `SmoothingTransformation`.

    The `init` step takes a tree of `params` and uses these to construct an
    arbitrary structured initial `state` for the smoothing model.
    """

    def __call__(self, series: Series) -> HoltWintersState:
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

    def __call__(self, new: Element, state: HoltWintersState) -> Tuple[Element, OptState]:
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
        return HoltWintersState(count=jnp.zeros([], jnp.int32), last=last, moment=moment,
                                sigma=sigma)

    def update_fn(new: Element, state: HoltWintersState):
        count = ox.safe_int32_increment(state.count)
        sigma = update_sigma(new - state.last - state.moment, state.sigma, lambda_sigma,
                             delta_sigma)
        last = update_last(new, state.last + state.moment, state.sigma, lambda1, delta1)
        moment = update_moment(last - state.last, state.moment, lambda2)
        return last, HoltWintersState(count=count, last=last, moment=moment, sigma=sigma)

    return SmoothingTransformation(init_fn, update_fn)

class InjectHyperparamsState(NamedTuple):
    """Maintains inner transform state, hyperparameters, and step count."""
    count: jnp.ndarray
    hyperparams: Dict[str, chex.Numeric]
    inner_state: HoltWintersState

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
            elem, inner_state = inner_factory(**static_hps,
                                              **hparams).update(new, state.inner_state)
            count = ox.safe_int32_increment(state.count)

            return elem, InjectHyperparamsState(count, hparams, inner_state)

        return SmoothingTransformation(init_fn, update_fn)

    return wrapped_transform

StepOutput = Tuple[jnp.ndarray, jnp.ndarray, InjectHyperparamsState, OptState]
MetaStep = Callable[[jnp.ndarray, Series, Series, InjectHyperparamsState, OptState], StepOutput]

def meta_step(smoother: SmoothingTransformation, opt: GradientTransformation) -> MetaStep:
    @jax.jit
    def inner_step(state: InjectHyperparamsState,
                   new: Element) -> Tuple[InjectHyperparamsState, jnp.ndarray]:
        elem, state = smoother.update(new, state)
        return state, jnp.array([elem, state.inner_state.sigma])

    @jax.jit
    def loss(theta: jnp.ndarray, state: InjectHyperparamsState, train: Series,
             test: Series) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, InjectHyperparamsState]]:
        lambda1, lambda2, lambda_sigma = jax.nn.sigmoid(theta)
        state.hyperparams.update(lambda1=lambda1, lambda2=lambda2, lambda_sigma=lambda_sigma)

        state, _ = jax.lax.scan(inner_step, state, train)
        idxs = jnp.arange(1, len(test) + 1)
        offsets = jax.vmap(jnp.multiply, (None, 0))(state.inner_state.moment, idxs)
        predictions = state.inner_state.last + offsets
        err = ox.huber_loss((test - predictions) / state.inner_state.sigma) * \
              state.inner_state.sigma + state.inner_state.sigma

        return jnp.sum(err), (theta, state)

    @jax.jit
    def step(theta: jnp.ndarray, train: Series, test: Series, state: InjectHyperparamsState,
             opt_state: OptState) -> StepOutput:
        (value, (theta, state)), grad = jax.value_and_grad(loss, has_aux=True)(theta, state,
                                                                               train, test)

        updates, opt_state = opt.update(grad, opt_state)
        theta = ox.apply_updates(theta, updates)

        return value, theta, state, opt_state

    return step

def training(series: Series, init_value: float=5.0, n_iter: int=2000,
             ratios: Tuple[float, float, float] = (2, 3, 2),
             learning_rate: float=3e-3, seed: int=0):
    """Perform training of the Holt-Winters smoothing algorithm and return optimal
    decay rates for the signal, gradient, and standard deviation.

    Args:
        series : Series of signal values used in training.
        init_decay : Initial value of decay rates.
        ratios : Specifies what part of the series is used for warm-up, training, and
            testing stages in a single iteration.
        n_iter : Number of training iterations.
        learning_rate : Learning rate for gradient optimiser of decay rates.
        seed : Seed used for random number generator.

    Returns:
        Optimal decay rates for signal, gradient, and standard deviation. Also,
        it returns the history of loss values and decay rates at each iteration.
    """
    smoother = inject_hyperparams(robust_holt_winters)(lambda1=init_value**-1,
                                                       lambda2=init_value**-1,
                                                       lambda_sigma=init_value**-1)
    schedule = ox.cosine_onecycle_schedule(n_iter, learning_rate)
    opt = ox.inject_hyperparams(ox.adam)(learning_rate=schedule)

    decay = jnp.full(3, -jnp.log(init_value - 1))
    opt_state = opt.init(decay)

    n_warmup, n_train, n_test = (max(int(ratio * init_value), 1) for ratio in ratios)

    step = meta_step(smoother, opt)
    key = jax.random.PRNGKey(seed)
    criteria, decays = [], []

    with tqdm.auto.tqdm(range(n_iter), desc="Training") as pbar:
        for _ in pbar:
            idx = jax.random.randint(key, (1,), n_warmup, series.size - n_train - n_test)[0]

            state = smoother.init(series[idx - n_warmup:idx])
            train, test = series[idx:idx + n_train], series[idx + n_train:idx + n_train + n_test]

            crit, decay, state, opt_state = step(decay, train, test, state, opt_state)
            pbar.set_postfix_str(f"loss = {crit:.2e}")
            criteria.append(crit)
            decays.append(decay)

    return jax.nn.sigmoid(decays[-1]), (criteria, decays)

def create_smoother(smoothe_over_signal: float, smoothe_over_gradient: float,
                    smoothe_over_variance: float) -> SmoothingTransformation:
    """Create a new Holt-Winters transformation.

    Args:
        smoothe_over_signal : Number of periods used to smoothe over the signal.
        smoothe_over_gradient : Number of periods used to smoothe over the 
            gradient.
        smoothe_over_variance : Number of periods used to smoothe over the variance.

    Returns:
        A new Holt-Winters transformation.
    """
    return robust_holt_winters(lambda1=smoothe_over_signal**-1,
                               lambda2=smoothe_over_gradient**-1,
                               lambda_sigma=smoothe_over_variance**-1)

def initialise(smoother: SmoothingTransformation, train: Series) -> HoltWintersState:
    """Calculate an initial smoothing state based on a preliminary series (train)
    of signal values.

    Args:
        smoother : Holt-Winters smoothing transformation.
        train : A series of signal values used to calculate initial smoothing
            state.

    Returns:
        Initial smoothing state.
    """
    return smoother.init(train)

SmoothingStep = Callable[[Element, HoltWintersState], Tuple[Element, HoltWintersState]]

def smoothing_step(smoother: SmoothingTransformation) -> SmoothingStep:
    """Returns a smoothing step function used in real-time smoothing. The function
    takes a new signal point and a current smoothing state and returns a smoothed
    signal and an updated state.

    Args:
        smoother : Holt-Winters smoothing transformation.

    Returns:
        Smoothing step function.
    """
    @jax.jit
    def step(new: Element, state: HoltWintersState) -> Tuple[Element, HoltWintersState]:
        return smoother.update(new, state)

    return step

def main(out_path: str, log_path: str, sensor: int=1, period: int=10, init_value: float=10.0,
         n_iter: int=2000, sizes: Tuple[int, int, int] = (100, 200, 100),
         learning_rate: float=3e-3, seed: int=0):
    """Perform training of the Holt-Winters smoothing algorithm and return optimal
    decay rates for the signal, gradient, and standard deviation.

    Args:
        series : Series of signal values used in training.
        init_decay : Initial value of decay rates.
        sizes : Specifies the size of a sequence used for warm-up, training, and
            testing stages in a single iteration.
        n_iter : Number of training iterations.
        learning_rate : Learning rate for gradient optimiser of decay rates.
        seed : Seed used for random number generator.

    Returns:
        Optimal decay rates for signal, gradient, and standard deviation. Also,
        it returns the history of loss values and decay rates at each iteration.
    """
    keys = {1: {"signal": 'Mag3 [QCM,S1 signal]', "background": 'Mag3 [QCM,S1 background]'},
            2: {"signal": 'Mag3 [QCM,S1 signal]', "background": 'Mag3 [QCM,S1 background]'}}

    print(f"Reading '{keys[sensor]['signal']}' and '{keys[sensor]['background']}' from {log_path}")

    theta_raw = jnp.unwrap(read_csv(log_path, columns['theta']))
    signal = read_csv(log_path, keys[sensor]["signal"])
    background = read_csv(log_path, keys[sensor]["background"])

    print("Preparing data...")

    _, signal = extract_rotations(theta_raw, signal)
    _, background = extract_rotations(theta_raw, background)

    series = (signal - background)[::period]

    smoother = inject_hyperparams(robust_holt_winters)(lambda1=init_value**-1,
                                                       lambda2=init_value**-1,
                                                       lambda_sigma=init_value**-1)
    schedule = ox.cosine_onecycle_schedule(n_iter, learning_rate)
    opt = ox.inject_hyperparams(ox.adam)(learning_rate=schedule)

    decay = jnp.full(3, -jnp.log(init_value - 1))
    opt_state = opt.init(decay)

    n_warmup, n_train, n_test = sizes

    step = meta_step(smoother, opt)
    key = jax.random.PRNGKey(seed)
    criteria, decays = [], []

    with tqdm.trange(n_iter, desc="Training") as pbar:
        for i in pbar:
            if i % 100 == 0:
                new_values = jax.nn.sigmoid(decay)**-1
                pbar.write(f"sum_over_signal = {new_values[0]:<7.3f}, " +
                           f"sum_over_gradient = {new_values[1]:<7.3f}, " +
                           f"sum_over_variance = {new_values[2]:<7.3f}")
            idx = jax.random.randint(key, (1,), n_warmup, series.size - n_train - n_test)[0]

            state = smoother.init(series[idx - n_warmup:idx])
            train, test = series[idx:idx + n_train], series[idx + n_train:idx + n_train + n_test]

            crit, decay, state, opt_state = step(decay, train, test, state, opt_state)
            pbar.set_postfix_str(f"loss = {crit:.2e}")

            criteria.append(crit)
            decays.append(decay)

    new_values = jax.nn.sigmoid(decay)**-1
    print(f"Optimal parameters: sum_over_signal = {new_values[0]:<7.3f}, " +
          f"sum_over_gradient = {new_values[1]:<7.3f}, " +
          f"sum_over_variance = {new_values[2]:<7.3f}")

    with h5py.File(out_path, 'w') as out_file:
        values = jax.nn.sigmoid(jnp.stack(decays))**-1
        out_file.create_dataset('data/criteria', data=criteria)
        out_file.create_dataset('data/sum_over_signal', data=values[:, 0])
        out_file.create_dataset('data/sum_over_gradient', data=values[:, 1])
        out_file.create_dataset('data/sum_over_variance', data=values[:, 2])

    print(f"Results saved to {out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Performing training of Robust Holt-Winters smoothing transformation")
    parser.add_argument('out_path', type=str, help="Path to a HDF5 file where the results of the training will be saved")
    parser.add_argument('log_path', type=str, help="Path to the log file with the QC measurements read-outs")
    parser.add_argument('--sensor', '-s', type=int, choices=[1, 2], default=1,
                        help="Choose between 'Sensor1' (1) and 'Sensor2' (2)")
    parser.add_argument('--period', '-p', type=int, default=10, help="Choose over how many layers the QC measurements"\
                        "should be summed over (Sum over)")
    parser.add_argument('--init', '-i', type=float, default=10.0, help="The initial smoothe over value [in periods]")
    parser.add_argument('--n_iter', '-n', type=int, default=2000, help="Number of training iterations")
    parser.add_argument('--sizes', type=int, nargs=3, default=(100, 200, 100),
                        help="Specify the size of series used for warm-up, smoothing, and testing stages [in periods]")
    parser.add_argument('--lrate', '-l', type=float, default=3e-3, help="Learning rate of gradient optimiser")

    args = parser.parse_args()

    main(out_path=args.out_path, log_path=args.log_path, sensor=args.sensor, period=args.period, init_value=args.init,
         n_iter=args.n_iter, sizes=args.sizes, learning_rate=args.lrate, seed=666420)