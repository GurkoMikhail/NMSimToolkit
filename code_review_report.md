## Mathematical review of the implementation

### Source decay models (`core/source/sources.py`):
1.  **Activity formula:** $A(t) = A_0 \cdot 2^{-t / T_{1/2}}$. Correctly implemented in `get_activity` as `self.initial_activity * (2.0 ** (-t / self.half_life))`.
2.  **Integral of activity:** $N(\Delta t) = \int_{t_1}^{t_2} A_0 \cdot 2^{-t / T_{1/2}} dt$.
    Substitute $\lambda = \ln(2) / T_{1/2}$ and $A(t) = A_0 \cdot e^{-\lambda t}$.
    $\int_{t_1}^{t_2} A(t_1) \cdot e^{-\lambda (t - t_1)} dt = \frac{A(t_1)}{\lambda} \left( 1 - e^{-\lambda (t_2 - t_1)} \right)$.
    This correctly matches the implementation in `get_expected_particles`: `(self.get_activity(t1) / lambd) * (1 - np.exp(-lambd * dt))`.
3.  **Linear limit:** If $\lambda \cdot \Delta t \ll 1$, $\frac{A(t_1)}{\lambda} \left( 1 - e^{-\lambda \Delta t} \right) \approx \frac{A(t_1)}{\lambda} \cdot \lambda \Delta t = A(t_1) \cdot \Delta t$.
    This also matches the edge case in `get_expected_particles` where it returns `self.get_activity(t1) * dt` when $\lambda \cdot dt < 10^{-6}$.
4.  **Inverse Transform Sampling:** CDF of the decay process on interval $[0, \Delta t]$ is $F(\tau) = \frac{1 - e^{-\lambda \tau}}{1 - e^{-\lambda \Delta t}}$.
    Setting $U = F(\tau)$ and solving for $\tau$:
    $U \cdot (1 - e^{-\lambda \Delta t}) = 1 - e^{-\lambda \tau}$
    $e^{-\lambda \tau} = 1 - U \cdot (1 - e^{-\lambda \Delta t})$
    $\tau = -\frac{1}{\lambda} \ln(1 - U \cdot (1 - e^{-\lambda \Delta t}))$
    This correctly matches the implementation in `inject`: `emission_time = t1 - (1.0 / lambd) * np.log(1.0 - u * (1.0 - np.exp(-lambd * dt)))`.
    The linear limit for emission time gives `t1 + u * dt` which is also correct.

### Newton-Raphson global timer (`core/transport/simulation_managers.py`):
We want to find $dt$ such that $\sum N_i(dt) = N$, where $N_i(dt)$ is the expected number of particles from source $i$ over interval $[t_{global}, t_{global} + dt]$.
Let $f(dt) = \sum N_i(dt) - N$. We want $f(dt) = 0$.
The derivative of $f(dt)$ with respect to $dt$ is:
$f'(dt) = \frac{d}{d(dt)} \sum \int_{0}^{dt} A_i(t_{global} + \tau) d\tau = \sum A_i(t_{global} + dt)$.
This correctly matches the implementation:
`f(dt) = sum(src.get_expected_particles(global_timer, global_timer + dt)) - num_to_inject`
`df(dt) = sum(src.get_activity(global_timer + dt))`
The Newton-Raphson update step is $dt = dt - \frac{f(dt)}{f'(dt)}$, which is correctly implemented.

### Quota distribution and adjust remainder
`quotas = np.floor(expected).astype(int)`
`remainders = expected - quotas`
`shortfall = num_to_inject - np.sum(quotas)`
The code then distributes the `shortfall` particles by iterating over the sources with the largest `remainders`. This correctly ensures that exactly `num_to_inject` particles are requested across all sources while keeping integer quotas as proportional as possible to expected continuous values.

The implementation appears mathematically and structurally sound and meets all requirements from DOD logic and physics limits.
