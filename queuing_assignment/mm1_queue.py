from random_generators import exponential


def simulate_mm1(lam, mu, n_events=200_000):
    """
    Simulate M/M/1 queue using event-driven simulation.

    Parameters
    ----------
    lam      : float — arrival rate (lambda)
    mu       : float — service rate (mu)
    n_events : int   — number of arrival events to simulate

    Returns
    -------
    pn : dict — {n: fraction of time spent with n customers in system}
    """
    mean_interarrival = 1.0 / lam
    mean_service      = 1.0 / mu

    clock           = 0.0       # simulation clock
    n_in_system     = 0         # current number of customers in system
    next_arrival    = exponential(mean=mean_interarrival, size=1)[0]
    next_departure  = float('inf')  # no departure until someone arrives

    time_in_state   = {}        # {n: total time spent in state n}
    arrivals_done   = 0

    while arrivals_done < n_events:
        # ── Next event: arrival or departure ────────────────────────────────
        if next_arrival <= next_departure:
            dt    = next_arrival - clock
            clock = next_arrival

            # accumulate time in current state
            time_in_state[n_in_system] = time_in_state.get(n_in_system, 0.0) + dt

            # process arrival
            n_in_system  += 1
            arrivals_done += 1

            # schedule next arrival
            next_arrival  = clock + exponential(mean=mean_interarrival, size=1)[0]

            # if server was idle, start service immediately
            if n_in_system == 1:
                next_departure = clock + exponential(mean=mean_service, size=1)[0]

        else:
            dt    = next_departure - clock
            clock = next_departure

            # accumulate time in current state
            time_in_state[n_in_system] = time_in_state.get(n_in_system, 0.0) + dt

            # process departure
            n_in_system -= 1

            if n_in_system == 0:
                next_departure = float('inf')   # server goes idle
            else:
                next_departure = clock + exponential(mean=mean_service, size=1)[0]

    # ── Compute stationary probabilities ────────────────────────────────────
    total_time = sum(time_in_state.values())
    pn = {n: t / total_time for n, t in time_in_state.items()}
    return pn
