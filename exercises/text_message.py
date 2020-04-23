import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
# @#TODO remember about listing resources
figsize(12.5, 5)


def main():
    count_data = np.loadtxt('../data/txtdata.csv')
    n_count_data = len(count_data)

    with pm.Model() as model:
        # a good rule of thumb is to set the exponential parameter equal to the inverse of the average of count data
        alpha = 1.0 / count_data.mean()

        lambda_1 = pm.Exponential('lambda_1', alpha)
        lambda_2 = pm.Exponential('lambda_2', alpha)

        # because of the noisiness of the data, it's difficult to pick out a priory when τ might have occurred
        # instead we can assign a uniform prior belief to every possible day
        tau = pm.DiscreteUniform('tau', lower=0, upper=n_count_data - 1)

    with model:
        idx = np.arange(n_count_data)

        # the switch function assigns λ_1 or λ_2 as the value of λ_, depending on what side of τ we are on.
        lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)

    with model:
        # combine our data, count_data, with our proposed data generation scheme,
        # given by the variable λ_, through the observed keyword.
        observation = pm.Poisson('obs', lambda_, observed=count_data)

    with model:
        step = pm.Metropolis()
        trace = pm.sample(10_000, tune=5_000, step=step)

    lambda_1_samples = trace['lambda_1']
    lambda_2_samples = trace['lambda_2']
    tau_samples = trace['tau']

    # τ_samples, λ_1_samples, λ_2_samples contain N samples from the corresponding posterior distribution
    N = tau_samples.shape[0]
    expected_texts_per_day = np.zeros(n_count_data)

    for day in range(0, n_count_data):
        # ix is a bool index of all τ samples corresponding to the switch point occurring prior to value of `day`
        ix = day < tau_samples

        # each posterior sample corresponds to a value of τ. for each day, that value of τ indicates whether
        # we're "before" (in the λ_1 "regime") or "after" (in the λ_2 "regime") the switchpoint.
        # by taking the posterior sample of λ_1/2 accordingly, we can average over all samples
        # to get and expected value for λ on that day.
        expected_texts_per_day[day] = (lambda_1_samples[ix].sum() + lambda_2_samples[~ix].sum()) / N

    # DETERMINING STATISTICALLY IF THE TWO λs ARE INDEED DIFFERENT?
    # what is the probability that the values differ by at least 1? 2? 5? 10?
    for d in [1, 2, 5, 10]:
        v = (abs(lambda_1_samples - lambda_2_samples) >= d).mean()
        print(f'What is the probability the difference is larger than {d}? {v:.2f}')

    print()

    # what is the expected percentage increase in text-message rates?
    relative_increase_samples = (lambda_2_samples - lambda_1_samples) / lambda_1_samples
    print(f'percentage increase in text-message rates: {relative_increase_samples.mean() * 100:.0f}%')

    print()

    # what is the mean of 𝜆_1 given that we know 𝜏 is less than 45.
    # that is, suppose we have been given new information that the change in behaviour occurred prior to day 45.
    ix = tau_samples < 45
    print(f'mean of 𝜆_1 given that we know 𝜏 is less than 45: {lambda_1_samples[ix].mean():.2f}')

    plt.plot(range(n_count_data), expected_texts_per_day, lw=4, color="#E24A33",
             label="expected number of text-messages received")
    plt.xlim(0, n_count_data)
    plt.xlabel("Day")
    plt.ylabel("Expected # text-messages")
    plt.title("Expected number of text-messages received")
    plt.ylim(0, 60)
    plt.bar(np.arange(len(count_data)), count_data, color="#348ABD", alpha=0.65, label="observed texts per day")
    plt.legend(loc="upper left")
    plt.savefig("../pictures/expected_number_of_text-messages_received.png")
    plt.show()


if __name__ == '__main__':
    main()
