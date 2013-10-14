# sBG model of customer retention

A python implementation of the shifted-beta-geometric (sBG) model from Fader and Hardie's "How to Project Customer
Retention" (2006).

## Example

```python
from shifted_beta_geometric import derl, fit, predicted_survival

# measured percentage of cohort that survives over time
example_data = [.869, .743, .653, .593, .551, .517, .491]

# fit our observed data to the sBG model, which returns the parameters alpha and beta
alpha, beta = fit(example_data)

# predict the next 5 time samples:
future = predicted_survival(alpha, beta, len(example_data) + 5)[-5:]

# future = [0.460, 0.436, 0.414, 0.395, 0.378]

# compute the discounted expected residual lifetime (DERL) for the survivors of this cohort at point in time t:
discount = 0.10  # rate at which we discount future revenue to get value in today's terms, e.g. 10%/year
t = len(example_data)
residual_cohort_lifetime = derl(alpha, beta, discount, t)

# residual_cohort_lifetime = 7.530

# if our average revenue per period per customer is a constant v_avg, to get the residual customer lifetime value (CLV)
# of this cohort we simply multiply the residual_cohort_lifetime by v_avg:

v_avg = 10
residual_cohort_clv = residual_cohort_lifetime * v_avg

# thus residual_cohort_clv = $75.30 per customer in this cohort
```

## Requirements
sBG requires `numpy` and `scipy` for fitting and the gauss hypergeometric function.
