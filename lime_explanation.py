import lime
import random
import lime.lime_tabular
import numpy as np
from sklearn.ensemble import RandomForestRegressor

EXPLORATION_PROBABILITY = 0.1
HOURS = 24

# Generate probabilities
def high(hours):
  return np.linspace(0.90, 0.99, hours)

def med(hours):
  return np.linspace(0.70, 0.89, hours)

def low(hours):
  return np.linspace(0.0, 0.69, hours)

# Calculate probability and cost
def calculate_probability_and_cost(data):
  # Fetch values from data
  pVitalParamsPicked = data.get("pVitalParamsPicked", 0)
  pChangeResult = data.get("pChangeResult", 0)
  pDrug = data.get("pDrug", 0)
  pAnalysis = data.get("pAnalysis", 0)
  pAlarm = data.get("pAlarm", 0)

  # Calculate costs using modifier
  cDrug = pDrug * 3
  cAlarm = pAlarm * 5
  cAnalysis = pAnalysis * 1

  # Calculate reliability
  reliability = (pVitalParamsPicked * pChangeResult * pDrug * pAnalysis
                  - pVitalParamsPicked * pChangeResult * pAlarm * pAnalysis
                  + pVitalParamsPicked * pAlarm * pAnalysis
                  - pVitalParamsPicked * pAlarm
                  + pAlarm)

  # Calculate cost
  cost = (pVitalParamsPicked * pChangeResult * cDrug * pAnalysis
          - pVitalParamsPicked * pChangeResult * cAlarm * pAnalysis
          + pVitalParamsPicked * cAlarm * pAnalysis
          + pVitalParamsPicked * cAnalysis
          - pVitalParamsPicked * cAlarm
          + cAlarm)

  return reliability, cost

# Multi-armed bandit using epsilon-greedy strategy
def epsilon_greedy_bandit(choices, hours, epsilon=EXPLORATION_PROBABILITY):
  n_arms = len(choices)
  rewards = np.zeros(n_arms)
  counts = np.zeros(n_arms)

  # Best features
  BEST_CHOICES = []
  # Reliability scores
  y_reliability = []
  # Cost scores
  y_cost = []

  for hour in range(hours):
    if random.random() < epsilon:
      # Choose a random arm
      arm = random.choice(range(n_arms))
    else:
      # Choose the best arm so far
      avg_rewards = rewards / (counts + 1e-10)
      arm = np.argmax(avg_rewards)

    choice = choices[arm]
    data = {
      "pVitalParamsPicked": 1,
      "pChangeResult": 1,
      "pDrug": choice["pDrug"][hour],
      "pAnalysis": choice["pAnalysis"][hour],
      "pAlarm": choice["pAlarm"][hour],
    }

    # Calculate reliability and cost
    reliability, cost = calculate_probability_and_cost(data)

    # Reward can be based on reliability minus cost
    reward = reliability + cost

    # Update rewards and counts for the selected arm
    rewards[arm] += reward
    counts[arm] += 1

    # Store the data for training
    BEST_CHOICES.append([choice["pDrug"][hour], choice["pAnalysis"][hour], choice["pAlarm"][hour]])
    y_reliability.append(reliability)
    y_cost.append(cost)

    print(f"Hour {hour}: Best Arm {arm} with Avg Reward {avg_rewards[arm]:.2f}")

  return np.array(BEST_CHOICES), np.array(y_reliability), np.array(y_cost)

# Generate probability arrays
p1Drug = high(HOURS)
p2Drug = low(HOURS)
p3Drug = med(HOURS)

p1Analysis = high(HOURS)
p2Analysis = low(HOURS)
p3Analysis = low(HOURS)

p1Alarm = high(HOURS)
p2Alarm = high(HOURS)
p3Alarm = high(HOURS)

# Define choices
choices = [
  {"pDrug": p3Drug, "pAnalysis": p1Analysis, "pAlarm": p1Alarm},
  {"pDrug": p3Drug, "pAnalysis": p2Analysis, "pAlarm": p2Alarm},
  {"pDrug": p1Drug, "pAnalysis": p1Analysis, "pAlarm": p1Alarm},
  {"pDrug": p1Drug, "pAnalysis": p3Analysis, "pAlarm": p3Alarm},
]

# Calculate using epsilon-greedy multi-armed bandit algorithm
BEST_CHOICES, y_reliability, y_cost = epsilon_greedy_bandit(choices, HOURS)

# Train a Random Forest Regressor on the selected arm's data
clf_reliability = RandomForestRegressor()
clf_cost = RandomForestRegressor()

clf_reliability.fit(BEST_CHOICES, y_reliability)
clf_cost.fit(BEST_CHOICES, y_cost)

# Use LIME to explain the decisions
explainer = lime.lime_tabular.LimeTabularExplainer(BEST_CHOICES, mode='regression', feature_names=["pDrug", "pAnalysis", "pAlarm"])

# Get explanations for the reliability and cost
exp_reliability = explainer.explain_instance(BEST_CHOICES[0], clf_reliability.predict, num_features=3)
exp_cost = explainer.explain_instance(BEST_CHOICES[0], clf_cost.predict, num_features=3)

# Save explanations to HTML
exp_reliability.save_to_file('lime_explanation_reliability.html')
exp_cost.save_to_file('lime_explanation_cost.html')

# Output for the first hour (for demonstration purposes)
print(f"Hour 0 Reliability: {y_reliability[0]}")
print(f"Hour 0 Cost: {y_cost[0]:.2f}")
