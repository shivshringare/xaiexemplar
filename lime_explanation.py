import math
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

HOURS = 24
PROFILE_COUNT = 27
RELIABILITY_THRESHOLD = 0.5
COST_INCREASE_LIMIT = 5

# Generate probabilities
def high(hours):
  return np.linspace(0.90, 0.99, hours)

def med(hours):
  return np.linspace(0.60, 0.89, hours)

def low(hours):
  return np.linspace(0.0, 0.59, hours)

# Calculate reliability and cost
def calculate_reliability_and_cost(data):
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

# Multi-armed bandit using Upper confidence bound algorithm
def ucb_bandit(profiles):
  n_arms = len(profiles)
  rewards = np.zeros(n_arms)
  counts = np.zeros(n_arms)
  total_count = 0

  # selected profiles
  SELECTED_PROFILES = []
  # Reliability scores
  y_reliability = []
  # Cost scores
  y_cost = []

  for hour in range(HOURS):
    arm = 0
    ucb_values = np.zeros(n_arms)

    for i in range(n_arms):
      if counts[i] == 0:
        # If an arm hasn't been selected yet, select it to ensure exploration
        ucb_values[i] = float('inf')
      else:
        # Calculate the UCB value for the arm
        avg_reward = rewards[i] / counts[i]
        confidence_interval = math.sqrt((2 * math.log(total_count + 1)) / counts[i])
        ucb_values[i] = avg_reward + confidence_interval

    # Select the arm with the highest UCB value
    arm = np.argmax(ucb_values)

    profile = profiles[arm]
    data = {
      "pVitalParamsPicked": 1 if profile['type'] == 'young' else 0.4,
      "pChangeResult": 1,
      "pDrug": profile["pDrug"][hour],
      "pAnalysis": profile["pAnalysis"][hour],
      "pAlarm": profile["pAlarm"][hour],
    }

    # Calculate reliability and cost
    reliability, cost = calculate_reliability_and_cost(data)

    # Service degradation handling
    if reliability < RELIABILITY_THRESHOLD:
      if profile['type'] == 'young' and data['pAlarm'] < 0.3:
        print(f"Hour {hour}: Reliability dropped, but alarm is not sensitive for young profile. No switch.")
      else:
          # Look for a new service with better reliability but not much higher cost
          for j in range(n_arms):
            alternate_service = profiles[j]
            alternate_data = {
              "pVitalParamsPicked": 0.4,
              "pChangeResult": 1,
              "pDrug": alternate_service["pDrug"][hour],
              "pAnalysis": alternate_service["pAnalysis"][hour],
              "pAlarm": alternate_service["pAlarm"][hour],
            }
            alt_reliability, alt_cost = calculate_reliability_and_cost(alternate_data)

            if alt_reliability > reliability and alt_cost - cost <= COST_INCREASE_LIMIT:
              print(f"Hour {hour}: Service degradation detected. Switching to a more reliable service.")
              data = alternate_data
              reliability = alt_reliability
              cost = alt_cost
              break

    # Reward can be based on reliability minus cost
    reward = reliability - cost

    # Update rewards and counts for the selected arm
    rewards[arm] += reward
    counts[arm] += 1
    total_count += 1

    # Store the data for training
    SELECTED_PROFILES.append([profile["pDrug"][hour], profile["pAnalysis"][hour], profile["pAlarm"][hour]])
    y_reliability.append(reliability)
    y_cost.append(cost)

    print(f"Hour {hour}: Best Arm {arm} with Avg Reward {rewards[arm]:.2f}")

  return np.array(SELECTED_PROFILES), np.array(y_reliability), np.array(y_cost)

# Generate user profiles
young_profiles = []
elderly_profiles = []

# Create 27 elderly profiles
for i in range(PROFILE_COUNT):
  elderly_profiles.append({
    'type': 'elderly',
    'pDrug': med(HOURS),
    'pAnalysis': high(HOURS),
    'pAlarm': high(HOURS)      # Elderly have higher sensitivity to alarms
  })

  young_profiles.append({
    'type': 'young',
    'pDrug': high(HOURS),
    'pAnalysis': med(HOURS),
    'pAlarm': low(HOURS + i)   # Young profiles have lower alarm sensitivity
  })


# Calculate using multi-armed bandit algorithm
SELECTED_PROFILES, y_reliability, y_cost = ucb_bandit(young_profiles)

# Train a Random Forest Regressor on the selected arm's data
clf_reliability = RandomForestRegressor()
clf_cost = RandomForestRegressor()

clf_reliability.fit(SELECTED_PROFILES, y_reliability)
clf_cost.fit(SELECTED_PROFILES, y_cost)

# Use LIME to explain the decisions
explainer = LimeTabularExplainer(SELECTED_PROFILES, mode='regression', feature_names=["pDrug", "pAnalysis", "pAlarm"])

# Save explanations to HTML
# Combine all explanations to a single output file
with open('lime_explanations_all_hours.html', 'w') as f:
  f.write("<html><head><title>LIME Explanations for Cost and Reliability</title></head><body>")
  f.write("<h1>LIME Explanations for Cost and Reliability</h1>")

  for i in range(HOURS):
    # Get LIME explanation for reliability
    exp_reliability = explainer.explain_instance(SELECTED_PROFILES[i], clf_reliability.predict, num_features=3)
    exp_reliability_html = exp_reliability.as_html()

    # Get LIME explanation for cost
    exp_cost = explainer.explain_instance(SELECTED_PROFILES[i], clf_cost.predict, num_features=3)
    exp_cost_html = exp_cost.as_html()

    # Write explanations to the file for each hour
    f.write(f"<h2>Hour {i}</h2>")
    f.write("<h3>Reliability Explanation:</h3>")
    f.write(exp_reliability_html)
    f.write("<h3>Cost Explanation:</h3>")
    f.write(exp_cost_html)
    f.write("<hr>")

  # Close the HTML document
  f.write("</body></html>")

feature_impact_reliability = []
feature_impact_cost = []

for i in range(HOURS):
  exp_reliability = explainer.explain_instance(SELECTED_PROFILES[i], clf_reliability.predict, num_features=3)
  exp_cost = explainer.explain_instance(SELECTED_PROFILES[i], clf_cost.predict, num_features=3)

  # Extract the weights for reliability and cost explanations
  reliability_weights = [weight for (feature, weight) in exp_reliability.as_list()]
  cost_weights = [weight for (feature, weight) in exp_cost.as_list()]

  feature_impact_reliability.append(reliability_weights)
  feature_impact_cost.append(cost_weights)

feature_impact_reliability = np.array(feature_impact_reliability)
feature_impact_cost = np.array(feature_impact_cost)

# Plot Feature impact on cost and reliability over time
plt.figure(figsize=(12, 6))
bar_width = 0.25
r1 = np.arange(HOURS)
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# For reliability
plt.bar(r1, feature_impact_reliability[:, 0], color='#a6cee3', width=bar_width, edgecolor='grey', label='pDrug (Reliability)')
plt.bar(r2, feature_impact_reliability[:, 1], color='#b2df8a', width=bar_width, edgecolor='grey', label='pAnalysis (Reliability)')
plt.bar(r3, feature_impact_reliability[:, 2], color='#fb9a99', width=bar_width, edgecolor='grey', label='pAlarm (Reliability)')

plt.xlabel('Hours', fontsize=12)
plt.ylabel('Feature Impact', fontsize=12)
plt.title('Feature Impact for Reliability Over Time', fontsize=14)
plt.xticks([r + bar_width for r in range(HOURS)], range(HOURS))
plt.legend()
plt.grid(axis='y')
plt.show()

# For cost
plt.figure(figsize=(12, 6))
plt.bar(r1, feature_impact_cost[:, 0], color='#a6cee3', width=bar_width, edgecolor='grey', label='pDrug (Cost)')
plt.bar(r2, feature_impact_cost[:, 1], color='#b2df8a', width=bar_width, edgecolor='grey', label='pAnalysis (Cost)')
plt.bar(r3, feature_impact_cost[:, 2], color='#fb9a99', width=bar_width, edgecolor='grey', label='pAlarm (Cost)')
plt.xlabel('Hours', fontsize=12)
plt.ylabel('Feature Impact', fontsize=12)
plt.title('Feature Impact for Cost Over Time', fontsize=14)
plt.xticks([r + bar_width for r in range(HOURS)], range(HOURS))
plt.legend()
plt.grid(axis='y')
plt.show()
