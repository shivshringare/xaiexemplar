import math
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

HOURS = 24
PROFILE_COUNT = 27
RELIABILITY_THRESHOLD = 0.5
COST_INCREASE_LIMIT = 5

# Define concrete services with names
alarm_services = [
  {"name": "Alarm1", "reliability": 0.7, "cost": 5},
  {"name": "Alarm2", "reliability": 0.6, "cost": 4},
  {"name": "Alarm3", "reliability": 0.8, "cost": 6},
  {"name": "Alarm4", "reliability": 0.4, "cost": 3},
]

analysis_services = [
  {"name": "Analysis1", "reliability": 0.9, "cost": 3},
  {"name": "Analysis2", "reliability": 0.85, "cost": 2},
  {"name": "Analysis3", "reliability": 0.8, "cost": 4},
  {"name": "Analysis4", "reliability": 0.5, "cost": 2},
]

drug_services = [
  {"name": "Drug1", "reliability": 0.75, "cost": 7},
  {"name": "Drug2", "reliability": 0.8, "cost": 6},
  {"name": "Drug3", "reliability": 0.65, "cost": 5},
  {"name": "Drug4", "reliability": 0.4, "cost": 3},
]

# Probability generators for profile attributes
def high(hours):
  return np.linspace(0.90, 0.99, hours)

def med(hours):
  return np.linspace(0.60, 0.89, hours)

def low(hours):
  return np.linspace(0.0, 0.59, hours)

# Calculate reliability and cost based on selected service combination
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

# Multi-armed bandit using Upper Confidence Bound (UCB) for service selection
def ucb_bandit_service_selection(services, counts, rewards, total_count):
  n_services = len(services)
  ucb_values = np.zeros(n_services)

  # Calculate UCB for each provider
  for i in range(n_services):
    if counts[i] == 0:
      ucb_values[i] = float('inf')  # Ensure exploration
    else:
      avg_reward = rewards[i] / counts[i]
      confidence_interval = math.sqrt((2 * math.log(total_count + 1)) / counts[i])
      ucb_values[i] = avg_reward + confidence_interval

  # Select the provider with the highest UCB value
  selected_provider = np.argmax(ucb_values)

  return selected_provider

# Multi-armed bandit with Upper Confidence Bound (UCB)
def tas_workflow(profiles):
  # Initialize service-specific counters and rewards
  drug_counts = np.zeros(len(drug_services))
  drug_rewards = np.zeros(len(drug_services))

  analysis_counts = np.zeros(len(analysis_services))
  analysis_rewards = np.zeros(len(analysis_services))

  alarm_counts = np.zeros(len(alarm_services))
  alarm_rewards = np.zeros(len(alarm_services))

  total_count = 0

  selected_profiles = []
  y_reliability = []
  y_cost = []
  selected_service_names = []

  for hour in range(HOURS):
    # Randomly select a user profile for this hour
    profile = profiles[np.random.choice(len(profiles))]
    profile['pVitalParamsPicked'] = 0.9 if profile['type'] == 'young' else 0.4
    selected_profiles.append(profile)

    # Select concrete services for Drug, Analysis, and Alarm using UCB
    drug_index = ucb_bandit_service_selection(drug_services, drug_counts, drug_rewards, total_count)
    analysis_index = ucb_bandit_service_selection(analysis_services, analysis_counts, analysis_rewards, total_count)
    alarm_index = ucb_bandit_service_selection(alarm_services, alarm_counts, alarm_rewards, total_count)

    selected_services = {
      "drug": drug_services[drug_index],
      "analysis": analysis_services[analysis_index],
      "alarm": alarm_services[alarm_index]
    }

    # Prepare data for reliability and cost calculation
    data = {
      "pVitalParamsPicked": profile['pVitalParamsPicked'],
      "pChangeResult": 1,
      "pDrug": selected_services["drug"]["reliability"],
      "pAnalysis": selected_services["analysis"]["reliability"],
      "pAlarm": selected_services["alarm"]["reliability"]
    }

    # Calculate reliability and cost
    reliability, cost = calculate_reliability_and_cost(data)

    # Service degradation handling
    if reliability < RELIABILITY_THRESHOLD:
      print(f"Hour {hour}: Service degradation detected. Considering alternative services.")
      # Check all possible combinations for better alternatives within cost limits
      for d in range(len(drug_services)):
        for a in range(len(analysis_services)):
          for al in range(len(alarm_services)):
            alt_services = {
              "drug": drug_services[d],
              "analysis": analysis_services[a],
              "alarm": alarm_services[al]
            }
            alt_data = {
              "pVitalParamsPicked": profile['pVitalParamsPicked'],
              "pChangeResult": 1,
              "pDrug": alt_services["drug"]["reliability"],
              "pAnalysis": alt_services["analysis"]["reliability"],
              "pAlarm": alt_services["alarm"]["reliability"]
            }
            alt_reliability, alt_cost = calculate_reliability_and_cost(alt_data)

            if alt_reliability > reliability and (alt_cost - cost) <= COST_INCREASE_LIMIT:
              print(f"Switching to more reliable service.")
              selected_services = alt_services
              reliability = alt_reliability
              cost = alt_cost
              break

    # Reward is based on reliability minus cost
    reward = reliability - cost

    # Update rewards and counts for the selected services
    drug_rewards[drug_index] += reward
    drug_counts[drug_index] += 1

    analysis_rewards[analysis_index] += reward
    analysis_counts[analysis_index] += 1

    alarm_rewards[alarm_index] += reward
    alarm_counts[alarm_index] += 1

    total_count += 1

    # Store data for training and visualization
    y_reliability.append(reliability)
    y_cost.append(cost)
    selected_service_names.append(f"{selected_services['drug']['name']}, "
                                  f"{selected_services['analysis']['name']}, "
                                  f"{selected_services['alarm']['name']}")

    print(f"Hour {hour}: Selected Services: {selected_services['drug']['name']}, "
          f"{selected_services['analysis']['name']}, {selected_services['alarm']['name']}")

  return np.array(selected_profiles), np.array(y_reliability), np.array(y_cost), selected_service_names

# Generate profiles for elderly and young
young_profiles = [
  {
    "type": "young",
    "pDrug": high(HOURS),
    "pAnalysis": med(HOURS),
    "pAlarm": low(HOURS),
  }
  for _ in range(PROFILE_COUNT)
]
elderly_profiles = [
  {
    "type": "elderly",
    "pDrug": med(HOURS),
    "pAnalysis": high(HOURS),
    "pAlarm": high(HOURS),
  }
  for _ in range(PROFILE_COUNT)
]

# run MAB on the user profiles
selected_profiles, y_reliability, y_cost, selected_service_names = tas_workflow(young_profiles)

detailed_service_data = {
  "Hour": [],
  "Profile Type": [],
  "pVitalParamsPicked": [],
  "Selected Drug": [],
  "Drug Reliability": [],
  "Drug Cost": [],
  "Selected Analysis": [],
  "Analysis Reliability": [],
  "Analysis Cost": [],
  "Selected Alarm": [],
  "Alarm Reliability": [],
  "Alarm Cost": [],
  "Total Reliability": [],
  "Total Cost": [],
}

# Populate the data for each hour
for hour in range(HOURS):
  selected_services = selected_service_names[hour].split(", ")
  selected_profile = selected_profiles[hour]

  # Extract the selected services' details
  drug = next(d for d in drug_services if d["name"] == selected_services[0])
  analysis = next(a for a in analysis_services if a["name"] == selected_services[1])
  alarm = next(al for al in alarm_services if al["name"] == selected_services[2])

  # Calculate total reliability and cost for the hour
  total_reliability = (drug["reliability"] * analysis["reliability"] * alarm["reliability"])
  total_cost = drug["cost"] + analysis["cost"] + alarm["cost"]

  # Add data to the dictionary
  detailed_service_data["Hour"].append(hour)
  detailed_service_data["Profile Type"].append(selected_profile["type"])
  detailed_service_data["pVitalParamsPicked"].append(selected_profile['pVitalParamsPicked'])
  detailed_service_data["Selected Drug"].append(drug["name"])
  detailed_service_data["Drug Reliability"].append(drug["reliability"])
  detailed_service_data["Drug Cost"].append(drug["cost"])
  detailed_service_data["Selected Analysis"].append(analysis["name"])
  detailed_service_data["Analysis Reliability"].append(analysis["reliability"])
  detailed_service_data["Analysis Cost"].append(analysis["cost"])
  detailed_service_data["Selected Alarm"].append(alarm["name"])
  detailed_service_data["Alarm Reliability"].append(alarm["reliability"])
  detailed_service_data["Alarm Cost"].append(alarm["cost"])
  detailed_service_data["Total Reliability"].append(total_reliability)
  detailed_service_data["Total Cost"].append(total_cost)

# Convert to DataFrame
service_data_df = pd.DataFrame(detailed_service_data)

# Export to CSV
csv_filename = 'tas_results.csv'
service_data_df.to_csv(csv_filename, index=False)

print(f"Data has been exported to {csv_filename}")

features = pd.DataFrame(
  {
    "Drug Reliability": detailed_service_data["Drug Reliability"],
    "Drug Cost": detailed_service_data["Drug Cost"],
    "Analysis Reliability": detailed_service_data["Analysis Reliability"],
    "Analysis Cost": detailed_service_data["Analysis Cost"],
    "Alarm Reliability": detailed_service_data["Alarm Reliability"],
    "Alarm Cost": detailed_service_data["Alarm Cost"],
  }
).values

# Train Random Forest models for total reliability and total cost
rf_reliability = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cost = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the models
rf_reliability.fit(features, detailed_service_data["Total Reliability"])
rf_cost.fit(features, detailed_service_data["Total Cost"])

# Use LIME to explain predictions with continuous features
explainer = LimeTabularExplainer(
  features,
  mode="regression",
  feature_names=[
    "Drug Reliability",
    "Drug Cost",
    "Analysis Reliability",
    "Analysis Cost",
    "Alarm Reliability",
    "Alarm Cost",
  ],
  discretize_continuous=True,
)

# Generate explanations for each hour and combine into a single HTML file
with open("tas_lime_explanations_all_hours.html", "w") as f:
  f.write(
    "<html><head><title>LIME Explanations for Reliability and Cost</title></head><body>"
  )
  f.write("<h1>LIME Explanations for Reliability and Cost Over All Hours</h1>")

  for hour in range(HOURS):
    # Explain the prediction for total reliability
    exp_reliability = explainer.explain_instance(
      features[hour], rf_reliability.predict, num_features=6
    )

    # Explain the prediction for total cost
    exp_cost = explainer.explain_instance(
      features[hour], rf_cost.predict, num_features=6
    )

    # Add explanations to the HTML file
    f.write(f"<h2>Hour {hour}</h2>")
    f.write("<h3>Reliability Explanation:</h3>")
    f.write(exp_reliability.as_html())
    f.write("<h3>Cost Explanation:</h3>")
    f.write(exp_cost.as_html())
    f.write("<hr>")

  f.write("</body></html>")
