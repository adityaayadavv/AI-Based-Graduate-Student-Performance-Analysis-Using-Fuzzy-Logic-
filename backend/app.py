from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)
CORS(app)

# Define input variables
cgpa = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'cgpa')
po = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'po')
tech_contribution = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'tech_contribution')
project_outcome = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'project_outcome')
feedback = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'feedback')

demand = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'demand')

# membership functions
cgpa_mu = (2.5, 6.0, 8.5)
cgpa_sigma = (1.3, 0.9, 1.1)

cgpa['low'] = fuzz.gaussmf(cgpa.universe, cgpa_mu[0], cgpa_sigma[0])
cgpa['medium'] = fuzz.gaussmf(cgpa.universe, cgpa_mu[1], cgpa_sigma[1])
cgpa['high'] = fuzz.gaussmf(cgpa.universe, cgpa_mu[2], cgpa_sigma[2])

mu_small = (0.5, 1.5, 2.5)
sigma_small = (0.35, 0.35, 0.35)

po['low'] = fuzz.gaussmf(po.universe, mu_small[0], sigma_small[0])
po['medium'] = fuzz.gaussmf(po.universe, mu_small[1], sigma_small[1])
po['high'] = fuzz.gaussmf(po.universe, mu_small[2], sigma_small[2])

tech_contribution['low'] = fuzz.gaussmf(tech_contribution.universe, mu_small[0], sigma_small[0])
tech_contribution['medium'] = fuzz.gaussmf(tech_contribution.universe, mu_small[1], sigma_small[1])
tech_contribution['high'] = fuzz.gaussmf(tech_contribution.universe, mu_small[2], sigma_small[2])

project_outcome['low'] = fuzz.gaussmf(project_outcome.universe, mu_small[0], sigma_small[0])
project_outcome['medium'] = fuzz.gaussmf(project_outcome.universe, mu_small[1], sigma_small[1])
project_outcome['high'] = fuzz.gaussmf(project_outcome.universe, mu_small[2], sigma_small[2])

feedback['poor'] = fuzz.gaussmf(feedback.universe, mu_small[0], sigma_small[0])
feedback['average'] = fuzz.gaussmf(feedback.universe, mu_small[1], sigma_small[1])
feedback['excellent'] = fuzz.gaussmf(feedback.universe, mu_small[2], sigma_small[2])

# Demand gaussian params
demand_mu = (3.0, 6.5, 9.0)
demand_sigma = (1.1, 0.9, 1.0)

demand['low'] = fuzz.gaussmf(demand.universe, demand_mu[0], demand_sigma[0])
demand['medium'] = fuzz.gaussmf(demand.universe, demand_mu[1], demand_sigma[1])
demand['high'] = fuzz.gaussmf(demand.universe, demand_mu[2], demand_sigma[2])



rules = [
    # High demand (8)
    ctrl.Rule(cgpa['high'] & po['high'] & tech_contribution['high'] & project_outcome['high'] & feedback['excellent'], demand['high']),
    ctrl.Rule(cgpa['high'] & po['high'] & tech_contribution['high'] & project_outcome['medium'] & feedback['excellent'], demand['high']),
    ctrl.Rule(cgpa['high'] & po['high'] & tech_contribution['medium'] & project_outcome['high'] & feedback['excellent'], demand['high']),
    ctrl.Rule(cgpa['high'] & po['medium'] & tech_contribution['high'] & project_outcome['high'] & feedback['excellent'], demand['high']),
    ctrl.Rule(cgpa['medium'] & po['high'] & tech_contribution['high'] & project_outcome['high'] & feedback['excellent'], demand['high']),
    ctrl.Rule(cgpa['high'] & po['high'] & tech_contribution['high'] & project_outcome['high'] & feedback['average'], demand['high']),
    ctrl.Rule(cgpa['high'] & po['high'] & tech_contribution['high'] & project_outcome['medium'] & feedback['average'], demand['high']),
    ctrl.Rule(cgpa['high'] & po['high'] & tech_contribution['medium'] & project_outcome['high'] & feedback['average'], demand['high']),

    # Medium demand (8)
    ctrl.Rule(cgpa['high'] & po['medium'] & tech_contribution['medium'] & project_outcome['medium'] & feedback['average'], demand['medium']),
    ctrl.Rule(cgpa['medium'] & po['high'] & tech_contribution['medium'] & project_outcome['medium'] & feedback['average'], demand['medium']),
    ctrl.Rule(cgpa['medium'] & po['medium'] & tech_contribution['high'] & project_outcome['medium'] & feedback['average'], demand['medium']),
    ctrl.Rule(cgpa['medium'] & po['medium'] & tech_contribution['medium'] & project_outcome['high'] & feedback['average'], demand['medium']),
    ctrl.Rule(cgpa['high'] & po['medium'] & tech_contribution['medium'] & project_outcome['high'] & feedback['average'], demand['medium']),
    ctrl.Rule(cgpa['medium'] & po['high'] & tech_contribution['high'] & project_outcome['medium'] & feedback['average'], demand['medium']),
    ctrl.Rule(cgpa['medium'] & po['medium'] & tech_contribution['medium'] & project_outcome['medium'] & feedback['excellent'], demand['medium']),
    ctrl.Rule(cgpa['high'] & po['medium'] & tech_contribution['medium'] & project_outcome['medium'] & feedback['poor'], demand['medium']),  # (logic kept as in original)

    # Low demand (4)
    ctrl.Rule(cgpa['low'] & po['low'] & tech_contribution['low'] & project_outcome['low'] & feedback['poor'], demand['low']),
    ctrl.Rule(cgpa['low'] & po['medium'] & tech_contribution['low'] & project_outcome['low'] & feedback['poor'], demand['low']),
    ctrl.Rule(cgpa['medium'] & po['low'] & tech_contribution['low'] & project_outcome['low'] & feedback['poor'], demand['low']),
    ctrl.Rule(cgpa['low'] & po['low'] & tech_contribution['medium'] & project_outcome['low'] & feedback['poor'], demand['low']),
]

performance_ctrl = ctrl.ControlSystem(rules)

@app.route('/api/estimate', methods=['POST'])
def estimate():
    data = request.json
    performance = ctrl.ControlSystemSimulation(performance_ctrl)

    performance.input['cgpa'] = float(data['cgpa'])
    performance.input['po'] = float(data['po'])
    performance.input['tech_contribution'] = float(data['tech'])
    performance.input['project_outcome'] = float(data['project'])
    performance.input['feedback'] = float(data['feedback'])

    performance.compute()
    score = performance.output['demand']

    if score >= 8:
        category = "High Tech Demand"
    elif score >= 5:
        category = "Medium Tech Demand"
    else:
        category = "Low Tech Demand"

    return jsonify({"score": round(score, 2), "category": category})

if __name__ == '__main__':
    app.run(debug=True)
