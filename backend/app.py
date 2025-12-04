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


sigma_0_10 = 2.1233  

cgpa['low'] = fuzz.gaussmf(cgpa.universe, 0, sigma_0_10)
cgpa['medium'] = fuzz.gaussmf(cgpa.universe, 5, sigma_0_10)
cgpa['high'] = fuzz.gaussmf(cgpa.universe, 10, sigma_0_10)

demand['low'] = fuzz.gaussmf(demand.universe, 0, sigma_0_10)
demand['medium'] = fuzz.gaussmf(demand.universe, 5, sigma_0_10)
demand['high'] = fuzz.gaussmf(demand.universe, 10, sigma_0_10)


sigma_0_3 = 0.6370  

po['low'] = fuzz.gaussmf(po.universe, 0, sigma_0_3)
po['medium'] = fuzz.gaussmf(po.universe, 1.5, sigma_0_3)
po['high'] = fuzz.gaussmf(po.universe, 3, sigma_0_3)

tech_contribution['low'] = fuzz.gaussmf(tech_contribution.universe, 0, sigma_0_3)
tech_contribution['medium'] = fuzz.gaussmf(tech_contribution.universe, 1.5, sigma_0_3)
tech_contribution['high'] = fuzz.gaussmf(tech_contribution.universe, 3, sigma_0_3)

project_outcome['low'] = fuzz.gaussmf(project_outcome.universe, 0, sigma_0_3)
project_outcome['medium'] = fuzz.gaussmf(project_outcome.universe, 1.5, sigma_0_3)
project_outcome['high'] = fuzz.gaussmf(project_outcome.universe, 3, sigma_0_3)

feedback['poor'] = fuzz.gaussmf(feedback.universe, 0, sigma_0_3)
feedback['average'] = fuzz.gaussmf(feedback.universe, 1.5, sigma_0_3)
feedback['excellent'] = fuzz.gaussmf(feedback.universe, 3, sigma_0_3)


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
