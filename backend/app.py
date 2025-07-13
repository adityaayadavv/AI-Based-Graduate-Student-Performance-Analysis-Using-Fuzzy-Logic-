
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

cgpa['low'] = fuzz.trimf(cgpa.universe, [0, 0, 5])
cgpa['medium'] = fuzz.trimf(cgpa.universe, [4, 6, 8])
cgpa['high'] = fuzz.trimf(cgpa.universe, [7, 10, 10])

po['low'] = fuzz.trimf(po.universe, [0, 0, 1])
po['medium'] = fuzz.trimf(po.universe, [0.5, 1.5, 2.5])
po['high'] = fuzz.trimf(po.universe, [2, 3, 3])

tech_contribution['low'] = fuzz.trimf(tech_contribution.universe, [0, 0, 1])
tech_contribution['medium'] = fuzz.trimf(tech_contribution.universe, [0.5, 1.5, 2.5])
tech_contribution['high'] = fuzz.trimf(tech_contribution.universe, [2, 3, 3])

project_outcome['low'] = fuzz.trimf(project_outcome.universe, [0, 0, 1])
project_outcome['medium'] = fuzz.trimf(project_outcome.universe, [0.5, 1.5, 2.5])
project_outcome['high'] = fuzz.trimf(project_outcome.universe, [2, 3, 3])

feedback['poor'] = fuzz.trimf(feedback.universe, [0, 0, 1])
feedback['average'] = fuzz.trimf(feedback.universe, [0.5, 1.5, 2.5])
feedback['excellent'] = fuzz.trimf(feedback.universe, [2, 3, 3])

demand['low'] = fuzz.trimf(demand.universe, [0, 0, 5])
demand['medium'] = fuzz.trimf(demand.universe, [4.5, 6.5, 8])
demand['high'] = fuzz.trimf(demand.universe, [7.5, 10, 10])

rules = [
    ctrl.Rule(cgpa['high'] & po['high'] & tech_contribution['high'], demand['high']),
    ctrl.Rule(cgpa['medium'] & po['high'] & tech_contribution['high'], demand['high']),
    ctrl.Rule(cgpa['medium'] & po['medium'] & tech_contribution['medium'], demand['medium']),
    ctrl.Rule(cgpa['low'] & po['low'] & tech_contribution['low'], demand['low']),
    ctrl.Rule(cgpa['low'] & project_outcome['low'], demand['low']),
    ctrl.Rule(project_outcome['high'] & tech_contribution['high'], demand['high']),
    ctrl.Rule(project_outcome['medium'] & feedback['excellent'], demand['medium']),
    ctrl.Rule(cgpa['high'] & feedback['excellent'], demand['high']),
    ctrl.Rule(cgpa['high'] & po['medium'] & tech_contribution['medium'], demand['high']),
    ctrl.Rule(cgpa['medium'] & po['medium'] & project_outcome['high'], demand['medium']),
    ctrl.Rule(cgpa['low'] & feedback['poor'], demand['low']),
    ctrl.Rule(tech_contribution['low'] & feedback['poor'], demand['low']),
    ctrl.Rule(cgpa['medium'] & project_outcome['medium'] & feedback['average'], demand['medium']),
    ctrl.Rule(po['low'] & tech_contribution['low'], demand['low']),
    ctrl.Rule(cgpa['high'] & po['high'], demand['high']),
    ctrl.Rule(project_outcome['high'] & feedback['excellent'], demand['high']),
    ctrl.Rule(cgpa['medium'] & tech_contribution['medium'] & feedback['average'], demand['medium']),
    ctrl.Rule(cgpa['low'] & po['medium'] & feedback['poor'], demand['low']),
    ctrl.Rule(cgpa['medium'] & po['high'] & feedback['excellent'], demand['high']),
    ctrl.Rule(cgpa['high'] & tech_contribution['high'] & project_outcome['high'], demand['high']),
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
