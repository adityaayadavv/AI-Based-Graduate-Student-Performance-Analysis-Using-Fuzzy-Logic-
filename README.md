# 🎓 AI-Based Graduate Student Performance Analysis using Fuzzy Logic

This project uses **fuzzy logic** to assess a graduate student's potential **technical demand in the job market**. Based on academic and behavioral inputs like CGPA, program outcomes, technical contributions, project outcomes, and discipline feedback, the system outputs a score and classifies the student as **Low**, **Medium**, or **High Tech Demand**.

---

## 🚀 Features

- 🔍 Fuzzy logic with 20+ expert-defined rules
- 🧠 Intelligent decision-making from vague inputs
- 🌐 Web-based interface with Flask backend
- 📊 Score + qualitative category (High, Medium, Low)
- 📦 Easily deployable and extensible

---

## 🧠 Inputs and Outputs

### ✅ Inputs (All values are numeric and normalized)

| Parameter               | Range | Description                                                                 |
|------------------------|-------|-----------------------------------------------------------------------------|
| CGPA                   | 0–10  | Student's academic grade (scaled)                                           |
| PO (Program Outcomes)  | 0–3   | Direct and indirect program outcome score (combined or averaged)            |
| Technical Contribution | 0–3   | Score based on publications, patents, or hackathons                        |
| Project Outcome        | 0–3   | Performance in final year or major project                                  |
| Feedback               | 0–3   | Faculty feedback on discipline and behavior                                 |

### ✅ Output

- A **Tech Demand Score** in the range 0–10
- A corresponding category:
  - `High Tech Demand` (8–10)
  - `Medium Tech Demand` (5–8)
  - `Low Tech Demand` (1–5)

---

## 🧰 Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask, scikit-fuzzy
- **API**: JSON-based POST to `/api/estimate`
- **Tools**: Flask-CORS, NumPy
