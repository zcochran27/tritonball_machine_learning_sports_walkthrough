# 🧠 Triton Ball: Machine Learning for Sports — Code Walkthrough

This repository contains the notebook **`ML_walkthrough.ipynb`** and supporting data for **Triton Ball**, the UCSD sports analytics club.  
In this walkthrough, we demonstrate how to apply machine learning methods to solve classic problems in sports analytics across multiple sports.

---

## 🏀 1. Linear Regression — Identifying Key NBA Statistics

We begin with a **linear regression model** to identify statistically significant season-level metrics that correlate with a high win percentage in the NBA.

We explore:
- Feature selection and correlation analysis  
- Model fitting and interpretation of coefficients  
- Statistical significance testing  

---

## ⚽ 2. Expected Goals (xG) — Building a Soccer Model from Scratch

Next, we construct an **expected goals (xG)** model using **logistic regression** and open-source data from **StatsBomb** via `statsbombpy`.

We demonstrate:
- Data cleaning, processing, and feature engineering for shot events  
- Modeling the probability that a shot results in a goal  

Two approaches are compared:
1. **Basic model** — uses Cartesian and one-hot encoded features  
2. **Advanced model** — incorporates freeze-frame data to encode player positioning context  

Model evaluation includes:
- Logistic loss  
- AUC (Area Under the Curve)  
- Calibration curves  

---

## ⚾ 3. Expected Batting Average (xBA) — Using Generalized Additive Models

We then move to baseball analytics, building an **expected batting average (xBA)** model using **Generalized Additive Models (GAMs)** and **Statcast** data.

We show how to:
- Model the probability of a hit from exit velocity and launch angle  
- Visualize the optimal batting zone for hitters  

---

## ⚽ 4. VAEP Framework — Valuing On-Ball Actions in Soccer

We implement the **VAEP framework (Valuing Actions by Estimating Probabilities)** to evaluate the contribution of each on-ball action to a team’s scoring or conceding probability.

Using event data from `statsbombpy`, we engineer the full set of **VAEP features**, including:

### 🔹 Action Context
- `type_name` — action type (pass, dribble, shot, tackle, interception, etc.)  
- `result` — success or failure  
- `bodypart` — foot, head, other  
- `time` — match time (seconds or normalized)  
- `team` — team in possession  

### 🔹 Spatial Features
*(Coordinates standardized to [0, 1])*
- `start_x`, `start_y`, `end_x`, `end_y` — action start and end positions  
- `dx`, `dy` — displacement  
- `distance_to_goal`, `angle_to_goal` — geometric features from the goal  
- `end_distance_to_goal`, `end_angle_to_goal` — same for end location  

### 🔹 Dynamic / Contextual Features
- `team_in_possession` — binary flag  
- `scoreline` — current goal differential  
- `period` — first half, second half, or extra time  
- `time_remaining` — normalized time left  
- `possession_sequence_length` — actions in current possession  
- `seconds_since_last_action`, `distance_from_last_action`  
- `same_team_as_last_action`  

We train two **XGBoost** models:
- P₍score₎ — probability of scoring within the next few actions  
- P₍concede₎ — probability of conceding  

The **VAEP value** for each action is computed as:

ΔP₍score₎ − ΔP₍concede₎

We visualize the distribution of VAEP predictions and identify the most valuable players (on-ball contribution) in the **2015/16 Bundesliga** and **La Liga** seasons.

---

## 🏀 5. Graph Neural Networks — Turnover-to-Score Prediction in Basketball

Finally, we explore **Graph Neural Networks (GNNs)** to predict the probability that a **turnover** leads to an opponent scoring opportunity.

We use **NBA tracking data** from  
👉 [Hugging Face: dcayton/nba_tracking_data_15_16](https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16)

Steps include:
1. Detecting turnover events and defining possessions  
2. Encoding **player–team–ball** relationships as graph structures  
3. Defining:
   - **Node features:** position (x, y), speed, direction, turnover flag  
   - **Edge features:** coordinate distance, speed difference  
4. Labeling turnovers that lead to opponent scores within the next 10 seconds  
5. Training and visualizing a **GNN classifier** to predict these outcomes  

---

## 📘 Summary

Each section of this walkthrough highlights a unique **machine learning paradigm** applied to sports data:

| Sport | Task | Model Type |
|--------|------|------------|
| 🏀 Basketball | Win prediction | Linear Regression |
| ⚽ Soccer | Expected goals (xG) | Logistic Regression |
| ⚾ Baseball | Expected batting average (xBA) | GAM |
| ⚽ Soccer | Action valuation (VAEP) | XGBoost |
| 🏀 Basketball | Turnover scoring prediction | Graph Neural Network |
