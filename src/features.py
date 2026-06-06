"""Feature engineering functions for Model A and Model B's experimentation.

These must match the functions that were cloudpickled in the notebooks.
They are injected into __main__ before loading the pkl files so that
cloudpickle can resolve the references.
"""

import sys


def engineer_features_a(df):
    df = df.copy()
    df["stress_x_anxiety"]     = df["stress_level"] * df["anxiety_score"]
    df["stress_x_depression"]  = df["stress_level"] * df["depression_score"]
    df["stress_x_exam"]        = df["stress_level"] * df["exam_pressure"]
    df["anxiety_x_depression"] = df["anxiety_score"] * df["depression_score"]
    df["study_sleep_ratio"]    = df["study_hours_per_day"] / (df["sleep_hours"] + 1)
    df["pressure_support_gap"] = (
        (df["exam_pressure"] + df["financial_stress"] + df["family_expectation"]) / 3
    ) - df["social_support"]
    df["stress_level_sq"]      = df["stress_level"] ** 2
    df["anxiety_score_sq"]     = df["anxiety_score"] ** 2
    df["sleep_deprived"]       = (df["sleep_hours"] < 5).astype(int)
    df["high_stress"]          = (df["stress_level"] > 7).astype(int)
    return df

# Feature engineering for Model B experimentation (modelB_v2.ipynb)
# NOT used for the deployment — modelB.pkl was trained without feature engineering
# Kept here for reference and future experimentation only
def engineer_features_b(df):
    df = df.copy()
    df["study_proportion"]     = df["study_hours"] / (24 - df["sleep_hours"] + 1e-6)
    df["study_leisure_ratio"]  = df["study_hours"] / (df["social_hours"] + df["physical_hours"] + 1)
    df["study_x_stress"]       = df["study_hours"] * df["stress_level"]
    df["effective_study"]      = df["study_hours"] * (1 - df["stress_level"] / 4)
    df["wellbeing_index"]      = df["sleep_hours"] + df["social_hours"] - df["stress_level"]
    df["academic_intensity"]   = df["study_hours"] + df["eca_hours"] - df["social_hours"]
    df["study_efficiency"]     = df["study_hours"] / (df["study_hours"] + df["social_hours"] + df["physical_hours"] + 1)
    hours = df[["study_hours", "sleep_hours", "social_hours", "physical_hours"]]
    df["balanced_lifestyle"]   = 1 - (hours.std(axis=1) / (hours.mean(axis=1) + 1e-6))
    return df


def patch_main():
    """Inject feature functions into __main__ so cloudpickle can resolve them."""
    main = sys.modules.get("__main__")
    if main is not None:
        main.engineer_features_a = engineer_features_a
        main.engineer_features_b = engineer_features_b
