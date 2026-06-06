"""Central configuration for Academic Shield app."""

# ---------------------------------------------------------------------------
# Stress-level mappings (different encoding per model)
# ---------------------------------------------------------------------------
STRESS_MAPPING_A = {"Low": 2.29, "Moderate": 4.80, "High": 7.42}
STRESS_MAPPING_B = {"Low": 0, "Moderate": 1, "High": 2}

# ---------------------------------------------------------------------------
# Model A — Burnout classification
# ---------------------------------------------------------------------------
MODEL_A_BASE_FEATURES = [
    "study_hours_per_day",
    "sleep_hours",
    "exam_pressure",
    "stress_level",
    "financial_stress",
    "social_support",
    "anxiety_score",
    "depression_score",
    "family_expectation",
    "physical_activity",
]

BURNOUT_CLASSES = {0: "Healthy", 1: "Mildly Burnout", 2: "Burnout"}
# Converts predict_proba output → gauge score (0-100)
# Weights chosen so pure-class predictions land in correct gauge segment:
# Class 0 - Healthy (weight = 20) → gauge [0-40]
# Class 1 - Mild (weight = 55) → gauge [40-70]
# Class 2 - Burnout (weight = 85) → gauge [70-100]
# Formula = proba[0] * 20 + proba[1] * 55 + proba[2] * 85
BURNOUT_SCORE_WEIGHTS = [20, 55, 85]

# ---------------------------------------------------------------------------
# Model B — GPA regression
# ---------------------------------------------------------------------------
MODEL_B_BASE_FEATURES = [
    "study_hours",
    "eca_hours",
    "sleep_hours",
    "social_hours",
    "physical_hours",
    "stress_level",
]

GPA_LABELS = [
    (3.5, "Excellent! Keep it up."),
    (3.0, "Good job!"),
    (2.5, "Not so great."),
    (0.0, "Needs improvement."),
]

# ---------------------------------------------------------------------------
# UI — Claude-style warm palette
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#FAF6F1",
    "bg_warm": "#F5EDE4",
    "surface": "#FFFFFF",
    "surface_hover": "#FDF9F5",
    "border": "rgba(0, 0, 0, 0.06)",
    "border_strong": "rgba(0, 0, 0, 0.12)",
    "accent": "#D97757",
    "accent_light": "#E8956F",
    "accent_bg": "rgba(217, 119, 87, 0.08)",
    "accent_border": "rgba(217, 119, 87, 0.2)",
    "text": "#2D2D2D",
    "text_secondary": "#6B6B6B",
    "text_muted": "#9B9B9B",
    "healthy": "#4AA785",
    "mild": "#D4A843",
    "burnout": "#C75050",
    "white": "#FFFFFF",
}

# Page 1 session-state keys
PAGE1_KEYS = ["study_hours", "sleep_hours", "eca_hours", "social_hours", "physical_hours"]

# All session-state keys required before prediction
ALL_INPUT_KEYS = PAGE1_KEYS + [
    "stress_level_category",
    "exam_pressure",
    "family_expectation",
    "financial_stress",
    "social_support",
    "anxiety_score",
    "depression_score",
]

# Insight messages keyed by burnout class
INSIGHTS = {
    2: "Your burnout level is high. Try increasing your sleep to at least 7 hours, reduce study overload, and schedule regular breaks throughout the day.",
    1: "You're showing early signs of stress. Make sure to schedule breaks, maintain social connections, and balance your workload intentionally.",
    0: "You're in good shape! Keep maintaining this balanced lifestyle and stay consistent with your habits.",
}

# Supabase table name for feedback
FEEDBACK_TABLE = "feedback"
