# CarMatch – Find Your Perfect Car
# v0.2.1 – local version

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import gradio as gr

# ── 1. Data Loading & Cleaning ────────────────────────────────────────────────

df = pd.read_csv("vehicles.csv")
df = df[(df['price'] > 1000) & (df['price'] < 50000)]
df = df[df['odometer'] < 300000]
df = df.sample(20000, random_state=42)

# ── 2. Clustering (KMeans) ────────────────────────────────────────────────────

cluster_df = pd.get_dummies(df[['fuel', 'drive', 'type']])
cluster_df['cylinders'] = df['cylinders'].fillna(4) * 2

type_dummies = pd.get_dummies(df['type'])
cluster_df = pd.concat([cluster_df, type_dummies], axis=1)

scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_df)

kmeans = KMeans(n_clusters=25, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled)

# ── 3. Cluster Labels ─────────────────────────────────────────────────────────

cluster_labels = {
    0:  "🚗 City Sedan FWD",
    1:  "🚛 RWD Truck",
    2:  "🚙 Compact SUV FWD",
    3:  "🌲 AWD Wagon",
    4:  "🏎️ Sporty Coupe RWD V8",
    5:  "👨‍👩‍👧 Family SUV 4WD",
    6:  "🛻 Everyday Pickup 4WD",
    7:  "👨‍👩‍👧 Family Minivan",
    8:  "🛋️ Cruiser Van RWD",
    9:  "🌿 Eco Hatchback",
    10: "🚌 Bus",
    11: "☀️ Convertible RWD",
    12: "⛰️ Offroad 4WD",
    13: "❓ Other FWD",
    14: "⚡ Electric",
    15: "🔧 Other 4WD",
    16: "🚛 Work Truck 4WD Gas",
    17: "🚛 Heavy Duty Truck Diesel",
    18: "🌱 Hybrid Hatchback",
    19: "🛻 Heavy Duty Pickup Diesel",
    20: "🏎️ Sport Sedan RWD",
    21: "🏎️ Sporty Coupe FWD",
    22: "🚗 AWD Sedan",
    23: "🛻 RWD Pickup",
    24: "❓ Heavy Other 4WD",
}

df['cluster_name'] = df['cluster'].map(cluster_labels)

# ── 4. Car Profiles ───────────────────────────────────────────────────────────

car_profiles = df.groupby(['cluster', 'cluster_name', 'manufacturer', 'model']).agg({
    'odometer': 'mean',
    'year':     'mean',
    'fuel':     'first',
    'drive':    'first',
    'type':     'first',
    'cylinders':'mean'
}).reset_index()

# ── 5. Button Config ──────────────────────────────────────────────────────────

IGNORE_CLUSTERS = {10, 13, 15, 24}

button_config = {
    "⚡ Fuel Efficient":       {"include": [0, 9, 14, 18, 22], "exclude": [1, 4, 6, 16, 17, 19, 23], "require": {"cylinders_max": 6}},
    "🔧 Reliable & Simple":    {"include": [0, 2, 9, 22],      "exclude": [1, 8, 16, 17, 19],         "require": {"cylinders_max": 6}},
    "💪 Powerful":             {"include": [4, 5, 6, 16, 17, 19, 20, 23], "exclude": [9, 14, 18],     "require": {"cylinders_min": 6}},
    "🏎️ Sporty":              {"include": [4, 20, 21, 11],     "exclude": [6, 7, 16, 17, 19],         "require": {}},
    "🐂 Strong Puller":        {"include": [6, 16, 17, 19, 23],"exclude": [0, 9, 14, 18, 21],         "require": {"cylinders_min": 6}},
    "🌱 Electric/Hybrid":      {"include": [14, 18],            "exclude": [1, 4, 6, 16, 17, 19, 23], "require": {}},

    "🎯 Sharp & Sporty (RWD)": {"include": [4, 8, 11, 20, 23], "exclude": [0, 2, 7, 9, 14, 18, 22],  "require": {"drive": "rwd"}},
    "🌧️ All conditions (AWD/4WD)": {"include": [3, 5, 6, 12, 16, 17, 19, 22], "exclude": [0, 8, 9, 14, 21], "require": {"drive": "4wd"}},
    "🏙️ Easy & Efficient (FWD)":   {"include": [0, 2, 7, 9, 14, 18, 21],      "exclude": [4, 6, 12, 16, 17, 19, 23], "require": {"drive": "fwd"}},

    "🪑 Just me (Coupe)":      {"include": [4, 11, 21],         "exclude": [5, 6, 7, 16, 17, 19],     "require": {}},
    "🏙️ City small (Hatchback)": {"include": [9, 14, 18, 21],  "exclude": [5, 6, 7, 16, 17, 19],     "require": {}},
    "🚗 Everyday (Sedan)":     {"include": [0, 20, 22],         "exclude": [6, 7, 16, 17, 19],         "require": {}},
    "👨‍👩‍👧 Family (Van/SUV)":    {"include": [2, 5, 7],           "exclude": [4, 11, 20, 21, 23],        "require": {}},
    "🛠️ Tools/Work (Pickup)":  {"include": [6, 16, 17, 19, 23],"exclude": [0, 2, 7, 9, 11, 14, 18, 20, 21], "require": {}},
    "🏔️ Adventure (SUV/Jeep)": {"include": [5, 12],             "exclude": [0, 7, 9, 14, 18, 21],     "require": {}},

    "🏙️ City only":            {"include": [0, 9, 14, 18, 21], "exclude": [6, 16, 17, 19, 23],        "require": {"cylinders_max": 6}},
    "🛣️ Highway cruising":     {"include": [0, 3, 8, 20, 22],  "exclude": [6, 12, 16, 17, 19],        "require": {}},
    "🪨 Some offroad":         {"include": [5, 6, 12],          "exclude": [0, 7, 9, 14, 18, 21],      "require": {"drive": "4wd"}},
    "⛰️ Heavy offroad":        {"include": [12],                "exclude": [0, 2, 7, 9, 14, 18, 20, 21], "require": {"drive": "4wd"}},
    "🏗️ Heavy towing/hauling": {"include": [16, 17, 19, 6],    "exclude": [0, 2, 7, 9, 11, 14, 18, 20, 21], "require": {"cylinders_min": 6}},

    "💰 Value for money":      {"include": [0, 2, 9, 22],       "exclude": [1, 8, 16, 17],             "require": {"cylinders_max": 8}},
    "✨ Luxury & comfort":     {"include": [0, 5, 8, 20, 22],   "exclude": [6, 9, 12, 16, 17, 19],    "require": {}},
    "🔒 Safety & reliability": {"include": [0, 2, 5, 7, 22],    "exclude": [1, 8, 23],                 "require": {}},
    "😎 Style & looks":        {"include": [4, 11, 20, 21],      "exclude": [6, 7, 16, 17, 19],         "require": {}},
    "🚀 Performance":          {"include": [4, 20],              "exclude": [0, 7, 9, 14, 18],          "require": {"cylinders_min": 6}},
}

# ── 6. Choices ────────────────────────────────────────────────────────────────

engine_choices   = ["⚡ Fuel Efficient", "🔧 Reliable & Simple", "💪 Powerful",
                    "🏎️ Sporty", "🐂 Strong Puller", "🌱 Electric/Hybrid"]
drive_choices    = ["🎯 Sharp & Sporty (RWD)", "🌧️ All conditions (AWD/4WD)",
                    "🏙️ Easy & Efficient (FWD)"]
space_choices    = ["🪑 Just me (Coupe)", "🏙️ City small (Hatchback)", "🚗 Everyday (Sedan)",
                    "👨‍👩‍👧 Family (Van/SUV)", "🛠️ Tools/Work (Pickup)", "🏔️ Adventure (SUV/Jeep)"]
usage_choices    = ["🏙️ City only", "🛣️ Highway cruising", "🪨 Some offroad",
                    "⛰️ Heavy offroad", "🏗️ Heavy towing/hauling"]
priority_choices = ["💰 Value for money", "✨ Luxury & comfort", "🔒 Safety & reliability",
                    "😎 Style & looks", "🚀 Performance"]

# ── 7. Recommender Logic ──────────────────────────────────────────────────────

def carmatch(engine, drive_style, space, usage, priority, min_km, max_km, min_year, max_year):
    errors = []
    if len(engine) > 2:      errors.append("⚠️ Engine: bitte max. 2 auswählen")
    if len(drive_style) > 1: errors.append("⚠️ Drive: bitte nur 1 auswählen")
    if len(space) > 1:       errors.append("⚠️ Space: bitte nur 1 auswählen")
    if len(usage) > 2:       errors.append("⚠️ Usage: bitte max. 2 auswählen")
    if len(priority) > 2:    errors.append("⚠️ Priority: bitte max. 2 auswählen")
    if errors:
        return pd.DataFrame({'⚠️ Bitte anpassen': errors})

    selected = list(engine) + list(drive_style) + list(space) + list(usage) + list(priority)
    if not selected:
        return pd.DataFrame({'Info': ['Bitte mindestens eine Option wählen!']})

    cluster_scores  = {i: 0 for i in range(25)}
    hard_excludes   = set()
    feature_filters = {}

    for button in selected:
        if button not in button_config: continue
        cfg = button_config[button]
        for c in cfg["include"]: cluster_scores[c] += 3
        for c in cfg["exclude"]: hard_excludes.add(c)
        req = cfg["require"]
        if "drive" in req:         feature_filters["drive"] = req["drive"]
        if "cylinders_min" in req: feature_filters["cylinders_min"] = max(feature_filters.get("cylinders_min", 0),  req["cylinders_min"])
        if "cylinders_max" in req: feature_filters["cylinders_max"] = min(feature_filters.get("cylinders_max", 99), req["cylinders_max"])

    hard_excludes |= IGNORE_CLUSTERS
    included = {c for btn in selected if btn in button_config for c in button_config[btn]["include"]}
    hard_excludes -= included

    valid_scores = {c: s for c, s in cluster_scores.items() if s > 0 and c not in hard_excludes}
    if not valid_scores:
        return pd.DataFrame({'Info': ['Keine passenden Cluster – andere Kombination versuchen!']})

    max_score       = max(valid_scores.values())
    top_cluster_ids = [c for c, s in valid_scores.items() if s >= max_score * 0.5]

    filtered = car_profiles[
        (car_profiles['cluster'].isin(top_cluster_ids)) &
        (car_profiles['odometer'] >= min_km) &
        (car_profiles['odometer'] <= max_km) &
        (car_profiles['year'] >= min_year) &
        (car_profiles['year'] <= max_year)
    ].copy()

    if "drive" in feature_filters:
        tmp = filtered[filtered['drive'] == feature_filters["drive"]]
        if not tmp.empty: filtered = tmp
    if "cylinders_min" in feature_filters:
        tmp = filtered[filtered['cylinders'] >= feature_filters["cylinders_min"]]
        if not tmp.empty: filtered = tmp
    if "cylinders_max" in feature_filters:
        tmp = filtered[filtered['cylinders'] <= feature_filters["cylinders_max"]]
        if not tmp.empty: filtered = tmp

    if filtered.empty:
        return pd.DataFrame({'Info': ['Keine Autos gefunden – Filter etwas lockerer setzen!']})

    filtered['score'] = (filtered['year'] * 0.5) - (filtered['odometer'] * 0.00001)

    n_buttons = len(selected)
    max_clusters = 2 if n_buttons <= 3 else (3 if n_buttons <= 6 else 4)
    top_cluster_ids = top_cluster_ids[:max_clusters]
    filtered = filtered[filtered['cluster'].isin(top_cluster_ids)]

    best_cluster_score = max(valid_scores.values())
    def max_per_manufacturer(manufacturer):
        scores = [valid_scores.get(c, 0) for c in filtered[filtered['manufacturer'] == manufacturer]['cluster'].unique()]
        s = max(scores, default=0)
        return 3 if s >= best_cluster_score * 0.9 else (2 if s >= best_cluster_score * 0.6 else 1)

    result_rows = []
    for manufacturer, group in filtered.groupby('manufacturer'):
        limit = max_per_manufacturer(manufacturer)
        result_rows.append(group.sample(min(limit, len(group)), random_state=None))

    top = pd.concat(result_rows).sample(frac=1, random_state=None).head(8)
    return top[['manufacturer', 'model', 'year', 'odometer', 'fuel', 'drive', 'cylinders', 'type', 'cluster_name']].round(0)

# ── 8. Gradio UI ──────────────────────────────────────────────────────────────

main_inputs = [
    gr.CheckboxGroup(choices=engine_choices,   label="1️⃣ What kind of engine do you want? (max 2)"),
    gr.CheckboxGroup(choices=drive_choices,    label="2️⃣ How should it drive? (pick 1)"),
    gr.CheckboxGroup(choices=space_choices,    label="3️⃣ How much space do you need? (pick 1)"),
    gr.CheckboxGroup(choices=usage_choices,    label="4️⃣ How do you use it? (max 2)"),
    gr.CheckboxGroup(choices=priority_choices, label="5️⃣ What matters most? (max 2)"),
    gr.Slider(0, 500000, value=0,      step=5000, label="🛣️ Min Mileage"),
    gr.Slider(0, 500000, value=150000, step=5000, label="🛣️ Max Mileage"),
    gr.Slider(1990, 2024, value=2010,  step=1,    label="📅 Min Year"),
    gr.Slider(1990, 2024, value=2024,  step=1,    label="📅 Max Year"),
]

main_ui = gr.Interface(
    fn=carmatch,
    inputs=main_inputs,
    outputs=gr.Dataframe(label="🚗 Your CarMatch Results"),
    title="🚗 CarMatch – Find Your Perfect Car",
    description="Select your preferences and find the car type that fits you – not ads, not listings, just the right kind of car."
)

app = main_ui

if __name__ == "__main__":
    app.launch()