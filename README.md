# 🚗 CarMatch

**CarMatch helps you find the right *kind* of used car — not ads, not listings, just a recommendation that fits your lifestyle.**

You answer five questions (engine feel, drive style, space, usage, priorities) and CarMatch returns up to 8 specific make/model suggestions from a dataset of ~83,000 real used car listings.

---

## How it works

### 1. Dataset & filtering
Loads ~83k used vehicle listings and keeps only cars priced $1k–$50k with under 300k miles. A random sample of 20,000 is used for processing.

### 2. Unsupervised clustering
KMeans (k=25) groups vehicles by their characteristics — fuel type, drivetrain, body type, and cylinder count — using one-hot encoding + `StandardScaler`. Each cluster gets a human-readable label like *"🏎️ Sporty Coupe RWD V8"* or *"🌿 Eco Hatchback"*.

### 3. Button-to-cluster mapping
Each UI button (e.g. *"⚡ Fuel Efficient"* or *"🏔️ Adventure SUV/Jeep"*) maps to a set of:
- **include_clusters** — clusters that get a positive score
- **exclude_clusters** — clusters that are hard-filtered out
- **optional constraints** — drive type or cylinder range

### 4. Scoring & ranking
When you submit your selections, cluster scores are summed across your chosen buttons. Clusters scoring ≥50% of the top score survive. Vehicles in those clusters are then ranked by:

```
score = year × 0.5 − odometer × 0.00001
```

Results are capped at 8, spread across 2–4 clusters, and limited to 1–3 cars per manufacturer to ensure variety.

---

## Tech stack

| Layer | Library |
|---|---|
| UI | [Gradio](https://gradio.app) |
| Data | Pandas, NumPy |
| ML | scikit-learn (KMeans, StandardScaler) |
| Runtime | Python 3.x |

---

## Run locally

```bash
git clone https://github.com/tfForster/CarMatch-.git
cd CarMatch

pip install -r requirements.txt

# Add vehicles.csv to the project root (not included in repo)
python app.py
```

Open `http://localhost:7860` in your browser.

> `vehicles.csv` is not included in the repo. The app expects a CSV with columns: `price`, `year`, `odometer`, `manufacturer`, `model`, `fuel`, `transmission`, `cylinders`, `type`, `drive`, `condition`. A compatible dataset is available on [Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data).

---

## UI overview

| Section | Options | Limit |
|---|---|---|
| 1️⃣ Engine | Fuel Efficient, Reliable, Powerful, Sporty, Strong Puller, Electric/Hybrid | max 2 |
| 2️⃣ Drive style | RWD · AWD/4WD · FWD | pick 1 |
| 3️⃣ Space | Coupe, Hatchback, Sedan, Family SUV, Pickup, Adventure SUV | pick 1 |
| 4️⃣ Usage | City, Highway, Some offroad, Heavy offroad, Towing | max 2 |
| 5️⃣ Priority | Value, Luxury, Safety, Style, Performance | max 2 |
| Sliders | Mileage range · Year range | — |
