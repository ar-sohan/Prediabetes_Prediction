"""Draft data/feature_spec.csv from YOUR train.csv. Run once:  python make_feature_spec.py
min/max come from the training data (so coded variables are handled). REVIEW the classes and descriptions afterwards."""
import pandas as pd, yaml
cfg = yaml.safe_load(open("config.yaml")); T = cfg["target"]; excl = set(cfg.get("exclude", []) or [])
tr = pd.read_csv(cfg["paths"]["train"])

# class: easy = a chat can ask directly | noisy = self-report, imprecise | not_chat = not realistically collectable in chat
E, N, X = "easy", "noisy", "not_chat"
S = {  # feature: (class, unit, description)   -- descriptions are shown to the LLM, so make them clear
 "MealsOut": (E, "meals/week", "Number of meals per week not prepared at home"),
 "SchBkfst_0": (E, "0/1", "School breakfast category 0 (indicator)"), "SchBkfst_1": (E, "0/1", "School breakfast category 1 (indicator)"),
 "SchBkfst_2": (E, "0/1", "School breakfast category 2 (indicator)"),
 "SchLunch_0": (E, "0/1", "School lunch category 0 (indicator)"), "SchLunch_1": (E, "0/1", "School lunch category 1 (indicator)"),
 "SchLunch_2": (E, "0/1", "School lunch category 2 (indicator)"),
 "NwkLunch": (E, "per week", "School lunches eaten per week"), "NwkBkfast": (E, "per week", "School breakfasts eaten per week"),
 "Milk30Days": (E, "category", "How often the person drank milk in the past 30 days"), "SchMeal": (E, "category", "Gets school meals"),
 "NwkMeals": (E, "per week", "Number of school meals per week"), "MilkType": (E, "category", "Type of milk usually drunk"),
 "AddSugars": (N, "tsp-eq/day", "Added sugar intake per day (approximate from sugary drinks and sweets)"),
 "BMXWT": (N, "kg", "Body weight (self-reported)"), "BMXHT": (N, "cm", "Body height (self-reported)"),
 "BMXWAIST": (N, "cm", "Waist circumference (self-measured)"), "BMIcat": (N, "category", "BMI category (computed from weight, height, age, sex)"),
 "HTChol": (E, "0/1", "Ever told by a doctor of high cholesterol"), "hypten": (E, "0/1", "Ever told by a doctor of high blood pressure"),
 "HUQ020_0": (E, "0/1", "General health compared to last year: category 0"), "HUQ020_1": (E, "0/1", "General health compared to last year: category 1"),
 "HUQ020_2": (E, "0/1", "General health compared to last year: category 2"),
 "HUQ090": (E, "0/1", "Saw a mental health professional in the past year"), "HealthCond": (E, "count", "Number of reported health conditions"),
 "MCQ010": (E, "0/1", "Ever told by a doctor of asthma"), "MCQ053": (E, "0/1", "Treated for anemia recently"),
 "MCQ092": (E, "0/1", "Ever received a blood transfusion"), "AgeAsthma": (E, "years", "Age when asthma was first diagnosed"),
 "StillAsthma": (E, "0/1", "Still has asthma"), "Rx30Days": (E, "count", "Number of prescription medicines taken in the past 30 days"),
 "PLimit": (E, "0/1", "Has a physical limitation affecting activity"), "PAminWk": (E, "minutes/week", "Physical activity minutes per week"),
 "ScreenTime": (E, "hours/day", "Screen time per day (TV, computer, phone)"),
 "HHSmkNum": (E, "count", "Number of smokers in the household"), "HHSmk": (E, "0/1", "Anyone in the household smokes"),
 "Smk5Days": (E, "0/1", "Person smoked in the past 5 days"),
 "RIAGENDR": (E, "0/1", "Sex of the person (as coded in the dataset)"), "RIDAGEYR": (E, "years", "Age in years"),
 "DMDCITZN": (E, "category", "Citizenship status"), "DMDHRGND": (E, "category", "Sex of the household reference person"),
 "CountBorn": (E, "category", "Country of birth category"), "NatAcc": (E, "category", "Acculturation: nativity category"),
 "EducGrp": (E, "category", "Education group of the person"), "HHRAge": (E, "category", "Age group of the household reference person"),
 "HHREdu": (E, "category", "Education of the household reference person"), "HHSizCat": (E, "category", "Household size category"),
 "FamIncC": (N, "category", "Family income category"), "FamPIRCat": (N, "category", "Family income to poverty ratio category"),
 "RaceEth_0": (E, "0/1", "Race/ethnicity category 0 (indicator)"), "RaceEth_1": (E, "0/1", "Race/ethnicity category 1 (indicator)"),
 "RaceEth_2": (E, "0/1", "Race/ethnicity category 2 (indicator)"), "RaceEth_3": (E, "0/1", "Race/ethnicity category 3 (indicator)"),
 "LangAcc": (E, "category", "Acculturation: language category"), "AccScore": (X, "score", "Acculturation score (survey-derived)"),
 "FdInHH": (E, "category", "Household food security status"), "FdInCH": (E, "category", "Child food security status"),
 "FdInAD": (E, "category", "Adult food security status"), "FdStmp": (E, "0/1", "Household receives food stamps (SNAP)"),
 "WIC": (E, "0/1", "Receives WIC benefits"), "Hager2Q": (E, "category", "Two-question household food insecurity screener"),
 "HOD050": (E, "count", "Number of rooms in the home"),
 "HOQ065_0": (E, "0/1", "Home ownership category 0 (indicator)"), "HOQ065_1": (E, "0/1", "Home ownership category 1 (indicator)"),
 "HOQ065_2": (E, "0/1", "Home ownership category 2 (indicator)"),
 "HIns4": (E, "0/1", "Has health insurance"), "OvntPT": (E, "0/1", "Had an overnight hospital stay in the past year"),
 "RoutineHCplace": (E, "0/1", "Has a routine place for health care"),
 "TypeHCplace_0": (E, "0/1", "Type of health care place 0 (indicator)"), "TypeHCplace_1": (E, "0/1", "Type of health care place 1 (indicator)"),
 "TypeHCplace_2": (E, "0/1", "Type of health care place 2 (indicator)"),
 "TimesHCLastYr": (E, "count", "Number of health care visits in the past year"), "InSchool": (E, "0/1", "Currently attends school"),
 "Pesticide": (E, "0/1", "Pesticide used in the home"),
}
for c in ["GTot","GRefine","GWhole","VTot","VDrkGr","VRedOrTot","VRedOr","VTomato","VStarchTot","VStarchPot","VStarchOth","VOther","Legume",
          "FTot","FCitMlB","FOther","DTot","DMilk","DYogurt","DCheese","PTot","Pmps","PMeat","PFrank","POrgan","PPoult","PFishLow",
          "PFishHigh","PEgg","PSoy","PNut","Oils","SolidFat","Alcohol"]:
    S[c] = (X, "servings/day", f"Dietary recall component {c} (needs a 24-hour recall)")

rows, review = [], []
for c in tr.columns:
    if c == T: continue
    if c in excl: rows.append((c, "", "Identifier / group label (excluded)", "excluded", 0, 0)); continue
    cls, unit, desc = S.get(c, (X, "", "REVIEW: unknown column"))
    if c not in S: review.append(c)
    lo, hi = tr[c].min(), tr[c].max(); pad = 0.1 * (hi - lo) if hi > lo else 1
    rows.append((c, unit, desc, cls, 0 if lo >= 0 else round(lo - pad, 2), round(hi + pad, 2)))  # keep non-negative vars >= 0
out = pd.DataFrame(rows, columns=["feature", "unit", "description", "chat_class", "min", "max"])
out.to_csv("data/feature_spec.csv", index=False)
print(out.chat_class.value_counts().to_string()); print("columns needing review:", review)
