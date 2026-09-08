from flask import Flask, render_template, request, redirect, url_for, session
import psycopg2

app = Flask(__name__)
app.secret_key = "super-secret-key"

SUPPORTED_LANGS = ["sk", "en"]


# -----------------------
#  DATABASE
# -----------------------
def get_db_conn():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="diplomka",
        user="daniellieskovsky",
        password=""
    )
    return conn


def save_raw_answers_to_db():
    if session.get("saved_to_db"):
        return

    data = {}
    for q in QUESTIONS:
        data[q["id"]] = session.get(f"{q['id']}_raw")

    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO survey_answers_raw (
            age_cat,
            gender,
            working_student,
            unpaid_work_cat,
            father_education,
            mother_education,
            mood_cat,
            sleep_cat,
            sleep_weekend_cat,
            distance_faculty_cat,
            public_transport_cat,
            private_transport_cat,
            walking_cat,
            bike_cat,
            public_bike_cat,
            physical_activity_importance,
            motivation_physical_activity,
            smoking,
            past_smoking,
            alcohol_consumption,
            vigorous_activity_cat,
            moderate_activity_cat,
            light_activity_cat,
            sitting_cat
        )
        VALUES (
            %(age_cat)s,
            %(gender)s,
            %(working_student)s,
            %(unpaid_work_cat)s,
            %(father_education)s,
            %(mother_education)s,
            %(mood_cat)s,
            %(sleep_cat)s,
            %(sleep_weekend_cat)s,
            %(distance_faculty_cat)s,
            %(public_transport_cat)s,
            %(private_transport_cat)s,
            %(walking_cat)s,
            %(bike_cat)s,
            %(public_bike_cat)s,
            %(physical_activity_importance)s,
            %(motivation_physical_activity)s,
            %(smoking)s,
            %(past_smoking)s,
            %(alcohol_consumption)s,
            %(vigorous_activity_cat)s,
            %(moderate_activity_cat)s,
            %(light_activity_cat)s,
            %(sitting_cat)s
        )
    """, data)

    conn.commit()
    cur.close()
    conn.close()

    session["saved_to_db"] = True


# -----------------------
#  TRANSLATIONS
# -----------------------
TRANSLATIONS = {
    "sk": {
        "lang_sk": "SK",
        "lang_en": "EN",
        "question": "Otázka",
        "of": "z",
        "next": "Ďalej",
        "back": "Predchádzajúca",
        "finish": "Dokončiť",
        "summary_title": "Sumár odpovedí",
        "fill_again": "Vyplniť znova",
        "suggestions": "Odporúčania",
        "back_to_summary": "Späť na sumár",
        "no_suggestion": "Momentálne nemáme odporúčanie.",
        "prediction_title": "Predikcia",
        "prediction_tree": "Predikcia spánku",
        "recommended_changes": "Odporúčané zmeny (asociačné pravidlá)",
        "no_recommended_changes": "Netreba nič meniť. :)",
        "feel_better_group": "Čo robiť pre to aby ste sa cítili lepšie.",
        "error_required": "Prosím, vyplň odpoveď.",
        "error_invalid_number": "Zadaj platné číslo.",
        "error_invalid_int": "Zadaj platné celé číslo.",
        "error_negative": "Hodnota nemôže byť záporná.",
        "error_age_range": "Vek musí byť v rozsahu 0–120.",
        "error_mood_range": "Hodnota musí byť v rozsahu 1–10.",
        "error_hours_range": "Hodnota musí byť v rozsahu 0–24 hodín.",
        "error_range_0_10": "Hodnota musí byť v rozsahu 0–10.",
        "error_range_0_100": "Hodnota musí byť v rozsahu 0–100.",

        "male": "Muž",
        "female": "Žena",
        "yes": "Áno",
        "no": "Nie",

        "primary": "Základné",
        "secondary": "Stredoškolské",
        "university": "Vysokoškolské",

        "agree": "Súhlasím",
        "neutral": "Neutrálny postoj",
        "disagree": "Nesúhlasím",

        "every_day": "Každý deň",
        "sometimes": "Niekedy",
        "never": "Nie",

        "alcohol_never": "Nikdy",
        "alcohol_week": "Raz za týždeň",
        "alcohol_month": "Raz za mesiac",
        "alcohol_rare": "Výnimočne",

        "q_age": "Koľko máš rokov?",
        "q_gender": "Pohlavie",
        "q_working_student": "Pracujete popri štúdiu?",
        "q_unpaid_work": "Neplatená práca popri štúdiu (zadaj počet hodín)",
        "q_father_edu": "Najvyššie dosiahnuté vzdelanie otca",
        "q_mother_edu": "Najvyššie dosiahnuté vzdelanie matky",
        "q_mood": "Ako sa dnes cítiš po zdravotnej stránke (1–10)?",
        "q_sleep_work": "Koľko hodín spíš cez pracovný deň?",
        "q_sleep_weekend": "Koľko hodín spíš cez víkendový deň?",
        "q_distance": "Koľko km je od Vášho domu ku fakulte?",
        "q_public_transport": "Koľkokrát do týždňa využívate verejnú dopravu na cestu do školy?",
        "q_private_transport": "Koľkokrát do týždňa využívate súkromnú dopravu na cestu do školy?",
        "q_walking": "Koľkokrát do týždňa využívate chôdzu na cestu do školy?",
        "q_bike": "Koľkokrát do týždňa využívate vlastný bicykel na cestu do školy?",
        "q_public_bike": "Koľkokrát do týždňa využívate verejný bicykel na cestu do školy?",
        "q_pa_importance": "Je dôležitá fyzická aktivita pre zdravie?",
        "q_pa_motivation": "Som motivovaný vykonávať fyzickú aktivitu?",
        "q_smoking": "Fajčíte?",
        "q_past_smoking": "Fajčili ste v minulosti?",
        "q_alcohol": "Ako často konzumujete alkohol?",
        "q_vigorous": "Koľko minút ste počas minulého týždňa venovali náročnej fyzickej aktivite?",
        "q_moderate": "Koľko minút ste počas minulého týždňa venovali mierne náročnej fyzickej aktivite?",
        "q_light": "Koľko minút ste počas minulého týždňa venovali nenáročnej fyzickej aktivite?",
        "q_sitting": "Koľko minút ste strávili sedením za posledných 7 dní?",

        "age_1": "do 20 rokov",
        "age_2": "21 – 30 rokov",
        "age_3": "nad 31 rokov",
        "mood_bad": "Zle",
        "mood_good": "Dobre",
        "sleep_low": "Málo",
        "sleep_ok": "Dobre",
        "sleep_high": "Veľa",
        "dist_ok": "OK",
        "dist_far": "Veľa",
        "freq_low": "Menej často",
        "freq_mid": "Priemerne",
        "freq_high": "Často",
        "sit_low": "Málo",
        "sit_avg": "Priemer",
        "sit_above": "Nadpriemer",
        "sit_high": "Veľa",
    },

    "en": {
        "lang_sk": "SK",
        "lang_en": "EN",
        "question": "Question",
        "of": "of",
        "next": "Next",
        "back": "Previous",
        "finish": "Finish",
        "summary_title": "Summary of answers",
        "fill_again": "Fill again",
        "suggestions": "Suggestions",
        "back_to_summary": "Back to summary",
        "no_suggestion": "No recommendation available.",
        "prediction_title": "Prediction",
        "prediction_tree": "Prediction (tree)",
        "recommended_changes": "Recommended changes (association rules)",
        "no_recommended_changes": "No changes needed. :)",
        "feel_better_group": "What to do to feel better.",
        "error_required": "Please fill in the answer.",
        "error_invalid_number": "Enter a valid number.",
        "error_invalid_int": "Enter a valid whole number.",
        "error_negative": "Value cannot be negative.",
        "error_age_range": "Age must be between 0 and 120.",
        "error_mood_range": "Value must be between 1 and 10.",
        "error_hours_range": "Value must be between 0 and 24 hours.",
        "error_range_0_10": "Value must be between 0 and 10.",
        "error_range_0_100": "Value must be between 0 and 100.",

        "male": "Male",
        "female": "Female",
        "yes": "Yes",
        "no": "No",

        "primary": "Primary",
        "secondary": "Secondary",
        "university": "University",

        "agree": "Agree",
        "neutral": "Neutral",
        "disagree": "Disagree",

        "every_day": "Every day",
        "sometimes": "Sometimes",
        "never": "No",

        "alcohol_never": "Never",
        "alcohol_week": "Once a week",
        "alcohol_month": "Once a month",
        "alcohol_rare": "Rarely",

        "q_age": "How old are you?",
        "q_gender": "Gender",
        "q_working_student": "Do you work while studying?",
        "q_unpaid_work": "Unpaid work during studies (enter hours)",
        "q_father_edu": "Father's highest education",
        "q_mother_edu": "Mother's highest education",
        "q_mood": "How do you feel health-wise today (1–10)?",
        "q_sleep_work": "How many hours do you sleep on weekdays?",
        "q_sleep_weekend": "How many hours do you sleep on weekends?",
        "q_distance": "How many km from your home to the faculty?",
        "q_public_transport": "How many times per week do you use public transport to school?",
        "q_private_transport": "How many times per week do you use private transport to school?",
        "q_walking": "How many times per week do you walk to school?",
        "q_bike": "How many times per week do you use your bike to school?",
        "q_public_bike": "How many times per week do you use a public bike to school?",
        "q_pa_importance": "Is physical activity important for health?",
        "q_pa_motivation": "I am motivated to do physical activity.",
        "q_smoking": "Do you smoke?",
        "q_past_smoking": "Have you smoked in the past?",
        "q_alcohol": "How often do you drink alcohol?",
        "q_vigorous": "How many minutes of vigorous activity last week?",
        "q_moderate": "How many minutes of moderate activity last week?",
        "q_light": "How many minutes of light activity last week?",
        "q_sitting": "How many minutes did you spend sitting in the last 7 days?",

        "age_1": "up to 20",
        "age_2": "21–30",
        "age_3": "31+",
        "mood_bad": "Bad",
        "mood_good": "Good",
        "sleep_low": "Low",
        "sleep_ok": "OK",
        "sleep_high": "High",
        "dist_ok": "OK",
        "dist_far": "Far",
        "freq_low": "Low",
        "freq_mid": "Average",
        "freq_high": "High",
        "sit_low": "Low",
        "sit_avg": "Average",
        "sit_above": "Above avg",
        "sit_high": "High",
    },
}


def get_lang() -> str:
    lang = session.get("lang", "sk")
    return lang if lang in SUPPORTED_LANGS else "sk"


def t(key: str) -> str:
    lang = get_lang()
    return TRANSLATIONS.get(lang, {}).get(key, key)


@app.context_processor
def inject_translator():
    return {"t": t, "lang": get_lang()}


# -----------------------
#  HELPERS FOR SUGGESTIONS
# -----------------------
def to_float(value):
    if value is None or value == "":
        raise ValueError("Missing value")
    return float(str(value).replace(",", "."))


CLASS_LABELS = {
    "sk": {
        0: "Budem málo spať",
        1: "Budem spať normálne",
        2: "Budem spať veľa",
    },
    "en": {
        0: "I will sleep little",
        1: "I will sleep normally",
        2: "I will sleep a lot",
    }
}


def predict_sleep_class_from_tree():
    public_transport = to_float(session.get("public_transport_cat_raw"))
    walking = to_float(session.get("walking_cat_raw"))
    distance = to_float(session.get("distance_faculty_cat_raw"))
    alcohol = to_float(session.get("alcohol_consumption_raw"))
    vigorous = to_float(session.get("vigorous_activity_cat_raw"))
    moderate = to_float(session.get("moderate_activity_cat_raw"))
    light = to_float(session.get("light_activity_cat_raw"))
    sitting = to_float(session.get("sitting_cat_raw"))
    age = to_float(session.get("age_cat_raw"))
    private_transport = to_float(session.get("private_transport_cat_raw"))
    unpaid_work = to_float(session.get("unpaid_work_cat_raw"))

    father_education = to_float(session.get("father_education"))
    mother_education = to_float(session.get("mother_education"))
    smoking = to_float(session.get("smoking"))
    motivation = to_float(session.get("motivation_physical_activity"))
    gender = to_float(session.get("gender"))
    working_student = to_float(session.get("working_student"))

    total_activity = vigorous + moderate + light

    if public_transport <= 4.5:
        if walking <= 9.0:
            if distance <= 1.25:
                return 1
            else:
                if alcohol <= 0.5:
                    return 1
                else:
                    if total_activity <= 410.0:
                        if sitting <= 3550.0:
                            if alcohol <= 2.5:
                                if distance <= 3.5:
                                    if public_transport <= 2.5:
                                        if father_education <= 1.5:
                                            return 1
                                        else:
                                            if age <= 20.0:
                                                return 1
                                            else:
                                                return 2
                                    else:
                                        return 2
                                else:
                                    if father_education <= 1.5:
                                        return 2
                                    else:
                                        return 1
                            else:
                                if walking <= 2.5:
                                    if smoking <= 0.5:
                                        return 1
                                    else:
                                        return 1
                                else:
                                    if smoking <= 1.0:
                                        return 1
                                    else:
                                        return 2
                        else:
                            return 2
                    else:
                        return 1
        else:
            if motivation <= 1.5:
                if gender <= 0.5:
                    return 1
                else:
                    if mother_education <= 2.5:
                        return 1
                    else:
                        return 2
            else:
                if private_transport <= 0.5:
                    if sitting <= 870.0:
                        if sitting <= 315.0:
                            if gender <= 0.5:
                                return 1
                            else:
                                if father_education <= 1.5:
                                    return 1
                                else:
                                    if working_student <= 0.5:
                                        return 0
                                    else:
                                        return 1
                        else:
                            return 1
                    else:
                        return 0
                else:
                    return 1
    else:
        if unpaid_work <= 1.0:
            if working_student <= 0.5:
                if distance <= 22.5:
                    if distance <= 12.5:
                        if total_activity <= 287.5:
                            if alcohol <= 1.5:
                                if distance <= 6.75:
                                    return 1
                                else:
                                    if distance <= 8.75:
                                        return 2
                                    else:
                                        return 1
                            else:
                                return 1
                        else:
                            if walking <= 1.5:
                                return 0
                            else:
                                return 1
                    else:
                        if motivation <= 1.5:
                            if age <= 19.5:
                                return 0
                            else:
                                return 1
                        else:
                            return 0
                else:
                    if total_activity <= 210.0:
                        if gender <= 0.5:
                            return 1
                        else:
                            return 1
                    else:
                        if motivation <= 2.0:
                            return 1
                        else:
                            return 2
            else:
                return 1
        else:
            if unpaid_work <= 2.5:
                return 0
            else:
                if private_transport <= 0.5:
                    if walking <= 0.5:
                        return 1
                    else:
                        return 1
                else:
                    return 0


def add_grouped_recommendation(recommendations_dict, group_key, text):
    if group_key not in recommendations_dict:
        recommendations_dict[group_key] = []
    if text not in recommendations_dict[group_key]:
        recommendations_dict[group_key].append(text)


def get_association_recommendations():
    recommendations = {}

    try:
        mood = session.get("mood_cat")

        sleep = session.get("sleep_cat")
        bike = session.get("bike_cat")
        public_bike = session.get("public_bike_cat")
        private_transport = session.get("private_transport_cat")
        pa_importance = session.get("physical_activity_importance")
        smoking = session.get("smoking")
        vigorous = session.get("vigorous_activity_cat")
        moderate = session.get("moderate_activity_cat")
        light = session.get("light_activity_cat")
        unpaid_work = session.get("unpaid_work_cat")

        if mood == 0:
            group_key = "feel_better_group"

            if sleep != 1:
                add_grouped_recommendation(
                    recommendations,
                    group_key,
                    "Skúste spať približne 7–9 hodín denne."
                )

            if bike != 0 and sleep != 1:
                add_grouped_recommendation(
                    recommendations,
                    group_key,
                    "Menej používajte bicykel na cestu do školy a udržujte pravidelný spánok."
                )

            if pa_importance != 1 and sleep != 1:
                add_grouped_recommendation(
                    recommendations,
                    group_key,
                    "Skúste viac vnímať dôležitosť fyzickej aktivity pre zdravie."
                )

            if light != 0 and sleep != 1:
                add_grouped_recommendation(
                    recommendations,
                    group_key,
                    "Zvýšte množstvo nenáročnej fyzickej aktivity, napríklad chôdzu."
                )

            if public_bike != 0 and sleep != 1:
                add_grouped_recommendation(
                    recommendations,
                    group_key,
                    "Skúste obmedziť používanie verejného bicykla a zlepšiť spánkový režim."
                )

            if smoking != 0 and sleep != 1:
                add_grouped_recommendation(
                    recommendations,
                    group_key,
                    "Zvážte obmedzenie alebo ukončenie fajčenia."
                )

            if private_transport != 0 and sleep != 1:
                add_grouped_recommendation(
                    recommendations,
                    group_key,
                    "Menej využívajte súkromnú dopravu a skúste viac aktívne formy presunu."
                )

            if smoking != 0 and vigorous != 1:
                add_grouped_recommendation(
                    recommendations,
                    group_key,
                    "Zvýšte intenzívnu fyzickú aktivitu aspoň na 75 minút týždenne."
                )

            if moderate != 0 and sleep != 1:
                add_grouped_recommendation(
                    recommendations,
                    group_key,
                    "Zvýšte mierne náročnú fyzickú aktivitu aspoň na 150 minút týždenne."
                )

            if unpaid_work != 0 and sleep != 1:
                add_grouped_recommendation(
                    recommendations,
                    group_key,
                    "Skúste obmedziť množstvo neplatenej práce počas štúdia."
                )

    except (TypeError, ValueError):
        pass

    return recommendations


# -----------------------
#  CATEGORIZATION FUNCS
# -----------------------
def age_to_category(age: int) -> int:
    if age <= 20:
        return 1
    elif 21 <= age <= 30:
        return 2
    else:
        return 3


def unpaid_work_to_category(x: int) -> int:
    if x <= 0:
        return 0
    elif 1 <= x <= 20:
        return 1
    else:
        return 2


def mood_to_category(x: int) -> int:
    if 1 <= x <= 5:
        return 0
    elif 6 <= x <= 10:
        return 1
    else:
        raise ValueError(t("error_mood_range"))


def sleep_to_category(x: float) -> int:
    if x < 7:
        return 0
    elif x <= 9:
        return 1
    else:
        return 2


def sleep_weekend_to_category(x: float) -> int:
    if x < 7:
        return 0
    elif x < 9:
        return 1
    else:
        return 2


def distance_to_faculty_category(km: float) -> int:
    return 1 if km <= 4 else 2


def public_transport_to_category(x: int) -> int:
    if x <= 3:
        return 0
    elif x <= 6:
        return 1
    else:
        return 2


def private_transport_to_category(x: int) -> int:
    if 0 <= x <= 3:
        return 0
    elif 4 <= x <= 6:
        return 1
    elif 7 <= x <= 100:
        return 2
    else:
        raise ValueError(t("error_range_0_100"))


def walking_to_category(x: int) -> int:
    if 0 <= x <= 3:
        return 0
    elif 4 <= x <= 6:
        return 1
    elif 7 <= x <= 100:
        return 2
    else:
        raise ValueError(t("error_range_0_100"))


def bike_to_category(x: int) -> int:
    if 0 <= x <= 3:
        return 0
    elif 4 <= x <= 6:
        return 1
    elif 7 <= x <= 100:
        return 2
    else:
        raise ValueError(t("error_range_0_100"))


def public_bike_to_category(x: int) -> int:
    if 0 <= x <= 3:
        return 0
    elif 4 <= x <= 6:
        return 1
    elif 7 <= x <= 100:
        return 2
    else:
        raise ValueError(t("error_range_0_100"))


def vigorous_activity_to_category(x: int) -> int:
    if 0 <= x < 75:
        return 0
    elif x >= 75:
        return 1
    else:
        raise ValueError(t("error_negative"))


def moderate_activity_to_category(x: int) -> int:
    if x < 0:
        raise ValueError(t("error_negative"))
    elif x < 150:
        return 0
    else:
        return 1


def light_activity_to_category(x: int) -> int:
    if x < 0:
        raise ValueError(t("error_negative"))
    elif x < 300:
        return 0
    else:
        return 1


def sitting_to_category(minutes: int) -> int:
    if minutes < 0:
        raise ValueError(t("error_negative"))
    if minutes <= 360:
        return 0
    elif minutes <= 480:
        return 1
    elif minutes <= 600:
        return 2
    else:
        return 3


# -----------------------
#  QUESTIONS
# -----------------------
QUESTIONS = [
    {"id": "age_cat", "label_key": "q_age", "type": "age_to_category", "required": True},
    {"id": "gender", "label_key": "q_gender", "type": "gender_binary", "required": True},
    {"id": "working_student", "label_key": "q_working_student", "type": "yes_no_binary", "required": True},
    {"id": "unpaid_work_cat", "label_key": "q_unpaid_work", "type": "unpaid_work_to_category", "required": True},
    {"id": "father_education", "label_key": "q_father_edu", "type": "father_education", "required": True},
    {"id": "mother_education", "label_key": "q_mother_edu", "type": "mother_education", "required": True},
    {"id": "mood_cat", "label_key": "q_mood", "type": "mood_to_category", "min": 1, "max": 10, "default": 5, "required": True},
    {"id": "sleep_cat", "label_key": "q_sleep_work", "type": "sleep_to_category", "min": 0, "max": 24, "step": 0.5, "required": True},
    {"id": "sleep_weekend_cat", "label_key": "q_sleep_weekend", "type": "sleep_weekend_to_category", "min": 0, "max": 24, "step": 0.5, "required": True},
    {"id": "distance_faculty_cat", "label_key": "q_distance", "type": "distance_to_faculty_category", "min": 0, "max": 9999999, "step": 0.1, "required": True},
    {"id": "public_transport_cat", "label_key": "q_public_transport", "type": "public_transport_to_category", "min": 0, "max": 9999999, "step": 1, "required": True},
    {"id": "private_transport_cat", "label_key": "q_private_transport", "type": "private_transport_to_category", "min": 0, "max": 100, "step": 1, "required": True},
    {"id": "walking_cat", "label_key": "q_walking", "type": "walking_to_category", "min": 0, "max": 100, "step": 1, "required": True},
    {"id": "bike_cat", "label_key": "q_bike", "type": "bike_to_category", "min": 0, "max": 100, "step": 1, "required": True},
    {"id": "public_bike_cat", "label_key": "q_public_bike", "type": "public_bike_to_category", "min": 0, "max": 100, "step": 1, "required": True},
    {"id": "physical_activity_importance", "label_key": "q_pa_importance", "type": "physical_activity_importance", "required": True},
    {"id": "motivation_physical_activity", "label_key": "q_pa_motivation", "type": "motivation_physical_activity", "required": True},
    {"id": "smoking", "label_key": "q_smoking", "type": "smoking", "required": True},
    {"id": "past_smoking", "label_key": "q_past_smoking", "type": "past_smoking", "required": True},
    {"id": "alcohol_consumption", "label_key": "q_alcohol", "type": "alcohol_consumption", "required": True},
    {"id": "vigorous_activity_cat", "label_key": "q_vigorous", "type": "vigorous_activity_to_category", "min": 0, "step": 1, "required": True},
    {"id": "moderate_activity_cat", "label_key": "q_moderate", "type": "moderate_activity_to_category", "min": 0, "step": 1, "required": True},
    {"id": "light_activity_cat", "label_key": "q_light", "type": "light_activity_to_category", "min": 0, "step": 1, "required": True},
    {"id": "sitting_cat", "label_key": "q_sitting", "type": "sitting_to_category", "min": 0, "step": 1, "required": True},
]


# -----------------------
#  SUMMARY FORMAT HELPERS
# -----------------------
def format_raw_answer(q, raw):
    if raw in [None, ""]:
        return "—"

    qtype = q["type"]

    if qtype == "gender_binary":
        return t("male") if raw == "male" else t("female")

    if qtype == "yes_no_binary":
        return t("yes") if raw == "yes" else t("no")

    if qtype in ["father_education", "mother_education"]:
        if raw == "1":
            return t("primary")
        elif raw == "2":
            return t("secondary")
        elif raw == "3":
            return t("university")

    if qtype in ["physical_activity_importance", "motivation_physical_activity"]:
        if raw == "1":
            return t("agree")
        elif raw == "2":
            return t("neutral")
        elif raw == "3":
            return t("disagree")

    if qtype in ["smoking", "past_smoking"]:
        if raw == "every_day":
            return t("every_day")
        elif raw == "sometimes":
            return t("sometimes")
        elif raw == "never":
            return t("never")

    if qtype == "alcohol_consumption":
        if raw == "0":
            return t("alcohol_never")
        elif raw == "1":
            return t("alcohol_week")
        elif raw == "2":
            return t("alcohol_month")
        elif raw == "3":
            return t("alcohol_rare")

    return str(raw)


def format_category_answer(q, val):
    if val == "—" or val is None:
        return "—"

    if q["id"] == "age_cat":
        return t(f"age_{val}")

    if q["id"] == "gender":
        return t("male") if val == 0 else t("female")

    if q["id"] == "working_student":
        return t("no") if val == 0 else t("yes")

    if q["id"] == "unpaid_work_cat":
        if val == 0:
            return "0"
        elif val == 1:
            return "1–20"
        else:
            return ">20"

    if q["id"] == "father_education":
        if val == 1:
            return t("primary")
        elif val == 2:
            return t("secondary")
        else:
            return t("university")

    if q["id"] == "mother_education":
        if val == 1:
            return t("primary")
        elif val == 2:
            return t("secondary")
        else:
            return t("university")

    if q["id"] == "mood_cat":
        return t("mood_bad") if val == 0 else t("mood_good")

    if q["id"] == "sleep_cat":
        return t("sleep_low") if val == 0 else (t("sleep_ok") if val == 1 else t("sleep_high"))

    if q["id"] == "sleep_weekend_cat":
        return t("sleep_low") if val == 0 else (t("sleep_ok") if val == 1 else t("sleep_high"))

    if q["id"] == "distance_faculty_cat":
        return t("dist_ok") if val == 1 else t("dist_far")

    if q["id"] == "public_transport_cat":
        return t("freq_low") if val == 0 else (t("freq_mid") if val == 1 else t("freq_high"))

    if q["id"] == "private_transport_cat":
        return t("freq_low") if val == 0 else (t("freq_mid") if val == 1 else t("freq_high"))

    if q["id"] == "walking_cat":
        return t("freq_low") if val == 0 else (t("freq_mid") if val == 1 else t("freq_high"))

    if q["id"] == "bike_cat":
        return t("freq_low") if val == 0 else (t("freq_mid") if val == 1 else t("freq_high"))

    if q["id"] == "public_bike_cat":
        return t("freq_low") if val == 0 else (t("freq_mid") if val == 1 else t("freq_high"))

    if q["id"] == "physical_activity_importance":
        if val == 1:
            return t("agree")
        elif val == 2:
            return t("neutral")
        else:
            return t("disagree")

    if q["id"] == "motivation_physical_activity":
        if val == 1:
            return t("agree")
        elif val == 2:
            return t("neutral")
        else:
            return t("disagree")

    if q["id"] == "smoking":
        if val == 1:
            return t("every_day")
        elif val == 2:
            return t("sometimes")
        else:
            return t("never")

    if q["id"] == "past_smoking":
        if val == 1:
            return t("every_day")
        elif val == 2:
            return t("sometimes")
        else:
            return t("never")

    if q["id"] == "alcohol_consumption":
        if val == 0:
            return t("alcohol_never")
        elif val == 1:
            return t("alcohol_week")
        elif val == 2:
            return t("alcohol_month")
        else:
            return t("alcohol_rare")

    if q["id"] == "vigorous_activity_cat":
        return "<75 min" if val == 0 else "75+ min"

    if q["id"] == "moderate_activity_cat":
        return "<150 min" if val == 0 else "≥150 min"

    if q["id"] == "light_activity_cat":
        return "<300 min" if val == 0 else "≥300 min"

    if q["id"] == "sitting_cat":
        return t("sit_low") if val == 0 else (t("sit_avg") if val == 1 else (t("sit_above") if val == 2 else t("sit_high")))

    return str(val)


# -----------------------
#  ROUTES
# -----------------------
@app.route("/lang/<lang_code>")
def set_language(lang_code):
    if lang_code in SUPPORTED_LANGS:
        session["lang"] = lang_code
    return redirect(request.referrer or url_for("start"))


@app.route("/")
def start():
    current_lang = session.get("lang", "sk")
    session.clear()
    session["lang"] = current_lang
    session["saved_to_db"] = False
    return redirect(url_for("question", index=0))


@app.route("/question/<int:index>", methods=["GET", "POST"])
def question(index):
    if index >= len(QUESTIONS):
        return redirect(url_for("summary"))
    if index < 0:
        return redirect(url_for("question", index=0))

    q = QUESTIONS[index]
    error = None
    current_value = session.get(f"{q['id']}_raw", "")

    if request.method == "POST":
        if "back" in request.form:
            return redirect(url_for("question", index=max(index - 1, 0)))

        raw = request.form.get("answer", "").strip()

        if q.get("required") and raw == "":
            error = t("error_required")
            return render_template(
                "question.html",
                question=q,
                index=index,
                total=len(QUESTIONS),
                error=error,
                current_value=current_value
            )

        try:
            if q["type"] == "age_to_category":
                age = int(raw)
                if age < 0 or age > 120:
                    raise ValueError(t("error_age_range"))
                session[f"{q['id']}_raw"] = raw
                session["age_cat"] = age_to_category(age)

            elif q["type"] == "gender_binary":
                if raw not in ["male", "female"]:
                    raise ValueError(t("error_required"))
                session[f"{q['id']}_raw"] = raw
                session["gender"] = 0 if raw == "male" else 1

            elif q["type"] == "yes_no_binary":
                if raw not in ["yes", "no"]:
                    raise ValueError(t("error_required"))
                session[f"{q['id']}_raw"] = raw
                session["working_student"] = 1 if raw == "yes" else 0

            elif q["type"] == "unpaid_work_to_category":
                x = int(raw)
                if x < 0:
                    raise ValueError(t("error_negative"))
                session[f"{q['id']}_raw"] = raw
                session["unpaid_work_cat"] = unpaid_work_to_category(x)

            elif q["type"] == "father_education":
                if raw not in ["1", "2", "3"]:
                    raise ValueError(t("error_required"))
                session[f"{q['id']}_raw"] = raw
                session["father_education"] = int(raw)

            elif q["type"] == "mother_education":
                if raw not in ["1", "2", "3"]:
                    raise ValueError(t("error_required"))
                session[f"{q['id']}_raw"] = raw
                session["mother_education"] = int(raw)

            elif q["type"] == "mood_to_category":
                mood = int(raw)
                if mood < 1 or mood > 10:
                    raise ValueError(t("error_mood_range"))
                session[f"{q['id']}_raw"] = raw
                session["mood_cat"] = mood_to_category(mood)

            elif q["type"] == "sleep_to_category":
                sleep = float(raw.replace(",", "."))
                if sleep < 0 or sleep > 24:
                    raise ValueError(t("error_hours_range"))
                session[f"{q['id']}_raw"] = raw
                session["sleep_cat"] = sleep_to_category(sleep)

            elif q["type"] == "sleep_weekend_to_category":
                sleep = float(raw.replace(",", "."))
                if sleep < 0 or sleep > 24:
                    raise ValueError(t("error_hours_range"))
                session[f"{q['id']}_raw"] = raw
                session["sleep_weekend_cat"] = sleep_weekend_to_category(sleep)

            elif q["type"] == "distance_to_faculty_category":
                km = float(raw.replace(",", "."))
                if km < 0:
                    raise ValueError(t("error_negative"))
                session[f"{q['id']}_raw"] = raw
                session["distance_faculty_cat"] = distance_to_faculty_category(km)

            elif q["type"] == "public_transport_to_category":
                trips = int(raw)
                if trips < 0:
                    raise ValueError(t("error_negative"))
                session[f"{q['id']}_raw"] = raw
                session["public_transport_cat"] = public_transport_to_category(trips)

            elif q["type"] == "private_transport_to_category":
                trips = int(raw)
                if trips < 0:
                    raise ValueError(t("error_negative"))
                session[f"{q['id']}_raw"] = raw
                session["private_transport_cat"] = private_transport_to_category(trips)

            elif q["type"] == "walking_to_category":
                trips = int(raw)
                if trips < 0:
                    raise ValueError(t("error_negative"))
                session[f"{q['id']}_raw"] = raw
                session["walking_cat"] = walking_to_category(trips)

            elif q["type"] == "bike_to_category":
                trips = int(raw)
                if trips < 0:
                    raise ValueError(t("error_negative"))
                session[f"{q['id']}_raw"] = raw
                session["bike_cat"] = bike_to_category(trips)

            elif q["type"] == "public_bike_to_category":
                trips = int(raw)
                if trips < 0:
                    raise ValueError(t("error_negative"))
                session[f"{q['id']}_raw"] = raw
                session["public_bike_cat"] = public_bike_to_category(trips)

            elif q["type"] == "physical_activity_importance":
                if raw not in ["1", "2", "3"]:
                    raise ValueError(t("error_required"))
                session[f"{q['id']}_raw"] = raw
                session["physical_activity_importance"] = int(raw)

            elif q["type"] == "motivation_physical_activity":
                if raw not in ["1", "2", "3"]:
                    raise ValueError(t("error_required"))
                session[f"{q['id']}_raw"] = raw
                session["motivation_physical_activity"] = int(raw)

            elif q["type"] == "smoking":
                if raw not in ["every_day", "sometimes", "never"]:
                    raise ValueError(t("error_required"))
                session[f"{q['id']}_raw"] = raw
                if raw == "every_day":
                    session["smoking"] = 1
                elif raw == "sometimes":
                    session["smoking"] = 2
                else:
                    session["smoking"] = 0

            elif q["type"] == "past_smoking":
                if raw not in ["every_day", "sometimes", "never"]:
                    raise ValueError(t("error_required"))
                session[f"{q['id']}_raw"] = raw
                if raw == "every_day":
                    session["past_smoking"] = 1
                elif raw == "sometimes":
                    session["past_smoking"] = 2
                else:
                    session["past_smoking"] = 0

            elif q["type"] == "alcohol_consumption":
                if raw not in ["0", "1", "2", "3"]:
                    raise ValueError(t("error_required"))
                session[f"{q['id']}_raw"] = raw
                session["alcohol_consumption"] = int(raw)

            elif q["type"] == "vigorous_activity_to_category":
                minutes = int(raw)
                if minutes < 0:
                    raise ValueError(t("error_negative"))
                session[f"{q['id']}_raw"] = raw
                session["vigorous_activity_cat"] = vigorous_activity_to_category(minutes)

            elif q["type"] == "moderate_activity_to_category":
                minutes = int(raw)
                if minutes < 0:
                    raise ValueError(t("error_negative"))
                session[f"{q['id']}_raw"] = raw
                session["moderate_activity_cat"] = moderate_activity_to_category(minutes)

            elif q["type"] == "light_activity_to_category":
                minutes = int(raw)
                if minutes < 0:
                    raise ValueError(t("error_negative"))
                session[f"{q['id']}_raw"] = raw
                session["light_activity_cat"] = light_activity_to_category(minutes)

            elif q["type"] == "sitting_to_category":
                minutes = int(raw)
                if minutes < 0:
                    raise ValueError(t("error_negative"))
                session[f"{q['id']}_raw"] = raw
                session["sitting_cat"] = sitting_to_category(minutes)

            return redirect(url_for("question", index=index + 1))

        except ValueError as e:
            error = str(e)
            return render_template(
                "question.html",
                question=q,
                index=index,
                total=len(QUESTIONS),
                error=error,
                current_value=raw
            )

    return render_template(
        "question.html",
        question=q,
        index=index,
        total=len(QUESTIONS),
        error=error,
        current_value=current_value
    )


@app.route("/summary")
def summary():
    save_raw_answers_to_db()

    answers = []

    for q in QUESTIONS:
        raw_val = session.get(f"{q['id']}_raw", "—")
        cat_val = session.get(q["id"], "—")

        raw_text = format_raw_answer(q, raw_val)
        cat_text = format_category_answer(q, cat_val)

        if raw_text == "—":
            display_value = "—"
        else:
            display_value = f"{raw_text} ({cat_text})"

        answers.append({
            "label": t(q["label_key"]),
            "value": display_value
        })

    return render_template("summary.html", answers=answers, total=len(QUESTIONS))


@app.route("/suggestions")
def suggestions():
    tree_result = None
    association_results = {}
    predicted_class = None

    try:
        predicted_class = predict_sleep_class_from_tree()
        current_lang = get_lang()
        tree_result = CLASS_LABELS.get(current_lang, CLASS_LABELS["sk"]).get(
            predicted_class,
            str(predicted_class)
        )
    except (TypeError, ValueError):
        tree_result = None

    try:
        association_results = get_association_recommendations()
    except (TypeError, ValueError):
        association_results = {}

    return render_template(
        "suggestions.html",
        tree_result=tree_result,
        association_results=association_results,
        predicted_class=predicted_class
    )


if __name__ == "__main__":
    app.run(debug=True)