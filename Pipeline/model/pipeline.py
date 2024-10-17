import dill
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler


def balance_df(df):
    drop_num = df.shape[0] - 230000  # Удаляем случайные строки
    random_rows = df[df.event_value == 0].sample(n=drop_num, random_state=42)
    df.drop(random_rows.index, inplace=True)
    df.to_csv("data/balanced_df.csv")


def cast_types(dataframe):
    df = dataframe.copy()
    # Приведение типов

    df["utm_source"] = df["utm_source"].astype("str")
    df["utm_campaign"] = df["utm_campaign"].astype("str")
    df["utm_adcontent"] = df["utm_adcontent"].astype("str")

    return df


def fill(dataframe):
    df = dataframe.copy()

    # print("Filling missed values")
    # Заполняем пропуски в бренде (доля 25%) на (no set)
    df.device_brand = df.device_brand.fillna("(not set)")
    # Заполняем (not set) бренда моделью AuMdmADEIoPXiWpTsBE (доля >50%)
    condition = (df["device_brand"] == "(not set)") & (df["device_model"] == "AuMdmADEIoPXiWpTsBEj")
    df.loc[condition, "device_brand"] = "AuMdm"

    # Заполнение пропусков ОС в Apple
    mask = df["device_brand"] == "Apple"
    df.loc[mask, "device_os"] = df.loc[mask, "device_os"].fillna("iOS")
    # Заполнение пропусков ОС для ПК
    mask = df["device_category"] == "desktop"
    df.loc[mask, "device_os"] = df.loc[mask, "device_os"].fillna("Windows")

    return df


def add_features(dataframe):
    df = dataframe.copy()

    # print("Creating new features")
    df["device_pixels"] = df["device_screen_resolution"].apply(lambda p: int(p.split("x")[0]) * int(p.split("x")[1]))
    df["visit_freq"] = df["visit_number"].apply(lambda n: "low" if n < 3 else ("high" if n > 50 else "medium"))
    df["visit_month"] = df["visit_date"].apply(lambda d: d.month)
    df["visit_day"] = df["visit_date"].apply(lambda d: d.day)
    df["visit_hour"] = df["visit_time"].apply(lambda t: t.hour)
    df["geo_russian"] = df["geo_country"].apply(lambda c: 1 if c == "Russia" else 0)
    df["utm_source0"] = df["utm_source"].apply(lambda s: s[0])
    df["utm_adcontent0"] = df["utm_adcontent"].apply(lambda c: c[0])
    df["utm_campaign0"] = df["utm_campaign"].apply(lambda c: c[0])

    return df


def reduce_val(dataframe):
    df = dataframe.copy()

    # print("Reducing dimension")
    # Формирование групп привлечения
    df["utm_medium"] = df["utm_medium"].replace(["cpc", "google_cpc", "yandex_cpc", "cpm", "CPM",
                                                 "cpv", "last", "users_msk", "reach", "cpa", "clicks", "linktest"],
                                                "efficiency")
    df["utm_medium"] = df["utm_medium"].replace(["banner", "static", "cbaafe", "dom_click", "nkp", "link",
                                                 "smartbanner", "qrcodevideo", "medium", "tablet", "(not set)",
                                                 "partner", "sber_app", "qr", "promo_sber", "promo_sbol",
                                                 "Sbol_catalog",
                                                 "catalogue", "app"], "media")
    df["utm_medium"] = df["utm_medium"].replace(["smm", "stories", "blogger_channel", "blogger_stories", "vk_smm",
                                                 "fb_smm", "ok_smm", "social", "tg", "blogger_header", "post",
                                                 "article"], "blog")
    df["utm_medium"] = df["utm_medium"].replace(["organic", "(none)", "referral", "landing", "landing_interests",
                                                 "web_polka", "main_polka"], "seo")
    df["utm_medium"] = df["utm_medium"].replace(["email", "info_text", "outlook", "sms", "push"], "info")

    # Сократим кол-во значений device_browser, взяв первое слово
    df.device_browser = df.device_browser.apply(lambda b: b.split()[0])

    def replace_unpopular(col, n_pop):
        top_values = df[col].value_counts().nlargest(n_pop).index
        df.loc[~df[col].isin(top_values), col] = "other"

    replace_unpopular("device_browser", 5)
    replace_unpopular("device_brand", 7)
    replace_unpopular("device_os", 4)
    replace_unpopular("geo_city", 10)
    replace_unpopular("utm_adcontent0", 5)
    replace_unpopular("utm_source0", 7)
    replace_unpopular("utm_campaign0", 5)
    return df


def remove_outliers(dataframe):
    df = dataframe.copy()

    def get_bound(attribute):
        q25 = attribute.quantile(0.25)
        q75 = attribute.quantile(0.75)
        iqr = q75 - q25
        return q75 + 1.5 * iqr

    # print("Removing outliers")
    condition = df.device_pixels < 240 * 320
    df.loc[condition, "device_pixels"] = 240 * 320  # Замена логически малых значений расширения
    bound = get_bound(df["device_pixels"])
    df.loc[df["device_pixels"] > bound, "device_pixels"] = round(bound)

    bound = get_bound(df["visit_number"])
    df.loc[df["visit_number"] > bound, "visit_number"] = round(bound)

    return df


def filter_data(dataframe):
    columns_to_drop = [
        "session_id", "client_id", "device_screen_resolution", "device_model", "hit_page_path", "hit_time",
        "hit_date", "hit_type", "hit_referer", "utm_source", "utm_adcontent", "utm_campaign", "utm_keyword",
        "visit_date", "visit_time", "geo_country", "hit_number", "event_action", "event_label", "event_category"
    ]
    return dataframe.drop(columns_to_drop, axis=1)


def main():
    print("Event Value Prediction Pipeline")

    df1 = pd.read_csv("data/ga_sessions.csv", parse_dates=["visit_date"], low_memory=False)
    df1["visit_time"] = df1["visit_time"].apply(lambda t: pd.to_datetime(t))
    df2 = pd.read_csv("data/ga_hits.csv")
    df = df1.set_index("session_id").join(df2.set_index("session_id"), how="inner").reset_index()
    del df1, df2

    # Формирование целевого признака
    cols = ["sub_car_claim_click", "sub_car_claim_submit_click",
            "sub_open_dialog_click", "sub_custom_question_submit_click",
            "sub_call_number_click", "sub_callback_submit_click", "sub_submit_success",
            "sub_car_request_submit_click"]
    df["event_value"] = df["event_action"].isin(cols).astype("int")
    balance_df(df)
    print("Created source dataframe")

    x = df.drop(["event_value"], axis=1)
    y = df["event_value"]

    numerical_features = make_column_selector(dtype_exclude=object)
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value="other")),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ("cast", FunctionTransformer(cast_types)),
        ("fill", FunctionTransformer(fill)),
        ("add", FunctionTransformer(add_features)),
        ("reduce", FunctionTransformer(reduce_val)),
        ("remove", FunctionTransformer(remove_outliers)),
        ("filter", FunctionTransformer(filter_data)),
        ('column_transformer', column_transformer)
    ])

    models = (
        LogisticRegression(C=1, max_iter=300, solver="liblinear"),
        RandomForestClassifier(bootstrap=True, max_depth=15, max_features="sqrt"),
        MLPClassifier(activation="tanh", alpha=0.01, hidden_layer_sizes=(50, 20),
                      max_iter=400)
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, x, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, auc_mean: {score.mean():.4f}, auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')
    best_pipe.fit(x, y)
    with open('event_pipe.pkl', 'wb') as file:
        dill.dump({
            "model": best_pipe,
            "metadata": {
                "name": "Event value prediction model",
                "author": "Ekaterina Shimchyonok",
                "version": 1,
                "type": type(best_pipe.named_steps["classifier"]).__name__,
                "roc_auc ": best_score
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
