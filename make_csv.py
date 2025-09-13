# make_csv.py
import math
import re
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd


# ---------- 0) Утилиты форматирования ----------
def fmt_currency_kzt(x: float) -> str:
    """Формат: 12 345 ₸ (без копеек, пробел как разделитель тысяч)."""
    try:
        n = int(round(x))
    except:
        n = 0
    s = f"{n:,}".replace(",", " ")
    return f"{s} ₸"

def month_name_ru(dt: pd.Timestamp) -> str:
    months = [
        "январе", "феврале", "марте", "апреле", "мае", "июне",
        "июле", "августе", "сентябре", "октябре", "ноябре", "декабре"
    ]
    m = int(dt.strftime("%m")) - 1
    return months[m]

def last_month(df_dates: pd.Series) -> pd.Timestamp:
    """Возвращает любую дату из последнего календарного месяца, присутствующего в данных."""
    if df_dates.empty:
        return pd.Timestamp(datetime.today())
    max_date = pd.to_datetime(df_dates.max())
    return max_date


# ---------- 1) Загрузка ----------
# Ожидаем, что рядом лежат CSV с такими колонками (минимум):
# clients.csv: client_code
# transactions.csv: client_code, date, transaction_category, amount
# transfers.csv: client_code, date, transfer_type, amount
clients = pd.read_csv("clients.csv")
transactions = pd.read_csv("transactions.csv")
transfers = pd.read_csv("transfers.csv")

# Приводим даты
for df in (transactions, transfers):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Справочники категорий/типов (при необходимости адаптируйте под ваш датасет)
TRAVEL_CATS = {"такси", "поездки", "travel", "taxi", "авиа", "жд"}
RESTAURANT_CATS = {"рестораны", "кафе", "restaurants", "ювелирка", "luxury"}
ONLINE_CATS = {"онлайн-сервисы", "подписки", "online_services"}

# ---------- 2) Фичи транзакций ----------
def category_share_features(tx_df: pd.DataFrame) -> pd.DataFrame:
    # сумма трат по клиенту
    spend = tx_df.groupby("client_code")["amount"].sum().rename("total_spend")
    # доли по наборам категорий
    def frac_mask(mask):
        s = tx_df[mask].groupby("client_code")["amount"].sum()
        return (s / spend).fillna(0.0)

    # нормализуем категорию к нижнему регистру
    tx_df = tx_df.copy()
    tx_df["cat_norm"] = tx_df["transaction_category"].astype(str).str.lower()

    share_travel = frac_mask(tx_df["cat_norm"].isin(TRAVEL_CATS)).rename("share_travel")
    share_rest   = frac_mask(tx_df["cat_norm"].isin(RESTAURANT_CATS)).rename("share_rest")
    share_online = frac_mask(tx_df["cat_norm"].isin(ONLINE_CATS)).rename("share_online")

    # концентрация: доля топ-1 категории от всех трат
    top1_share = (
        tx_df.groupby(["client_code", "cat_norm"])["amount"].sum()
        .reset_index()
        .sort_values(["client_code", "amount"], ascending=[True, False])
        .groupby("client_code")
        .apply(lambda g: g["amount"].iloc[0] / g["amount"].sum() if g["amount"].sum() > 0 else 0.0)
        .rename("share_top1_cat")
    )

    # кол-во транзакций в travel
    travel_cnt = (
        tx_df[tx_df["cat_norm"].isin(TRAVEL_CATS)]
        .groupby("client_code")["amount"].size()
        .rename("travel_cnt")
        .astype(float)
    )

    feats = pd.concat([spend, share_travel, share_rest, share_online, top1_share, travel_cnt], axis=1).fillna(0.0)
    return feats.reset_index()

tx_feats = category_share_features(transactions)

# ---------- 3) Фичи переводов ----------
def transfer_features(tr_df: pd.DataFrame) -> pd.DataFrame:
    tr_df = tr_df.copy()
    tr_df["type_norm"] = tr_df["transfer_type"].astype(str).str.lower()

    def sum_type(t):
        s = tr_df[tr_df["type_norm"] == t].groupby("client_code")["amount"].sum()
        return s.rename(f"sum_{t}")

    def cnt_type(t):
        c = tr_df[tr_df["type_norm"] == t].groupby("client_code")["amount"].size()
        return c.rename(f"cnt_{t}").astype(float)

    fx_vol = (sum_type("fx_buy").fillna(0) + sum_type("fx_sell").fillna(0)).rename("fx_vol")
    invest_vol = (sum_type("invest_in").fillna(0) + sum_type("invest_out").fillna(0)).rename("invest_vol")
    deposit_net_in = (sum_type("deposit_in").fillna(0) - sum_type("deposit_out").fillna(0)).rename("deposit_net_in")
    p2p_freq = (cnt_type("p2p_in").fillna(0) + cnt_type("p2p_out").fillna(0)).rename("p2p_freq")
    atm_cnt = cnt_type("atm_withdrawal").fillna(0).rename("atm_cnt")
    salary_in = sum_type("salary_in").fillna(0).rename("salary_in")
    stipend_in = sum_type("stipend_in").fillna(0).rename("stipend_in")

    feats = pd.concat([fx_vol, invest_vol, deposit_net_in, p2p_freq, atm_cnt, salary_in, stipend_in], axis=1).fillna(0.0)
    return feats.reset_index()

tr_feats = transfer_features(transfers)

# ---------- 4) Сводные фичи по клиенту ----------
feats = (
    clients[["client_code"]]
    .merge(tx_feats, on="client_code", how="left")
    .merge(tr_feats, on="client_code", how="left")
    .fillna(0.0)
)

# ---------- 5) Скоринг продуктов (0..100) ----------
def cap01(x):
    return max(0.0, min(1.0, x))

def scores_row(r):
    scores = {}

    # Travel Card
    s_travel = (
        60 * cap01(r["share_travel"] * 3) +      # доля трат в travel
        20 * cap01(r.get("travel_cnt", 0) / 8) + # частота поездок
        20 * cap01(r["share_online"] * 2)        # онлайн сервисы как косвенный сигнал
    )
    scores["travel_card"] = min(100, s_travel)

    # Premium Card
    s_premium = (
        50 * cap01(r["share_rest"] * 3) +
        30 * cap01(r["deposit_net_in"] / 300000) +
        20 * cap01(r.get("atm_cnt", 0) / 6)      # активность, потребность в лимитах
    )
    scores["premium_card"] = min(100, s_premium)

    # Credit Card
    s_cc = (
        55 * cap01(r["share_top1_cat"] * 2.5) +  # выраженные любимые категории
        25 * cap01((r["share_rest"] + r["share_online"]) * 1.5) +
        20 * cap01((r.get("p2p_freq", 0)) / 12)  # косвенная регулярность расходов
    )
    scores["credit_card"] = min(100, s_cc)

    # FX
    s_fx = 100 * cap01(r["fx_vol"] / 200000)
    scores["fx_exchange"] = min(100, s_fx)

    # Deposits
    s_dep_save = 100 * cap01(r["deposit_net_in"] / 300000)  # сберегательный
    s_dep_acc  = 80  * cap01(r["deposit_net_in"] / 150000) + 20 * cap01(r["share_online"] * 2)  # накопительный (гибкость)
    scores["deposit_savings"] = min(100, s_dep_save)
    scores["deposit_accum"]   = min(100, s_dep_acc)

    # Investments
    s_inv = 100 * cap01(r["invest_vol"] / 200000)
    scores["investments"] = min(100, s_inv)

    # Gold
    s_gold = 60 * cap01(r["deposit_net_in"] / 200000) + 40 * cap01(1 - r.get("atm_cnt", 0) / 10)
    scores["gold"] = min(100, s_gold)

    return scores

score_cols = [
    "travel_card", "premium_card", "credit_card",
    "fx_exchange", "deposit_savings", "deposit_accum",
    "investments", "gold"
]

scores_df = feats.apply(lambda r: pd.Series(scores_row(r), index=score_cols), axis=1)
feats = pd.concat([feats, scores_df], axis=1)

# ---------- 6) Топ-4 и выбор продукта ----------
def topn_products(row, n=4):
    pairs = [(p, row[p]) for p in score_cols]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [p for p, s in pairs[:n]]

def best_product(row):
    return topn_products(row, 1)[0]

feats["top4"] = feats.apply(topn_products, axis=1)
feats["product"] = feats.apply(best_product, axis=1)

# ---------- 7) Генерация пушей ----------
CTA_LIST = ["Открыть", "Подключить", "Посмотреть", "Настроить"]

def build_observation(row, tx_last_month: pd.DataFrame, tr_last_month: pd.DataFrame) -> dict:
    """Вытянем пару чисел для персонализации за последний месяц."""
    # дата для подписи месяца
    lm = last_month(pd.concat([tx_last_month["date"], tr_last_month["date"]], ignore_index=True))
    month_ru = month_name_ru(lm)

    # суммы по travel/restaurant/online за месяц
    def sum_cat(cset):
        mask = tx_last_month["transaction_category"].str.lower().isin(cset)
        return float(tx_last_month[mask]["amount"].sum())

    travel_sum = sum_cat(TRAVEL_CATS)
    rest_sum   = sum_cat(RESTAURANT_CATS)
    online_sum = sum_cat(ONLINE_CATS)

    # агрегаты по переводам за месяц
    fx_sum = float(tr_last_month[tr_last_month["transfer_type"].str.lower().isin({"fx_buy", "fx_sell"})]["amount"].sum())
    invest_sum = float(tr_last_month[tr_last_month["transfer_type"].str.lower().isin({"invest_in", "invest_out"})]["amount"].sum())
    dep_in = float(tr_last_month[tr_last_month["transfer_type"].str.lower().eq("deposit_in")]["amount"].sum())

    return dict(
        month=month_ru,
        travel_sum=travel_sum,
        rest_sum=rest_sum,
        online_sum=online_sum,
        fx_sum=fx_sum,
        invest_sum=invest_sum,
        dep_in=dep_in
    )

def render_push(product: str, obs: dict) -> str:
    """Короткие шаблоны: наблюдение → польза → CTA."""
    m = obs["month"]
    if product == "travel_card":
        base = f"В {m} вы тратились на поездки {fmt_currency_kzt(obs['travel_sum'])}. Карта для путешествий вернёт до 4% и даст привилегии в пути. Открыть"
    elif product == "premium_card":
        base = f"Частые рестораны в {m} — повод взять премиальную карту: повышенный кешбэк и лимиты на снятие. Подключить"
    elif product == "credit_card":
        base = f"Ваши траты сконцентрированы в 1–2 категориях. Кредитная карта даст до 10% там, где вы платите чаще. Подключить"
    elif product == "fx_exchange":
        base = f"В {m} у вас были операции с валютой на {fmt_currency_kzt(obs['fx_sum'])}. Меняйте по выгодному курсу в приложении. Посмотреть"
    elif product == "deposit_savings":
        base = f"Откладываете средства ({fmt_currency_kzt(obs['dep_in'])} в {m}). Сберегательный вклад — выше ставка и защита накоплений. Открыть"
    elif product == "deposit_accum":
        base = f"Нужна гибкость накоплений? Накопительный вклад — пополняйте и снимайте без потери процентов. Открыть"
    elif product == "investments":
        base = f"Инвестиций в {m}: {fmt_currency_kzt(obs['invest_sum'])}. Начните с простых инструментов и автопокупок. Посмотреть"
    elif product == "gold":
        base = f"Думаете о защите сбережений? Золото диверсифицирует и хранится на счёте без издержек. Открыть"
    else:
        base = "Персональное предложение под ваши привычки. Открыть"
    return base

def validate_push(text: str) -> dict:
    """Проверка 4 критериев: персонализация, TOV, ясность/краткость, редполитика/формат."""
    checks = {"personal": 0, "tov": 0, "clarity": 0, "policy": 0}
    t = text.strip()

    # 1) Персонализация: есть цифры/месяц
    if re.search(r"\d", t) or any(m in t for m in ["январе","феврале","марте","апреле","мае","июне","июле","августе","сентябре","октябре","ноябре","декабре"]):
        checks["personal"] = 1

    # 2) TOV: без КАПС-слов, вежливое «вы», без токсичности
    tokens = re.findall(r"[А-ЯA-Z]{4,}", t)
    if len(tokens) == 0 and "вы" in t.lower():
        checks["tov"] = 1

    # 3) Ясность/краткость: ≤220 символов, 1 мысль (эвристика: максимум 2 предложения), 1 CTA из списка
    short_enough = len(t) <= 220
    sent_cnt = len(re.findall(r"[.!?]", t))
    has_cta = any(cta in t for cta in CTA_LIST)
    if short_enough and sent_cnt <= 2 and has_cta:
        checks["clarity"] = 1

    # 4) Редполитика/формат: не более одного "!", корректный формат валют (" ₸"), без двоеточий в конце, без CAPS
    excl_ok = t.count("!") <= 1
    kzt_ok = (" ₸" in t) or ("тенге" in t.lower()) or (not re.search(r"\d", t))  # если нет сумм — ок
    if excl_ok and kzt_ok and len(tokens) == 0:
        checks["policy"] = 1

    return checks

# ---------- 8) Соберём пуши по клиентам ----------
rows = []
for _, r in feats.iterrows():
    cid = r["client_code"]
    prod = r["product"]

    tx_c = transactions[transactions["client_code"] == cid]
    tr_c = transfers[transfers["client_code"] == cid]

    # последний месяц из данных клиента
    lm_date = last_month(pd.concat([tx_c["date"], tr_c["date"]], ignore_index=True))
    lm_start = pd.Timestamp(lm_date.year, lm_date.month, 1)
    lm_end = (lm_start + pd.offsets.MonthEnd(1))

    tx_lm = tx_c[(tx_c["date"] >= lm_start) & (tx_c["date"] <= lm_end)]
    tr_lm = tr_c[(tr_c["date"] >= lm_start) & (tr_c["date"] <= lm_end)]

    obs = build_observation(r, tx_lm, tr_lm)
    text = render_push(prod, obs)
    checks = validate_push(text)

    # при желании можно перегенерировать, если не проходит проверку (эвристика)
    if sum(checks.values()) < 3:
        # fallback: более нейтральный/короткий
        text = "Персональная рекомендация под ваши траты. Открыть"

    rows.append({"client_code": int(cid), "product": prod, "push_notification": text})

final = pd.DataFrame(rows, columns=["client_code", "product", "push_notification"])
final.to_csv("final.csv", index=False)
print("Saved: final.csv")
