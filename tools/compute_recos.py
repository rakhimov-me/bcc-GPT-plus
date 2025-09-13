# tools/compute_recos.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import glob

# ---------- входные файлы ----------
CSV_DIR = Path("csv")
CLIENTS_PATH = CSV_DIR / "clients.csv"
TX_GLOB = str(CSV_DIR / "client_*_transactions_3m.csv")
TR_GLOB = str(CSV_DIR / "client_*_transfers_3m.csv")

TODAY = datetime(2025, 9, 13)
RU_MONTHS = {1:"январе",2:"феврале",3:"марте",4:"апреле",5:"мае",6:"июне",
             7:"июле",8:"августе",9:"сентябре",10:"октябре",11:"ноябре",12:"декабре"}

def fmt_kzt(x):
    x = float(x) if pd.notna(x) else 0.0
    r = int(round(x/10.0)*10)
    return f"{r:,}".replace(",", " ") + " ₸"

# ---------- читаем данные ----------
clients = pd.read_csv(CLIENTS_PATH)

tx_files = sorted(glob.glob(TX_GLOB))
tr_files = sorted(glob.glob(TR_GLOB))
if not tx_files:
    raise FileNotFoundError(f"Не найдены файлы транзакций по маске: {TX_GLOB}")
if not tr_files:
    raise FileNotFoundError(f"Не найдены файлы переводов по маске: {TR_GLOB}")

transactions = pd.concat((pd.read_csv(p) for p in tx_files), ignore_index=True)
transfers = pd.concat((pd.read_csv(p) for p in tr_files), ignore_index=True)

# ---------- препроцесс ----------
tx = transactions.copy()
tx["date"] = pd.to_datetime(tx["date"])
tx["month"] = tx["date"].dt.to_period("M")

tr = transfers.copy()
tr["date"] = pd.to_datetime(tr["date"])
tr["month"] = tr["date"].dt.to_period("M")

last_month_dt = TODAY.replace(day=1) - timedelta(days=1)
last_month = last_month_dt.month

spend = tx.groupby(["client_code","category"], as_index=False)["amount"].sum().rename(columns={"amount":"sum_spend"})
spend_lastm = tx[tx["date"].dt.month==last_month].groupby(["client_code","category"], as_index=False)["amount"].sum().rename(columns={"amount":"sum_spend_lastm"})

def spend_in(client_id, cats, use_lastm=False):
    if use_lastm:
        s = spend_lastm[(spend_lastm.client_code==client_id)&(spend_lastm.category.isin(cats))]["sum_spend_lastm"].sum()
    else:
        s = spend[(spend.client_code==client_id)&(spend.category.isin(cats))]["sum_spend"].sum()
    return float(s) if pd.notna(s) else 0.0

tr_types = tr.groupby(["client_code","type"], as_index=False)["amount"].sum().rename(columns={"amount":"sum_amount"})
tr_count = tr.groupby(["client_code","type"], as_index=False).size().rename(columns={"size":"count"})

def tr_sum(client_id, types):
    s = tr_types[(tr_types.client_code==client_id)&(tr_types.type.isin(types))]["sum_amount"].sum()
    return float(s) if pd.notna(s) else 0.0

def tr_cnt(client_id, types):
    c = tr_count[(tr_count.client_code==client_id)&(tr_count.type.isin(types))]["count"].sum()
    return int(c) if pd.notna(c) else 0

fx_spend_count = tx[tx.currency.ne("KZT")].groupby("client_code").size().rename("fx_tx_count")
fx_spend_count = fx_spend_count.reindex(clients.client_code).fillna(0).astype(int)

CATEGORIES = tx["category"].dropna().unique().tolist()

# ---------- модели выгод ----------
def benefit_travel_card(cid):
    return 0.04 * spend_in(cid, ["Путешествия","Такси","Отели"], use_lastm=True)

def benefit_premium_card(row, cid):
    bal = row["avg_monthly_balance_KZT"]
    rate = 0.04 if bal>=6_000_000 else (0.03 if bal>=1_000_000 else 0.02)
    boosted_cats = ["Ювелирные украшения","Косметика и Парфюмерия","Кафе и рестораны"]
    boosted = spend_in(cid, boosted_cats, use_lastm=True)
    other = spend_in(cid, CATEGORIES, use_lastm=True) - boosted
    expected = 0.04*boosted + rate*max(other,0)
    return min(expected, 100_000)

def benefit_credit_card(cid):
    cl = spend_lastm[spend_lastm.client_code==cid].sort_values("sum_spend_lastm", ascending=False)
    top3 = cl.category.head(3).tolist()
    online = ["Играем дома","Смотрим дома","Едим дома"]
    return 0.10 * (spend_in(cid, top3, use_lastm=True) + spend_in(cid, online, use_lastm=True))

def benefit_fx(cid):
    ops = tr_cnt(cid, ["fx_buy","fx_sell"]) + fx_spend_count.loc[cid]
    turnover = tr_sum(cid, ["fx_buy","fx_sell"])
    return 0.0015*turnover + 500*min(ops,5)

def benefit_cash_loan(row, cid):
    bal = row["avg_monthly_balance_KZT"]
    stress = tr_cnt(cid, ["cc_repayment_out","loan_payment_out","installment_payment_out"])
    return 15_000 + 2_000*(stress-2) if (bal<200_000 and stress>=3) else 0.0

def benefit_deposit_savings(row, cid):
    bal = row["avg_monthly_balance_KZT"]
    mspend = spend_in(cid, CATEGORIES, use_lastm=True)
    return (0.165/12)*bal if (bal>=1_500_000 and mspend < 0.3*bal) else 0.0

def benefit_deposit_accum(row, cid):
    bal = row["avg_monthly_balance_KZT"]
    topups = tr_sum(cid, ["deposit_topup_out"])
    return bal*(0.155/12)*0.5 if (bal>=500_000 or topups>0) else 0.0

def benefit_deposit_multi(row, cid):
    multi_curr = tx[tx.client_code==cid]["currency"].nunique() >= 2
    fx_dep = tr_sum(cid, ["deposit_fx_topup_out","deposit_fx_withdraw_in"])
    if multi_curr or fx_dep>0:
        alloc = max(row["avg_monthly_balance_KZT"]*0.3, fx_dep*0.3)
        return alloc*(0.145/12)
    return 0.0

def benefit_investments(row, cid):
    inv = tr_sum(cid, ["invest_out","invest_in"])
    return 5000 + 0.002*inv if (inv>0 or row["status"] in ["Студент","Зарплатный клиент"]) else 0.0

def benefit_gold(cid):
    g = tr_sum(cid, ["gold_buy_out","gold_sell_in"])
    return 3000 + 0.001*g if g>0 else 0.0

PRODUCTS = [
    "Карта для путешествий","Премиальная карта","Кредитная карта","Обмен валют",
    "Кредит наличными","Депозит Мультивалютный","Депозит Сберегательный",
    "Депозит Накопительный","Инвестиции","Золотые слитки"
]

def estimate(row):
    cid = row["client_code"]
    return {
        "Карта для путешествий": benefit_travel_card(cid),
        "Премиальная карта":    benefit_premium_card(row, cid),
        "Кредитная карта":      benefit_credit_card(cid),
        "Обмен валют":          benefit_fx(cid),
        "Кредит наличными":     benefit_cash_loan(row, cid),
        "Депозит Мультивалютный": benefit_deposit_multi(row, cid),
        "Депозит Сберегательный": benefit_deposit_savings(row, cid),
        "Депозит Накопительный":  benefit_deposit_accum(row, cid),
        "Инвестиции":           benefit_investments(row, cid),
        "Золотые слитки":       benefit_gold(cid),
    }

def gen_push(product, row, benefits):
    cid = row["client_code"]; name=row["name"]; mname=RU_MONTHS[last_month_dt.month]
    taxis = spend_in(cid, ["Такси"], use_lastm=True)
    trips = spend_in(cid, ["Путешествия","Отели"], use_lastm=True)
    bal = row["avg_monthly_balance_KZT"]
    cl = spend_lastm[spend_lastm.client_code==cid].sort_values("sum_spend_lastm", ascending=False)
    cats = cl.category.head(3).tolist()
    fx_ops = tr_cnt(cid, ["fx_buy","fx_sell"])
    fx_curr = next((c for c in ["USD","EUR","RUB"] if c in tx[tx.client_code==cid]["currency"].unique()), "USD")
    b = benefits.get(product, 0.0); bt = fmt_kzt(b) if b>0 else ""

    if product=="Карта для путешествий":
        total = taxis + trips
        return f"{name}, в {mname} у вас поездок и такси на {fmt_kzt(total)}. С тревел-картой вернули бы ≈{bt}. Оформить карту."
    if product=="Премиальная карта":
        return f"{name}, у вас крупный остаток ({fmt_kzt(bal)}). Премиальная карта даст повышенный кешбэк и бесплатные снятия — выгода ≈{bt}/мес. Оформить сейчас."
    if product=="Кредитная карта":
        cats_txt = ', '.join(cats) if cats else "ваших категориях"
        return f"{name}, ваши топ-категории — {cats_txt}. Кредитная карта даёт до 10% и на онлайн-сервисы. Потенциальная выгода ≈{bt}/мес. Оформить карту."
    if product=="Обмен валют":
        return f"{name}, вы часто платите в {fx_curr} и меняете валюту ({fx_ops} операции в {mname}). В приложении — выгодный курс и авто-покупка. Настроить обмен."
    if product=="Кредит наличными":
        return f"{name}, если нужен запас на крупные траты — можно оформить кредит наличными с гибкими выплатами. Узнать доступный лимит."
    if product=="Депозит Мультивалютный":
        return f"{name}, часть средств можно держать в {fx_curr}. Мультивалютный вклад — {bt} в месяц при текущем балансе. Открыть вклад."
    if product=="Депозит Сберегательный":
        return f"{name}, свободные {fmt_kzt(bal)} могут работать под 16,5%. Доходность ≈{bt}/мес при размещении. Открыть вклад."
    if product=="Депозит Накопительный":
        return f"{name}, удобно откладывать под 15,5% без снятий. С вашим балансом выгода ≈{bt}/мес. Открыть вклад."
    if product=="Инвестиции":
        return f"{name}, попробуйте инвестиции: без комиссий на старт и порог от 6 ₸. Начните с малого — счёт откроется за пару минут. Открыть счёт."
    if product=="Золотые слитки":
        return f"{name}, для диверсификации — золотые слитки 999,9. Покупка с предзаказом в приложении. Посмотреть варианты."
    return f"{name}, посмотрите подходящий продукт — это может дать вам ощутимую выгоду. Посмотреть."

# ---------- прогон ----------
rows, dbg = [], []
for _, r in clients.iterrows():
    bmap = {k:max(0.0,float(v)) for k,v in estimate(r).items()}
    ranked = sorted(bmap.items(), key=lambda kv: kv[1], reverse=True)
    prod, _ = ranked[0]
    push = gen_push(prod, r, dict(ranked))
    rows.append({"client_code": int(r["client_code"]), "product": prod, "push_notification": push})
    dbg.append({"client_code": int(r["client_code"]), "name": r["name"], **{f"rank{i+1}": p for i,(p,_) in enumerate(ranked[:4])}, **{f"score_{p}": s for p,s in ranked}})

pd.DataFrame(rows).sort_values("client_code").to_csv(
    "recommendations.csv",
    index=False,
    encoding="utf-8-sig",
    sep=";",
    decimal=",",
    float_format="%.2f"
)
pd.DataFrame(dbg).sort_values("client_code").to_csv(
    "top4_debug.csv",
    index=False,
    encoding="utf-8-sig",
    sep=";",
    decimal=",",
    float_format="%.2f"
)
print("Done: recommendations.csv, top4_debug.csv")
