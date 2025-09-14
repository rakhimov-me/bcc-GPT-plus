# tools/generate_mock_data.py
import os, tempfile, time, stat
import random
from datetime import datetime, timedelta
from pathlib import Path
import csv
import math
import pandas as pd

# ---------- пути ----------
CSV_DIR = Path("csv")
CSV_DIR.mkdir(parents=True, exist_ok=True)
CLIENTS_PATH = CSV_DIR / "clients.csv"

# ---------- даты (3 месяца под расчёт рекомендаций) ----------
START = datetime(2025, 6, 1, 8, 0, 0)
END   = datetime(2025, 8, 31, 22, 0, 0)

random.seed(42)

# ---------- справочники ----------
CITIES = ["Алматы","Астана","Караганда","Шымкент","Павлодар","Костанай","Тараз","Кызылорда","Усть-Каменогорск"]
STATUSES = ["Зарплатный клиент","Премиальный клиент","Стандартный клиент","Студент"]

# Категории — совпадают с используемыми в твоём коде
CATS_BASE = [
    "Продукты питания","Кафе и рестораны","Кино","Такси",
    "Играем дома","Смотрим дома","Едим дома","АЗС",
    "Путешествия","Отели","Косметика и Парфюмерия","Ювелирные украшения"
]

# Типы переводов — совпадают с твоими ключами расчёта
TR_TYPES_OUT = [
    "card_out","p2p_out","atm_withdrawal","utilities_out",
    "loan_payment_out","cc_repayment_out","installment_payment_out",
    "deposit_topup_out","invest_out","gold_buy_out","fx_buy"
]
TR_TYPES_IN = [
    "card_in","refund_in","cashback_in","salary_in","stipend_in",
    "family_in","invest_in","gold_sell_in","deposit_fx_withdraw_in","fx_sell"
]

CURRENCIES = ["KZT","USD","EUR","RUB"]

# ---------- 10 клиентов (61..70) + профили под твои продукты ----------
# profile: задаёт интересы/сигналы. Ниже прописаны правила генерации.
NEW_CLIENTS = [
    (61,"Ильяс","Зарплатный клиент",33,"Алматы",       365_000,"travel_card"),          # Тревел карта: Путешествия/Такси + USD/EUR
    (62,"Куралай","Премиальный клиент",46,"Астана",   3_450_000,"premium_card"),        # Премиум: высокий баланс + inflows + рестораны/поездки
    (63,"Тлек","Стандартный клиент",38,"Павлодар",     120_000,"cash_loan_stress"),     # Кредит наличными: кассовые разрывы + loan_payment_out
    (64,"Айжан","Студент",21,"Алматы",                  58_000,"credit_card_student"),   # Кредитка: онлайн-категории + cc/instalment
    (65,"Ернат","Зарплатный клиент",29,"Караганда",     780_000,"deposit_savings"),      # Депозит «Сберегательный»: стабильный баланс, низкая волатильность
    (66,"Даурен","Премиальный клиент",50,"Алматы",    6_200_000,"premium_highbalance"),  # Премиалка 2–4% + лидерские категории
    (67,"Бота","Зарплатный клиент",41,"Шымкент",       410_000,"gold_stack"),            # Золото: регулярные gold_buy/gold_sell
    (68,"Рамазан","Стандартный клиент",36,"Кызылорда",  185_000,"fx_regular"),           # Обмен валют: fx_buy/sell + USD/EUR траты
    (69,"Аягоз","Зарплатный клиент",27,"Тараз",         265_000,"deposit_multi"),        # Депозит мультивалютный: FX inflows + мультивалюта
    (70,"Сергей","Премиальный клиент",54,"Усть-Каменогорск", 4_050_000,"investor"),      # Инвестиции: invest_out/in + свободные остатки
]

# ---------- безопасная запись CSV для Windows ----------
def _write_csv_atomic(df: pd.DataFrame, path: Path, retries: int = 8, delay: float = 0.75):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass
    last_err = None
    for _ in range(retries):
        try:
            with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8-sig",
                                             dir=str(path.parent), delete=False) as tmp:
                tmp_name = tmp.name
            df.to_csv(tmp_name, index=False, encoding="utf-8-sig")
            os.replace(tmp_name, str(path))
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
        except Exception as e:
            try:
                if 'tmp_name' in locals() and os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            finally:
                raise
    raise PermissionError(f"Не удалось записать {path}. Закройте файл в Excel/IDE. Оригинальная ошибка: {last_err}")

# ---------- утилиты времени/распределений ----------
def daterange(start: datetime, end: datetime):
    span = (end - start).days
    for d in range(span + 1):
        yield start + timedelta(days=d)

def rand_time_on(day: datetime):
    h = random.randint(8, 21)
    m = random.randint(0, 59)
    s = random.randint(0, 59)
    return day.replace(hour=h, minute=m, second=s)

def wchoice(pairs):
    total = sum(w for _, w in pairs)
    r = random.uniform(0, total) if total > 0 else 0
    upto = 0
    for v, w in pairs:
        if upto + w >= r:
            return v
        upto += w
    return pairs[-1][0] if pairs else None

def amount_in_range(low, high, skew=1.0):
    u = random.random()
    u = u ** (1.0/skew)
    val = low + (high - low) * u
    return round(val, 2)

def kzt_or_fx(prob_fx=0.06):
    if random.random() < prob_fx:
        return random.choice(["USD","EUR","RUB"])
    return "KZT"

def ensure_parents(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# ---------- диапазоны по категориям ----------
BASE_RANGES = {
    "Продукты питания": (4000, 32000),
    "Кафе и рестораны": (2500, 22000),
    "Кино": (1800, 9000),
    "Такси": (2500, 12000),
    "Играем дома": (2000, 9000),
    "Смотрим дома": (1800, 9000),
    "Едим дома": (2000, 8000),
    "АЗС": (12000, 38000),
    "Путешествия": (50000, 380000),
    "Отели": (40000, 300000),
    "Косметика и Парфюмерия": (6000, 50000),
    "Ювелирные украшения": (70000, 700000),
}

# ---------- профили интересов → веса/поведение ----------
def build_tx_weights(profile):
    w = {c: 1 for c in CATS_BASE}

    if profile == "travel_card":
        for k in ["Путешествия","Отели","Такси","Кафе и рестораны"]:
            w[k] = 6
    elif profile == "premium_card":
        for k in ["Путешествия","Отели","Кафе и рестораны","Ювелирные украшения"]:
            w[k] = 5
    elif profile == "cash_loan_stress":
        for k in ["Продукты питания","Кафе и рестораны","АЗС","Такси"]:
            w[k] = 4
    elif profile == "credit_card_student":
        for k in ["Играем дома","Смотрим дома","Едим дома","Такси","Кафе и рестораны","Кино"]:
            w[k] = 6
    elif profile == "deposit_savings":
        for k in ["Продукты питания","Кафе и рестораны"]:
            w[k] = 3
    elif profile == "premium_highbalance":
        for k in ["Ювелирные украшения","Косметика и Парфюмерия","Кафе и рестораны","Путешествия"]:
            w[k] = 5
    elif profile == "gold_stack":
        for k in ["Продукты питания","Кафе и рестораны","АЗС","Такси"]:
            w[k] = 3
    elif profile == "fx_regular":
        for k in ["Продукты питания","Кафе и рестораны","Такси","Путешествия"]:
            w[k] = 4
    elif profile == "deposit_multi":
        for k in ["Продукты питания","Кафе и рестораны","Такси","Отели"]:
            w[k] = 3
    elif profile == "investor":
        for k in ["Кафе и рестораны","Продукты питания","Путешествия","Кино"]:
            w[k] = 3

    pairs = [(c, float(w.get(c,1))) for c in CATS_BASE]
    return pairs

def profile_fx_prob(profile):
    # вероятность трат в валюте в категориях travel/hotels
    if profile in ("travel_card","premium_card","fx_regular","deposit_multi","premium_highbalance","investor"):
        return 0.15
    return 0.08

def daily_tx_target(profile):
    # диапозон количества транзакций в день (для 92 дней суммарно выйдет ~65–85)
    if profile in ("credit_card_student","travel_card","premium_highbalance","premium_card"):
        return (0, 5)
    else:
        return (0, 4)

def tr_events_for_profile(profile, day_idx):
    """
    Набор трансферов под сигналы:
    - FX: fx_buy/fx_sell
    - Премиалка: salary_in/bonus_in, частые inflows
    - Кредитка: cc_repayment_out / installment_payment_out
    - Кредит наличными: loan_payment_out + кассовые разрывы
    - Депозиты: deposit_topup_out / deposit_fx_withdraw_in
    - Инвестиции: invest_out / invest_in
    - Золото: gold_buy_out / gold_sell_in
    - Плюс бытовые card_in/out, p2p_out, utilities_out, atm_withdrawal
    """
    out = []

    # общие нечастые входящие
    if day_idx % 14 == 0:
        out.append(("refund_in","in",(2000, 20000), 0.0))
    if day_idx % 7 == 0:
        out.append(("cashback_in","in",(3000, 25000), 0.0))

    # FX сигналы
    if profile in ("travel_card","premium_card","fx_regular","deposit_multi","premium_highbalance"):
        if day_idx % 11 == 2:
            out.append(("fx_buy","out",(200_000, 1_600_000), 1.0))
        if day_idx % 17 == 6:
            out.append(("fx_sell","in",(150_000, 1_000_000), 1.0))

    # Премиальные inflows
    if profile in ("premium_card","premium_highbalance","investor"):
        if day_idx % 15 in (4, 12):   # зарплата/бонусы
            out.append(("salary_in","in",(900_000, 2_500_000), 0.0))
        if day_idx % 30 == 10:
            out.append(("family_in","in",(150_000, 600_000), 0.0))

    # Кредитка — регулярные платежи
    if profile in ("credit_card_student","premium_card","premium_highbalance"):
        if day_idx % 9 in (1, 5):
            out.append(("cc_repayment_out","out",(40_000, 180_000), 0.0))
        if day_idx % 13 == 7:
            out.append(("installment_payment_out","out",(25_000, 110_000), 0.0))

    # Кредит наличными — loan payments чаще
    if profile in ("cash_loan_stress",):
        if day_idx % 14 == 6:
            out.append(("salary_in","in",(140_000, 350_000), 0.0))
        if day_idx % 9 in (2, 6):
            out.append(("loan_payment_out","out",(35_000, 85_000), 0.0))
        if day_idx % 7 == 2:
            out.append(("cc_repayment_out","out",(40_000, 170_000), 0.0))

    # Депозиты
    if profile in ("deposit_savings", "premium_highbalance"):
        # редкие пополнения сберегательного/срочного (у тебя это "Сберегательный")
        if day_idx % 18 == 8:
            out.append(("deposit_topup_out","out",(150_000, 900_000), 0.0))

    if profile in ("deposit_multi",):
        if day_idx % 16 == 6:
            out.append(("deposit_topup_out","out",(120_000, 700_000), 0.0))
        if day_idx % 22 == 11:
            out.append(("deposit_fx_withdraw_in","in",(120_000, 600_000), 1.0))

    # Инвестиции
    if profile in ("investor","premium_highbalance"):
        if day_idx % 10 == 5:
            out.append(("invest_out","out",(120_000, 1_400_000), 0.0))
        if day_idx % 15 == 10:
            out.append(("invest_in","in",(90_000, 900_000), 0.0))

    # Золото
    if profile in ("gold_stack",):
        if day_idx % 20 == 9:
            out.append(("gold_buy_out","out",(500_000, 2_000_000), 0.0))
        if day_idx % 27 == 12:
            out.append(("gold_sell_in","in",(400_000, 1_800_000), 0.0))

    # бытовое поведение
    if day_idx % 5 == 0:
        out.append(("card_in","in",(3000, 40000), 0.0))
    if day_idx % 3 == 0:
        out.append(("card_out","out",(3000, 50000), 0.0))
    if day_idx % 11 == 4:
        out.append(("p2p_out","out",(3000, 60000), 0.0))
    if day_idx % 8 == 3:
        out.append(("atm_withdrawal","out",(10000, 160000), 0.0))
    if day_idx % 10 == 7:
        out.append(("utilities_out","out",(15000, 70000), 0.0))

    return out

# ---------- генерация по клиенту ----------
def gen_for_client(client, tx_target_range=(65, 85), tr_target_range=(35, 50)):
    cid, name, status, age, city, bal, profile = client

    # TRANSACTIONS (покупки)
    tx_rows = []
    cat_weights = build_tx_weights(profile)
    fx_prob = profile_fx_prob(profile)
    daily_min, daily_max = daily_tx_target(profile)

    # «нагоняем» целевое число транзакций (псевдо-пуассоновский подход)
    days = list(daterange(START, END))
    # сколько транзакций хотим суммарно
    target_tx = random.randint(*tx_target_range)

    # сначала раскидаем по дням
    day_tx_counts = [0 for _ in days]
    # выделим «активные» дни с повышенной активностью (для реализма)
    active_days_idx = random.sample(range(len(days)), k=min(24, len(days)//3))
    for i in range(len(days)):
        base = random.randint(daily_min, daily_max)
        if i in active_days_idx:
            base += random.randint(1, 2)
        day_tx_counts[i] = base

    # нормируем к target_tx
    current_total = sum(day_tx_counts)
    if current_total == 0:
        day_tx_counts[random.randrange(len(days))] = target_tx
    else:
        scale = target_tx / current_total
        day_tx_counts = [max(0, int(round(c * scale))) for c in day_tx_counts]
        # подровняем до точного target_tx
        diff = target_tx - sum(day_tx_counts)
        while diff != 0:
            j = random.randrange(len(day_tx_counts))
            if diff > 0:
                day_tx_counts[j] += 1
                diff -= 1
            else:
                if day_tx_counts[j] > 0:
                    day_tx_counts[j] -= 1
                    diff += 1

    for i, day in enumerate(days):
        n_ops = day_tx_counts[i]
        for _ in range(n_ops):
            cat = wchoice(cat_weights)
            lo, hi = BASE_RANGES[cat]
            skew = 1.25 if cat in ("Путешествия","Отели","Ювелирные украшения") else 0.9
            amount = amount_in_range(lo, hi, skew=skew)
            cur = "KZT"
            if cat in ("Путешествия","Отели") and random.random() < fx_prob:
                cur = random.choice(["USD","EUR"])
            tx_rows.append([
                cid, name, "—", status, city,
                rand_time_on(day).strftime("%Y-%m-%d %H:%M:%S"),
                cat, amount, cur
            ])

    # TRANSFERS (операции/переводы)
    tr_rows = []
    target_tr = random.randint(*tr_target_range)
    # сгенерим набор событий по профилю, затем отнормируем к target_tr
    all_events = []
    for day_idx, day in enumerate(days):
        events = tr_events_for_profile(profile, day_idx)
        for ttype, direction, (lo, hi), fx_p in events:
            all_events.append((day, ttype, direction, lo, hi, fx_p))

    if len(all_events) == 0:
        # гарантируем минимум быта
        for day_idx, day in enumerate(days):
            if day_idx % 3 == 0:
                all_events.append((day, "card_out", "out", 3000, 40000, 0.0))

    # случайно отберём под target_tr (с небольшим «перебором» для вариативности)
    random.shuffle(all_events)
    pick = all_events[:target_tr]

    for (day, ttype, direction, lo, hi, fx_p) in pick:
        amount = amount_in_range(lo, hi, skew=1.0)
        cur = kzt_or_fx(fx_p)
        if ttype in ("gold_buy_out","gold_sell_in","invest_out","invest_in","cc_repayment_out",
                     "loan_payment_out","installment_payment_out","utilities_out","atm_withdrawal",
                     "deposit_topup_out","deposit_fx_withdraw_in","salary_in","stipend_in","cashback_in",
                     "refund_in","p2p_out","card_in","card_out","family_in","fx_sell"):
            # большинство — KZT, кроме настоящих fx_buy (out) и иногда fx_sell (in) можем оставить валюту, но
            # твоя логика FX считает по tr_sum(["fx_buy","fx_sell"]), валюта поля здесь не критична
            if ttype not in ("fx_buy", "fx_sell"):
                cur = "KZT"
        tr_rows.append([
            cid, name, "—", status, city,
            rand_time_on(day).strftime("%Y-%m-%d %H:%M:%S"),
            ttype, "out" if direction=="out" else "in", amount, cur
        ])

    # --- запись файлов ---
    tx_path = CSV_DIR / f"client_{cid}_transactions_3m.csv"
    tr_path = CSV_DIR / f"client_{cid}_transfers_3m.csv"
    ensure_parents(tx_path)
    ensure_parents(tr_path)

    with tx_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["client_code","name","product","status","city","date","category","amount","currency"])
        w.writerows(tx_rows)

    with tr_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["client_code","name","product","status","city","date","type","direction","amount","currency"])
        w.writerows(tr_rows)

    return tx_path, tr_path, len(tx_rows), len(tr_rows)

# ---------- upsert clients.csv ----------
def upsert_clients():
    df = pd.DataFrame(NEW_CLIENTS, columns=[
        "client_code","name","status","age","city","avg_monthly_balance_KZT","_profile"
    ])
    if CLIENTS_PATH.exists():
        exist = pd.read_csv(CLIENTS_PATH)
    else:
        exist = pd.DataFrame(columns=["client_code","name","status","age","city","avg_monthly_balance_KZT"])

    need_cols = ["client_code","name","status","age","city","avg_monthly_balance_KZT"]
    if not exist.empty:
        exist = exist[need_cols]
    to_add = df[need_cols].copy()

    merged = pd.concat([exist, to_add], ignore_index=True)
    merged = merged.drop_duplicates(subset=["client_code"], keep="first").sort_values("client_code")

    _write_csv_atomic(merged, CLIENTS_PATH)

    profiles = {row["client_code"]: prof for row, prof in zip(df[need_cols].to_dict("records"), df["_profile"])}
    return profiles

def main():
    profiles = upsert_clients()
    created = []
    total_tx = total_tr = 0
    for tpl in NEW_CLIENTS:
        cid = tpl[0]
        print(f"Generating CSV for client {cid}...")
        tx_path, tr_path, ntx, ntr = gen_for_client(tpl)
        total_tx += ntx
        total_tr += ntr
        created.append((cid, tx_path.name, tr_path.name, ntx, ntr))
    print("\nDone. Created/updated:")
    for cid, txn, trn, ntx, ntr in created:
        print(f"  {cid}: {txn} ({ntx} tx), {trn} ({ntr} tr)")
    print(f"Totals: {total_tx} transactions, {total_tr} transfers (≈{total_tx+total_tr} ops)")
    print(f"\nclients.csv upserted at: {CLIENTS_PATH}")

if __name__ == "__main__":
    main()
