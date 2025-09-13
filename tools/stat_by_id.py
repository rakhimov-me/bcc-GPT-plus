# tools/stat_by_id.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import glob

# ---------- настройки / константы ----------
CSV_DIR = Path("csv")
CLIENTS_PATH = CSV_DIR / "clients.csv"
TX_GLOB = str(CSV_DIR / "client_*_transactions_3m.csv")
TR_GLOB = str(CSV_DIR / "client_*_transfers_3m.csv")

# фиксируем "сегодня" для воспроизводимости
TODAY = datetime(2025, 9, 13)
RU_MONTHS = {1:"январе",2:"феврале",3:"марте",4:"апреле",5:"мае",6:"июне",
             7:"июле",8:"августе",9:"сентябре",10:"октябре",11:"ноябре",12:"декабре"}

def fmt_kzt(x: float) -> str:
    x = float(x) if pd.notna(x) else 0.0
    r = int(round(x / 10.0) * 10)  # округляем до десятков для пушей
    return f"{r:,}".replace(",", " ") + " ₸"

# ---------- чтение данных ----------
def load_all():
    clients = pd.read_csv(CLIENTS_PATH)
    tx_files = sorted(glob.glob(TX_GLOB))
    tr_files = sorted(glob.glob(TR_GLOB))
    if not tx_files:
        raise FileNotFoundError(f"Не найдены транзакции по маске: {TX_GLOB}")
    if not tr_files:
        raise FileNotFoundError(f"Не найдены переводы по маске: {TR_GLOB}")

    transactions = pd.concat((pd.read_csv(p) for p in tx_files), ignore_index=True)
    transfers = pd.concat((pd.read_csv(p) for p in tr_files), ignore_index=True)

    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx["month"] = tx["date"].dt.to_period("M")

    tr = transfers.copy()
    tr["date"] = pd.to_datetime(tr["date"])
    tr["month"] = tr["date"].dt.to_period("M")
    return clients, tx, tr

# ---------- агрегаты / хелперы ----------
def make_helpers(clients, tx, tr):
    last_month_dt = TODAY.replace(day=1) - timedelta(days=1)
    last_month = last_month_dt.month

    spend = tx.groupby(["client_code","category"], as_index=False)["amount"] \
              .sum().rename(columns={"amount":"sum_spend"})
    spend_lastm = tx[tx["date"].dt.month==last_month] \
        .groupby(["client_code","category"], as_index=False)["amount"] \
        .sum().rename(columns={"amount":"sum_spend_lastm"})

    tr_types = tr.groupby(["client_code","type"], as_index=False)["amount"] \
                 .sum().rename(columns={"amount":"sum_amount"})
    tr_count = tr.groupby(["client_code","type"], as_index=False) \
                 .size().rename(columns={"size":"count"})

    fx_spend_count = tx[tx.currency.ne("KZT")].groupby("client_code") \
                     .size().rename("fx_tx_count")
    fx_spend_count = fx_spend_count.reindex(clients.client_code).fillna(0).astype(int)

    CATEGORIES = tx["category"].dropna().unique().tolist()

    def spend_in(client_id, cats, use_lastm=False):
        if use_lastm:
            s = spend_lastm[(spend_lastm.client_code==client_id) &
                            (spend_lastm.category.isin(cats))]["sum_spend_lastm"].sum()
        else:
            s = spend[(spend.client_code==client_id) &
                      (spend.category.isin(cats))]["sum_spend"].sum()
        return float(s) if pd.notna(s) else 0.0

    def tr_sum(client_id, types):
        s = tr_types[(tr_types.client_code==client_id) &
                     (tr_types.type.isin(types))]["sum_amount"].sum()
        return float(s) if pd.notna(s) else 0.0

    def tr_cnt(client_id, types):
        c = tr_count[(tr_count.client_code==client_id) &
                     (tr_count.type.isin(types))]["count"].sum()
        return int(c) if pd.notna(c) else 0

    return {
        "last_month_dt": last_month_dt,
        "last_month": last_month,
        "spend_in": spend_in,
        "tr_sum": tr_sum,
        "tr_cnt": tr_cnt,
        "fx_spend_count": fx_spend_count,
        "CATEGORIES": CATEGORIES,
        "spend_lastm": spend_lastm,
        "tx": tx
    }

# ---------- расчёт выгод + подробности для одного клиента ----------
def compute_scores_for_client(row: pd.Series, H: dict) -> dict:
    cid = int(row["client_code"])
    bal = float(row["avg_monthly_balance_KZT"])

    # сигналы за последний месяц
    taxis = H["spend_in"](cid, ["Такси"], use_lastm=True)
    trips = H["spend_in"](cid, ["Путешествия","Отели"], use_lastm=True)
    rest  = H["spend_in"](cid, ["Кафе и рестораны"], use_lastm=True)
    boosted = H["spend_in"](cid, ["Ювелирные украшения","Косметика и Парфюмерия","Кафе и рестораны"], use_lastm=True)
    online = H["spend_in"](cid, ["Играем дома","Смотрим дома","Едим дома"], use_lastm=True)
    all_lastm = H["spend_in"](cid, H["CATEGORIES"], use_lastm=True)

    fx_ops = H["tr_cnt"](cid, ["fx_buy","fx_sell"])
    fx_turn = H["tr_sum"](cid, ["fx_buy","fx_sell"])
    fx_card_tx = int(H["fx_spend_count"].loc[cid])

    stress_cnt = H["tr_cnt"](cid, ["cc_repayment_out","loan_payment_out","installment_payment_out"])
    deposit_topups = H["tr_sum"](cid, ["deposit_topup_out"])
    fx_dep_sum = H["tr_sum"](cid, ["deposit_fx_topup_out","deposit_fx_withdraw_in"])
    g_turn = H["tr_sum"](cid, ["gold_buy_out","gold_sell_in"])

    multi_curr = H["tx"][H["tx"].client_code==cid]["currency"].nunique() >= 2

    cl_last = H["spend_lastm"][H["spend_lastm"].client_code==cid] \
                .sort_values("sum_spend_lastm", ascending=False)
    top3_cats = cl_last.category.head(3).tolist()

    # --- формулы выгод ---
    # 1) Карта для путешествий: 4% от (Путешествия + Такси + Отели)
    travel_base = taxis + trips
    score_travel = 0.04 * travel_base

    # 2) Премиальная карта: базовый % по балансу + 4% на boosted; кэп 100 000 ₸/мес
    rate = 0.04 if bal>=6_000_000 else (0.03 if bal>=1_000_000 else 0.02)
    other_spend = max(all_lastm - boosted, 0)
    premium_raw = 0.04 * boosted + rate * other_spend
    score_premium = min(premium_raw, 100_000)

    # 3) Кредитная карта: 10% на топ-3 категории + 10% на онлайн-сервисы
    score_credit = 0.10 * (H["spend_in"](cid, top3_cats, use_lastm=True) + online)

    # 4) Обмен валют: 0.15% от оборота + 500 ₸ за событие (до 5), события = fx_ops + non-KZT транзакции
    fx_events = fx_ops + fx_card_tx
    score_fx = 0.0015 * fx_turn + 500 * min(fx_events, 5)

    # 5) Кредит наличными: при низком балансе и наличии "стресса" по долгам
    score_cash = 15_000 + 2_000*(stress_cnt-2) if (bal < 200_000 and stress_cnt >= 3) else 0.0

    # 6) Депозиты
    score_dep_save = (0.165/12) * bal if (bal>=1_500_000 and all_lastm < 0.3*bal) else 0.0
    score_dep_acc  = bal*(0.155/12)*0.5 if (bal>=500_000 or deposit_topups>0) else 0.0
    if multi_curr or fx_dep_sum>0:
        alloc = max(bal*0.3, fx_dep_sum*0.3)
        score_dep_multi = alloc*(0.145/12)
    else:
        score_dep_multi = 0.0

    # 7) Инвестиции: базовая польза + активность
    inv_turn = H["tr_sum"](cid, ["invest_out","invest_in"])
    score_inv = 5000 + 0.002*inv_turn if (inv_turn>0 or row["status"] in ["Студент","Зарплатный клиент"]) else 0.0

    # 8) Золотые слитки: только если были операции
    score_gold = 3000 + 0.001*g_turn if g_turn>0 else 0.0

    scores = {
        "Карта для путешествий": score_travel,
        "Премиальная карта": score_premium,
        "Кредитная карта": score_credit,
        "Обмен валют": score_fx,
        "Кредит наличными": score_cash,
        "Депозит Мультивалютный": score_dep_multi,
        "Депозит Сберегательный": score_dep_save,
        "Депозит Накопительный": score_dep_acc,
        "Инвестиции": score_inv,
        "Золотые слитки": score_gold,
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = [p for p,_ in ranked[:4]]

    # пуш (по победителю)
    mname = RU_MONTHS[H["last_month_dt"].month]
    fx_curr = next((c for c in ["USD","EUR","RUB"]
                    if c in H["tx"][H["tx"].client_code==cid]["currency"].unique()), "USD")
    best_prod, best_benefit = ranked[0]
    if best_prod == "Карта для путешествий":
        push = f"{row['name']}, в {mname} у вас поездок и такси на {fmt_kzt(travel_base)}. С тревел-картой вернули бы ≈{fmt_kzt(best_benefit)}. Оформить карту."
    elif best_prod == "Премиальная карта":
        push = f"{row['name']}, у вас крупный остаток ({fmt_kzt(bal)}). Премиальная карта даст повышенный кешбэк и бесплатные снятия — выгода ≈{fmt_kzt(best_benefit)}/мес. Оформить сейчас."
    elif best_prod == "Кредитная карта":
        cats_txt = ", ".join(top3_cats) if top3_cats else "ваших категориях"
        push = f"{row['name']}, ваши топ-категории — {cats_txt}. Кредитная карта даёт до 10% и на онлайн-сервисы. Потенциальная выгода ≈{fmt_kzt(best_benefit)}/мес. Оформить карту."
    elif best_prod == "Обмен валют":
        push = f"{row['name']}, вы часто платите в {fx_curr} и меняете валюту ({fx_ops} операции в {mname}). В приложении — выгодный курс и авто-покупка. Настроить обмен."
    elif best_prod == "Кредит наличными":
        push = f"{row['name']}, если нужен запас на крупные траты — можно оформить кредит наличными с гибкими выплатами. Узнать доступный лимит."
    elif best_prod == "Депозит Мультивалютный":
        push = f"{row['name']}, часть средств можно держать в {fx_curr}. Мультивалютный вклад — {fmt_kzt(best_benefit)} в месяц при текущем балансе. Открыть вклад."
    elif best_prod == "Депозит Сберегательный":
        push = f"{row['name']}, свободные {fmt_kzt(bal)} могут работать под 16,5%. Доходность ≈{fmt_kzt(best_benefit)}/мес при размещении. Открыть вклад."
    elif best_prod == "Депозит Накопительный":
        push = f"{row['name']}, удобно откладывать под 15,5% без снятий. С вашим балансом выгода ≈{fmt_kzt(best_benefit)}/мес. Открыть вклад."
    elif best_prod == "Инвестиции":
        push = f"{row['name']}, попробуйте инвестиции: без комиссий на старт и порог от 6 ₸. Начните с малого — счёт откроется за пару минут. Открыть счёт."
    elif best_prod == "Золотые слитки":
        push = f"{row['name']}, для диверсификации — золотые слитки 999,9. Покупка с предзаказом в приложении. Посмотреть варианты."
    else:
        push = f"{row['name']}, посмотрите подходящий продукт — это может дать вам ощутимую выгоду. Посмотреть."

    # итоговая запись с полным разложением
    return {
        # профиль
        "client_code": cid,
        "name": row["name"],
        "status": row["status"],
        "age": int(row["age"]),
        "city": row.get("city", ""),
        "avg_monthly_balance_KZT": int(bal),

        # ключевые сигналы (последний месяц)
        "context_month": mname,
        "spend_taxi_lastm": round(taxis, 2),
        "spend_travel_hotels_lastm": round(trips, 2),
        "spend_rest_lastm": round(rest, 2),
        "spend_online_lastm": round(online, 2),
        "spend_boosted_lastm": round(boosted, 2),
        "spend_all_lastm": round(all_lastm, 2),
        "fx_ops_count": int(fx_ops),
        "fx_turnover": round(fx_turn, 2),
        "fx_card_nonKZT_tx_count": int(fx_card_tx),
        "loan_stress_count": int(stress_cnt),
        "deposit_topups_sum": round(deposit_topups, 2),
        "fx_deposit_sum": round(fx_dep_sum, 2),
        "gold_turnover": round(g_turn, 2),
        "multi_currency_flag": bool(multi_curr),
        "top3_categories_lastm": ", ".join(top3_cats) if top3_cats else "",

        # детали премиум-формулы
        "premium_rate": rate,
        "premium_boosted_spend": round(boosted, 2),
        "premium_other_spend": round(other_spend, 2),
        "premium_raw_uncapped": round(premium_raw, 2),
        "premium_cap_applied": premium_raw > 100_000,

        # финальные скоры по продуктам
        **{f"score_{k}": round(v, 2) for k, v in scores.items()},

        # топ-4 + победитель
        "rank1": top[0] if len(top) > 0 else "",
        "rank2": top[1] if len(top) > 1 else "",
        "rank3": top[2] if len(top) > 2 else "",
        "rank4": top[3] if len(top) > 3 else "",
        "winner_product": best_prod,
        "winner_benefit_fmt": fmt_kzt(best_benefit),
        "push_notification": push,
    }

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Подробная статистика и расчёты по клиентам (несколько id)")
    parser.add_argument("--ids", type=int, nargs="+", required=True,
                        help="client_code клиентов (через пробел)")
    parser.add_argument("--out", type=str, default="stat-by-id.csv",
                        help="путь к общему CSV (по умолчанию stat-by-id.csv)")
    parser.add_argument("--excel", action="store_true",
                        help="Excel-friendly CSV (UTF-8-SIG + ; + десятичная запятая)")
    args = parser.parse_args()

    clients, tx, tr = load_all()
    H = make_helpers(clients, tx, tr)

    valid_ids = set(clients["client_code"].astype(int))
    results = []
    for cid in args.ids:
        if cid not in valid_ids:
            print(f"[WARN] client_code={cid} не найден в clients.csv — пропускаем")
            continue
        row = clients[clients["client_code"].astype(int) == cid].iloc[0]
        results.append(compute_scores_for_client(row, H))

    if not results:
        raise SystemExit("Нет валидных клиентов для обработки.")

    df = pd.DataFrame(results)

    df.to_csv(
        args.out,
        index=False,
        encoding="utf-8-sig" if args.excel else "utf-8-sig",
        sep=";" if args.excel else ",",
        decimal="," if args.excel else ".",
        float_format="%.2f"
    )
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
