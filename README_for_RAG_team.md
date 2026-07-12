# Buy-Signal Model API — دليل تيم الـ RAG

## نظرة عامة
الـ API ده بيرجع **احتمالية إن السهم يطلع أكتر من 5% خلال 10 أيام تداول جاية**، بناءً على موديل XGBoost اتدرب على بيانات تاريخية (2010–2026) لأربعين سهم من مؤشرات مختلفة.

---

## Endpoint

```
POST /predict
Content-Type: application/json
```

### Request (اللي تيم الـ RAG لازم يبعته)

```json
{
  "tickers": ["AAPL", "MSFT", "TSLA"]
}
```

- `tickers`: **List[string]** — رموز الأسهم اللي عايزين تقييمها (لازم تكون رموز Yahoo Finance صحيحة، زي `BRK-B` مش `BRK.B`).
- مفيش حاجة تانية مطلوبة من تيم الـ RAG — الـ API بيجيب البيانات التاريخية وحساب الـ features لوحده.

### Response (اللي هيرجعله الموديل)

```json
{
  "predictions": [
    {
      "ticker": "AAPL",
      "sector": 0,
      "date": "2026-07-08",
      "buy_probability": 0.6832,
      "conviction": "moderate"
    }
  ],
  "model_version": "phase2_original",
  "threshold_high_conviction": 0.71,
  "threshold_moderate_conviction": 0.58,
  "notes": "buy_probability estimates the likelihood the stock rises more than 5% within the next 10 trading days. This is a probabilistic signal, not a guarantee. Model Test AUC ~0.63 — treat as one input among many, not a standalone investment decision."
}
```

### شرح كل حقل

| الحقل | المعنى |
|---|---|
| `ticker` | رمز السهم |
| `sector` | رقم القطاع الداخلي (0=Tech, 1=Consumer Disc., 2=Financials, 3=Healthcare, 4=Staples, 5=Industrials, 6=Energy, 7=Other) |
| `date` | آخر يوم تداول استُخدمت بياناته في التنبؤ |
| `buy_probability` | رقم من 0 لـ 1 — احتمالية الحركة الصاعدة (>5% خلال 10 أيام) |
| `conviction` | `"high"` / `"moderate"` / `"none"` — تصنيف جاهز بناءً على thresholds اتحسبت وقت التدريب لموازنة الدقة والتغطية |

---

## إزاي تيم الـ RAG يستخدم الأرقام دي في نصيحته للمستثمر

توصية عملية:
- **`conviction = "high"`** → precision أعلى، إشارات أقل عددًا لكن أدق. مناسب لصياغة "توصية قوية نسبيًا"
- **`conviction = "moderate"`** → دقة أقل، إشارات أكتر. مناسب لصياغة "إشارة تستحق المتابعة" مش "توصية قوية"
- **`conviction = "none"`** → الموديل مالوش رأي واضح — منصحش نصدر أي توصية شراء بناءً عليه

### ⚠️ تحذيرات لازم تتقال للمستثمر (مهم قانونيًا وأخلاقيًا)
1. **الموديل مش مثالي** — دقته (Test AUC) حوالي 0.63 من أصل 1.0 (يعني أفضل من العشوائية بوضوح، لكن مش دقة عالية).
2. **الأداء التاريخي مش ضمان للمستقبل** — الأرقام دي من backtest على فترة معينة، والأسواق بتتغير.
3. **المفروض دايمًا تتقال جملة تحذيرية** زي: *"هذه إشارة إحصائية وليست نصيحة استثمارية مضمونة، ويُنصح بمراجعة مستشار مالي مرخّص قبل اتخاذ أي قرار."*
4. الموديل **مبيقولش "بيع"** — بس بيدي احتمالية شراء. أي قرار بيع لازم يكون معتمد على منطق تاني (مش من الـ endpoint ده).

---

## حدود تقنية يعرفها تيم الـ RAG
- الـ API بيحتاج على الأقل ~400 يوم تداول من التاريخ لكل سهم عشان يحسب كل الـ features (فيه SMA 200 يوم مثلاً) — لو السهم جديد في البورصة (IPO حديث)، ممكن يرجع خطأ 422.
- زمن الاستجابة بيعتمد على عدد الأسهم المطلوبة (بيجيب بيانات لايف من Yahoo Finance في كل طلب) — يُفضّل تعمل caching لو بتنده على نفس الأسهم بشكل متكرر خلال نفس اليوم.
- لو حصل Timeout أو الرمز غلط، هيرجع HTTP error code واضح (400/404/422).

---

## ⚠️ مهم جدًا: الأسهم المدعومة فقط (Whitelist)

الموديل **مدرّب فقط على 40 سهم أمريكي كبير** موضحين تحت. **قبل ما تبعتوا أي طلب للموديل، لازم تتأكدوا إن رمز السهم موجود في القايمة دي.**

لو المستثمر سأل عن سهم مش موجود في القايمة (سهم مصري، أوروبي، شركة صغيرة، أو حتى شركة أمريكية كبيرة لكن مش من القايمة دي)، **متستخدموش الموديل خالص** — بدل كده، الرد المناسب يكون: *"هذا الموديل متخصص حاليًا في مجموعة محددة من الأسهم الأمريكية الكبرى، ولا يغطي هذا السهم."*

```
AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ,
WMT, PG, MA, HD, UNH, CVX, BAC, LLY, ABBV, XOM,
BRK-B, MRK, PEP, COST, AVGO, TMO, ACN, MCD, CSCO, ABT,
CRM, NFLX, AMD, INTC, QCOM, TXN, PM, UPS, CAT, DE
```

**ملحوظة على الرمز `BRK-B`:** ده الرمز الصحيح لسهم Berkshire Hathaway Class B على Yahoo Finance (بشرطة، مش نقطة زي `BRK.B`).

---

## كيفية التشغيل (Setup Instructions)

1. تأكدوا إن عندكم Python 3.12 مُثبّت
2. من داخل مجلد المشروع، شغّلوا:
   ```
   pip install -r requirements_actual.txt
   ```
3. شغّلوا السيرفر:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
4. اختبروا إنه شغال عن طريق فتح المتصفح على:
   ```
   http://localhost:8000/health
   ```
5. للتوثيق التفاعلي وتجربة الـ endpoints:
   ```
   http://localhost:8000/docs
   ```


