<div dir="rtl">

# تمرین ۱: طراحی و تحلیل شبکه عصبی کانولوشن روی دیتاست BCCD

طبقه‌بندی ۴ نوع گلبول سفید خون (Eosinophil, Lymphocyte, Monocyte, Neutrophil) با یک CNN که از صفر با PyTorch نوشته شده است
(بدون Transfer Learning). تصاویر به ۶۴×۶۴ کوچک شده‌اند و کل آموزش روی CPU اجرا می‌شود.

## فایل‌ها

| فایل | توضیح |
|---|---|
| `ADL_HW1_BCCD_CNN.ipynb` | نوت‌بوک اصلی؛ کامنت‌گذاری شده و همه سلول‌ها **اجرا شده‌اند** (فایل تحویلی ۱) |
| `report/ADL_HW1_Report.pdf` | گزارش ۴ صفحه‌ای (فایل تحویلی ۲) |
| `report/ADL_HW1_Report.docx` | نسخه Word همان گزارش (قابل ویرایش؛ فونت Tahoma) |
| `report/report.html` | سورس گزارش؛ برای ویرایش نام و متن گزارش |
| `best_model.pt` | وزن‌های بهترین مدل روی Validation (خروجی Checkpoint) |
| `outputs/` | نمودارها، ماتریس درهم‌ریختگی، نقشه‌های ویژگی و جدول‌ها (CSV) |
| `requirements.txt` | کتابخانه‌های لازم |

## اجرای مرحله به مرحله

۱. **نصب Python 3.10 یا جدیدتر** و سپس نصب کتابخانه‌ها در پوشه پروژه:

```bash
pip install -r requirements.txt
```

۲. **دیتاست:** پوشه `dataset2-master` باید کنار نوت‌بوک باشد و ساختار آن این‌طور باشد:

```
dataset2-master/images/TRAIN/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/*.jpeg
dataset2-master/images/TEST/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/*.jpeg
```

(پوشه `__MACOSX` فایل اضافه‌ی سیستم‌عامل مک است و استفاده نمی‌شود.)

۳. **اجرای نوت‌بوک:** با Jupyter باز کنید و از منوی `Kernel → Restart & Run All` همه سلول‌ها را اجرا کنید؛ یا در ترمینال:

```bash
jupyter nbconvert --to notebook --execute --inplace ADL_HW1_BCCD_CNN.ipynb
```

اجرا روی یک CPU چهار هسته‌ای حدود ۱۱ دقیقه طول می‌کشد (هر اپوک حدود ۴۰ ثانیه). در Google Colab هم کافی است پوشه دیتاست را کنار نوت‌بوک آپلود کنید.

۴. **ساخت دوباره PDF گزارش (اختیاری):** اگر نتایج شما بعد از اجرای دوباره کمی تغییر کرد، اعداد را در `report/report.html` به‌روز کنید،
فایل را در مرورگر (Chrome) باز کنید و با `Ctrl+P → Save as PDF` (کاغذ A4، بدون Header/Footer، با Background graphics) خروجی بگیرید.
**نام و شماره دانشجویی** خود را هم در بالای گزارش وارد کنید.

۵. **تحویل:** فایل `ADL_HW1_BCCD_CNN.ipynb` (اجرا شده) و فایل PDF گزارش.

## نکات طراحی

- **تقسیم داده:** همه تصاویر TRAIN و TEST جمع و به نسبت ۷۰٪/۱۵٪/۱۵٪ تقسیم شده‌اند. چون تصاویر این دیتاست از قبل Augment شده‌اند
  (از هر تصویر میکروسکوپی اصلی ده‌ها نسخه چرخیده/قرینه وجود دارد)، تقسیم بر اساس **تصویر اصلی** انجام می‌شود تا نسخه‌های یک تصویر هم‌زمان
  در Train و Test نباشند (نشت داده). با `SPLIT_MODE = "random"` در سلول تنظیمات می‌توانید تقسیم تصادفی ساده را هم امتحان کنید؛
  در آن حالت دقت تست بالاتر می‌شود اما به دلیل نشت داده واقعی نیست.
- **پیش‌پردازش:** تغییر اندازه به ۶۴×۶۴، نرمال‌سازی به [−1, 1]؛ Augmentation: چرخش ±۲۰°، قرینه افقی/عمودی، RandomResizedCrop، جابه‌جایی و ColorJitter.
- **معماری:** ۴ بلوک `Conv2d → BatchNorm → ReLU → MaxPool`، سپس `GlobalAvgPool → Dropout → FC(128) → ReLU → Dropout → FC(4)` و تابع زیان CrossEntropy (۴۲۴ هزار پارامتر).
- **آموزش:** Adam با lr=0.001 و weight decay=1e-4، زمان‌بند Cosine، Batch=64، حداکثر ۱۵ اپوک، ذخیره بهترین مدل بر اساس Validation loss و Early Stopping با patience=5.

## نتایج

| معیار | مقدار |
|---|---|
| دقت Test | **98.0%** |
| Macro F1 روی Test | 0.980 |
| بهترین Val Acc (اپوک ۱۵) | 97.3% |
| F1 کلاس‌ها (Eos / Lym / Mon / Neu) | 0.961 / 0.999 / 0.999 / 0.962 |

بیشتر خطاها بین Eosinophil و Neutrophil است (هر دو گرانولوسیت با هسته چندلوبی). نسخه اولیه مدل (ورودی [0,1] و Flatten→FC بزرگ)
فقط به دقت ۷۵٪ رسید و Validation آن بسیار ناپایدار بود؛ روند بهبود در بخش ۳.۱ نوت‌بوک و بخش ۳ گزارش آمده است.

</div>
