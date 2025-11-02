import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  # بارگذاری متغیرهای محیطی

# تنظیم کلاینت Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ساخت مدل با پرامپت سیستم (اینجا اضافه می‌شه)
model = genai.GenerativeModel(
    'gemini-2.5-flash',
    system_instruction="تو یک دستیار مفید و باهوش هستی که به زبان فارسی پاسخ می‌دی."  # پرامپت سیستم خودت رو بنویس
)

def get_reply_user(user_text: str) -> str:
    """
    این تابع متن کاربر رو می‌گیره و از Google Gemini پاسخ واقعی می‌گیره.
    """
    try:
        response = model.generate_content(user_text)  # مستقیم متن کاربر رو بده؛ پرامپت سیستم قبلاً تنظیم شده
        return response.text.strip()
    except Exception as e:
        return f"خطا در گرفتن پاسخ: {str(e)}"