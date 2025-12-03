import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .orchestrator import get_reply_user
import base64
import io
import struct

# --- وارد کردن ابزارهای Gemini API ---
import google.generativeai as genai
# نیازی به وارد کردن این موارد نیست مگر برای تنظیمات پیشرفته
# from google.generativeai.types import HarmCategory, HarmBlockThreshold 

load_dotenv()

# تعریف مدل داده برای درخواست‌ها
class UserMessage(BaseModel):
    user_message: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "Kore" # صدای پیش‌فرض

class SummarizeRequest(BaseModel):
    text_to_summarize: str

# --- متغیرهای جهانی و تنظیمات API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
tts_model_client = None
chat_model_client = None
SETUP_ERROR = None

if GEMINI_API_KEY:
    try:
        # تنظیم کلاینت Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        # مدل‌ها را به عنوان کلاینت‌های عمومی تعریف می‌کنیم
        tts_model_client = genai.GenerativeModel('gemini-2.5-flash-preview-tts')
        chat_model_client = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Gemini API clients initialized successfully.")
    except Exception as e:
        print(f"⚠️ خطا در تنظیم Gemini API: {str(e)}")
        SETUP_ERROR = f"خطا در تنظیمات اولیه مدل: {str(e)}"
else:
    SETUP_ERROR = "کلید API (GEMINI_API_KEY) در فایل‌های محیطی (مثل .env) یافت نشد."
    print(f"⚠️ {SETUP_ERROR}")

# ----------------------------------------------------------------------
# توابع کمکی تبدیل PCM به WAV
# ----------------------------------------------------------------------

def pcm_to_wav(pcm_data: bytes, sample_rate=24000):
    """
    داده خام PCM را به فرمت فایل WAV تبدیل می‌کند (24kHz, 16-bit, Mono).
    """
    wav_file = io.BytesIO()
    num_channels = 1
    sample_width = 2
    byte_rate = sample_rate * num_channels * sample_width
    data_size = len(pcm_data)
    
    # Write WAV Header (RIFF Chunk)
    wav_file.write(b'RIFF') 
    wav_file.write(struct.pack('<I', 36 + data_size)) 
    wav_file.write(b'WAVE') 

    # Write FMT Sub-chunk
    wav_file.write(b'fmt ') 
    wav_file.write(struct.pack('<I', 16))
    wav_file.write(struct.pack('<H', 1))
    wav_file.write(struct.pack('<H', num_channels))
    wav_file.write(struct.pack('<I', sample_rate))
    wav_file.write(struct.pack('<I', byte_rate))
    wav_file.write(struct.pack('<H', num_channels * sample_width))
    wav_file.write(struct.pack('<H', sample_width * 8))

    # Write DATA Sub-chunk
    wav_file.write(b'data') 
    wav_file.write(struct.pack('<I', data_size))
    wav_file.write(pcm_data) 

    wav_file.seek(0)
    return wav_file.read()

# ----------------------------------------------------------------------
# سرویس‌های FastAPI
# ----------------------------------------------------------------------

app = FastAPI()

# اضافه کردن CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ۱. سرویس‌دهی روت اصلی (/)
app.mount("/static_files", StaticFiles(directory="frontend"), name="frontend_static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Frontend file not found.")

# ۲. روت API برای دریافت پاسخ چت
@app.post("/reply")
def reply(data: UserMessage):
    # این روت از orchestrator.py استفاده می‌کند
    user_text = data.user_message
    response_text = get_reply_user(user_text)
    return {"response": response_text}

# ۳. روت API جدید برای TTS (تبدیل متن به گفتار)
from fastapi.responses import StreamingResponse

@app.post("/tts")
async def generate_tts_stream(data: TTSRequest):
    global tts_model_client, SETUP_ERROR

    if not tts_model_client or SETUP_ERROR:
        raise HTTPException(status_code=500, detail=str(SETUP_ERROR))

    text_to_speak = data.text[:300]

    try:
        response = tts_model_client.generate_content(
            contents=[text_to_speak],
            generation_config={
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": data.voice
                        }
                    }
                }
            },
            stream=True
        )

        sample_rate = 24000
        num_channels = 1
        sample_width = 2

        header_sent = False
        total_pcm = 0

        async def audio_stream():
            nonlocal header_sent, total_pcm
            for chunk in response:
                if not chunk or not chunk.candidates:
                    continue

                part = chunk.candidates[0].content.parts[0]
                if not hasattr(part, "inline_data"):
                    continue

                pcm = base64.b64decode(part.inline_data.data)
                total_pcm += len(pcm)

                if not header_sent:
                    # WAV header only once!
                    import struct
                    wav_header = b'RIFF' + struct.pack('<I', 36 + 99999999) + b'WAVE'
                    wav_header += b'fmt ' + struct.pack('<I', 16)
                    wav_header += struct.pack('<H', 1)
                    wav_header += struct.pack('<H', num_channels)
                    wav_header += struct.pack('<I', sample_rate)
                    wav_header += struct.pack('<I', sample_rate * num_channels * sample_width)
                    wav_header += struct.pack('<H', num_channels * sample_width)
                    wav_header += struct.pack('<H', sample_width * 8)
                    wav_header += b'data' + struct.pack('<I', 99999999)

                    yield wav_header
                    header_sent = True

                yield pcm  

        return StreamingResponse(audio_stream(), media_type="audio/wav",
                                 headers={
                                     "Transfer-Encoding": "chunked",
                                     "Connection": "keep-alive"
                                 })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Stream failed: {e}")


# ۴. روت API جدید برای خلاصه‌سازی متن
@app.post("/summarize")
async def summarize_text(data: SummarizeRequest):
    global chat_model_client, SETUP_ERROR
    
    if chat_model_client is None or SETUP_ERROR:
        raise HTTPException(status_code=500, detail=f"Summarization model setup failed: {SETUP_ERROR}")

    text_to_summarize = data.text_to_summarize
    
    # ساخت پرامپت خلاصه‌سازی
    summary_prompt = (
        "متن زیر را به صورت مختصر و در حد یک پاراگراف، به زبان فارسی خلاصه کن:\n\n"
        f"متن: \"{text_to_summarize}\""
    )

    try:
        response = chat_model_client.generate_content(summary_prompt, tools=[])
        
        # بررسی پاسخ‌های مسدود شده توسط سیستم ایمنی
        if response.candidates and response.candidates[0].finish_reason.name == 'SAFETY':
             return JSONResponse(
                 content={"summary": "⚠️ به دلیل خط‌مشی‌های ایمنی، امکان خلاصه‌سازی این متن وجود ندارد."},
                 media_type="application/json"
             )
        
        summary_text = response.text.strip()
        return JSONResponse(
            content={"summary": summary_text},
            media_type="application/json"
        )
        
    except Exception as e:
        print(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")