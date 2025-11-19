import json
import re
import shlex
import unicodedata
import difflib
from tabulate import tabulate
from datetime import datetime, timezone, timedelta
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes, CallbackQueryHandler, \
    JobQueue
import os
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
import qrcode
from io import BytesIO
import requests
from urllib.parse import urlparse

# تحميل البيئة
load_dotenv()


# ============ الإعدادات الأساسية ============
class Config:
    TOKEN = os.getenv("BOT_TOKEN")
    ADMINS = list(map(int, os.getenv("ADMINS").split(","))) if os.getenv("ADMINS") else []
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # مسارات الملفات
    USERS_FILE = "users.json"
    COURSES_FILE = "courses.json"
    DB_FILE = "database.json"
    REPLIES_FILE = "replies.json"
    QUESTIONS_FILE = "new_questions.json"
    TEXTS_FILE = "texts.json"
    SCHEDULE_FILE = "schedule.json"


# ============ إدارة الملفات ============
class FileManager:
    @staticmethod
    def safe_load_json(path: str, default=None):
        """تحميل ملف JSON بشكل آمن"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            print(f"File error {path}: {e}")
            if default is not None:
                FileManager.safe_save_json(path, default)
            return default

    @staticmethod
    def safe_save_json(path: str, data):
        """حفظ بيانات إلى ملف JSON بشكل آمن"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"خطأ في الحفظ: {e}")
            return False

    @staticmethod
    def append_to_json(path: str, entry):
        """إضافة مدخل جديدة إلى ملف JSON"""
        data = FileManager.safe_load_json(path, [])
        data.append(entry)
        return FileManager.safe_save_json(path, data)


# ============ تحميل البيانات ============
def load_all_data():
    """تحميل جميع البيانات مرة واحدة"""
    return {
        'users': FileManager.safe_load_json(Config.USERS_FILE, []),
        'courses': FileManager.safe_load_json(Config.COURSES_FILE, {}),
        'db': FileManager.safe_load_json(Config.DB_FILE, {}),
        'replies_db': FileManager.safe_load_json(Config.REPLIES_FILE, {}),
        'texts': FileManager.safe_load_json(Config.TEXTS_FILE, {}),
        'schedule': FileManager.safe_load_json(Config.SCHEDULE_FILE, {})
    }


# تحميل البيانات العالمية
data = load_all_data()
users = data['users']
courses = data['courses']
db = data['db']
replies_db = data['replies_db']
texts = data['texts']
schedule = data['schedule']

pdf_texts = texts.get("pdf_texts", {})


# ============ أدوات النصوص ============
class TextUtils:
    @staticmethod
    def normalize_arabic(text: str) -> str:
        """تطبيع النص العربي"""
        text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def search_summary(query: str, database: dict) -> dict:
        """الباحث في قاعدة البيانات"""
        q = TextUtils.normalize_arabic(query)
        best_match = None
        highest_score = 0

        for key, value in database.items():
            if isinstance(value, dict):
                options = [key] + value.get("aliases", [])
            else:
                options = [key]

            for opt in options:
                score = difflib.SequenceMatcher(None, q, TextUtils.normalize_arabic(opt)).ratio()
                if score > highest_score:
                    highest_score = score
                    best_match = value

        return best_match if highest_score >= 0.6 else None

    @staticmethod
    def interactive_replies(text: str) -> str:
        """الباحث في الردود التفاعلية"""
        text_norm = TextUtils.normalize_arabic(text)
        best_match = None
        highest_score = 0

        for key, reply in replies_db.items():
            score = difflib.SequenceMatcher(None, text_norm, TextUtils.normalize_arabic(key)).ratio()
            if score > highest_score:
                highest_score = score
                best_match = reply

        return best_match if highest_score >= 0.6 else None


# ============ إدارة المستخدمين ============
class UserManager:
    @staticmethod
    def find_user(user_id: int):
        """البحث عن مستخدم"""
        return next((u for u in users if u.get("user_id") == user_id), None)

    @staticmethod
    def get_user_and_index(user_id: int):
        """الحصول على المستخدم ومؤشره"""
        for i, u in enumerate(users):
            if u.get("user_id") == user_id:
                return u, i
        return None, None

    @staticmethod
    def save_users():
        """حفظ بيانات المستخدمين - كل طالب في سطر منفصل"""
        try:
            with open(Config.USERS_FILE, "w", encoding="utf-8") as f:
                f.write('[\n')  # فتح المصفوفة
                for i, user in enumerate(users):
                    if i > 0:
                        f.write(',\n')  # فاصلة بين المستخدمين
                    # تحويل المستخدم إلى JSON في سطر واحد مع مسافة بادئة
                    user_json = json.dumps(user, ensure_ascii=False)
                    f.write('  ' + user_json)  # إضافة مسافتين بادئتين
                f.write('\n]')  # إغلاق المصفوفة
            return True
        except Exception as e:
            print(f"خطأ في حفظ المستخدمين: {e}")
            return False

    @staticmethod
    def create_new_user(user_id: int, username: str):
        """إنشاء مستخدم جديد"""
        return {
            "user_id": user_id,
            "username": username,
            "year": None,
            "semester": None,
            "major": None,
            "current_module": 1,
            "completed_modules": [],
            "completed_courses": [],
            "certificates": [],
            "sent_reminders": {},  # 🔥 الجديد: تخزين التذكيرات المرسلة
            "last_active": datetime.now(timezone.utc).isoformat(),  # 🔥 الجديد: آخر نشاط
            "message_count": 0  # 🔥 الجديد: عدد الرسائل
        }

    @staticmethod
    def clean_user_data(user):
        """تنظيف بيانات المستخدم من الحقول المؤقتة - الإصلاح: عدم حذف awaiting_new_question"""
        fields_to_remove = [
            "messages_count", "quiz_session", "final_exam_session"
            # 🔥 تم إزالة "awaiting_new_question" من هنا
        ]

        for field in fields_to_remove:
            if field in user:
                del user[field]

        return user

    @staticmethod
    def update_user_activity(user_id: int):
        """تحديث نشاط المستخدم"""
        user = UserManager.find_user(user_id)
        if user:
            user["last_active"] = datetime.now(timezone.utc).isoformat()
            user["message_count"] = user.get("message_count", 0) + 1
            UserManager.save_users()


# ============ أدوات الكورسات ============
class CourseManager:
    @staticmethod
    def get_course_for_user(user):
        """الحصول على الدورة الخاصة بالمستخدم"""
        if not user:
            return None
        sem = str(user.get("semester"))
        return courses.get(sem) if sem else None

    @staticmethod
    def ensure_user_course_progress(user):
        """التأكد من وجود بيانات التقدم"""
        changed = False
        if "current_module" not in user:
            user["current_module"] = 1
            changed = True
        if "completed_modules" not in user:
            user["completed_modules"] = []
            changed = True
        return changed

    @staticmethod
    def get_schedule_for_user(user):
        """
        ترجع جدول كامل للفصل اللي اختارو المستخدم (سنة + فصل + مسلك)
        """
        if not all(k in user for k in ("year", "semester", "major")):
            return None

        year_key = f"S{user['year']}"
        semester_key = str(user["semester"])
        major_key = user["major"]

        try:
            return schedule.get(year_key, {}).get(major_key, {}).get(semester_key, {})
        except KeyError:
            return None


# ============ الذكاء الاصطناعي ============
class AIService:
    def __init__(self):
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def ask_gemini(self, question: str) -> str:
        """سؤال Gemini AI"""
        prompt = (
            "أنت مساعد ديني مخصص لطلاب كلية أصول الدين، مهمتك إعطاء معلومات دينية عامة صحيحة وموثوقة فقط.\n"
            "⚠️ ممنوع إصدار أي فتوى أو تقديم نصائح شخصية.\n"
            "🎓 **الهوية والاختصاص:**\n"
            "- أنت مساعد افتراضي تابع لكلية أصول الدين\n"
            "- تقدم المعلومات للطلاب فقط لأغراض تعليمية\n"
            "إذا كان السؤال يطلب فتوى، أو يتعلق بمسائل شخصية، أو سؤال غير ديني، أجب:\n"
            "'⚠️ عذراً، لا يمكنني إصدار فتاوى أو إعطاء نصائح شخصية. يمكنك مراجعة عالم موثوق.'\n\n"
            f"جاوبني باختصار، 50 كلمة أو أقل، بلغة عربية رسمية، ومناسبة لطلاب كلية أصول الدين:\nالسؤال: {question}"
        )

        try:
            response = self.model.generate_content(prompt)
            text = " ".join([line.strip() for line in response.text.splitlines() if line.strip()])

            # حصر الجواب في 50 كلمة
            words = text.split()
            if len(words) > 50:
                text = " ".join(words[:50]) + "..."

            # فلترة أي محتوى حساس
            forbidden_phrases = ["فتوى", "حلال", "حرام", "يجوز", "نصيحة شخصية"]
            if any(phrase in text for phrase in forbidden_phrases):
                return "⚠️ عذراً، لا يمكنني إصدار فتاوى أو إعطاء نصائح شخصية. يمكنك مراجعة عالم موثوق."

            return text
        except (genai.types.GenerateContentError, ConnectionError, TimeoutError) as e:
            print(f"Gemini AI error: {e}")
            return "⚠️ وقع خطأ أثناء محاولة جلب الإجابة."


# ============ نظام التذكيرات التلقائية ============
class ReminderSystem:
    @staticmethod
    def parse_time(time_str):
        """تحويل وقت النص إلى كائن datetime"""
        try:
            start_time_str = time_str.split('-')[0].strip()
            return datetime.strptime(start_time_str, "%H:%M")
        except:
            return None

    @staticmethod
    def get_today_schedule_for_user(user):
        """الحصول على جدول اليوم للمستخدم"""
        schedule_data = CourseManager.get_schedule_for_user(user)
        if not schedule_data:
            return None

        today_en = WEEK_KEYS[datetime.now().weekday()]
        return schedule_data.get(today_en, [])

    @staticmethod
    def should_send_reminder(lecture_time, user, lecture_key):
        """التحقق إذا كان يجب إرسال التذكير"""
        if not lecture_time:
            return False

        # 🔥 التحقق من عدم إرسال التذكير مسبقاً
        sent_reminders = user.get("sent_reminders", {})
        if sent_reminders.get(lecture_key):
            return False

        now = datetime.now()
        lecture_datetime = datetime.combine(now.date(), lecture_time.time())
        time_diff = lecture_datetime - now

        # 🔥 إرسال التذكير قبل 30 دقيقة بالضبط
        return timedelta(minutes=29) < time_diff <= timedelta(minutes=30)

    @staticmethod
    async def send_lecture_reminders(context: ContextTypes.DEFAULT_TYPE):
        """إرسال تذكيرات المحاضرات لجميع المستخدمين"""
        now = datetime.now()
        today_en = WEEK_KEYS[now.weekday()]

        for user in users:
            try:
                # 🔥 التحقق من اكتمال بيانات المستخدم
                if not all(k in user for k in ("year", "semester", "major")):
                    continue

                # 🔥 الحصول على جدول اليوم
                schedule_data = CourseManager.get_schedule_for_user(user)
                if not schedule_data or today_en not in schedule_data:
                    continue

                today_lectures = schedule_data[today_en]
                if not today_lectures:
                    continue

                for lecture in today_lectures:
                    if isinstance(lecture, str):
                        continue

                    time_range = lecture.get("time", "")
                    if not time_range or '-' not in time_range:
                        continue

                    # 🔥 تحليل وقت المحاضرة
                    start_time = ReminderSystem.parse_time(time_range.split('-')[0])
                    if not start_time:
                        continue

                    # 🔥 إنشاء مفتاح فريد للمحاضرة
                    lecture_key = f"{today_en}_{time_range}_{lecture.get('subject', '')}"

                    # 🔥 التحقق من إرسال التذكير
                    if ReminderSystem.should_send_reminder(start_time, user, lecture_key):
                        # 🔥 إنشاء رسالة التذكير
                        reminder_msg = ReminderSystem.create_reminder_message(lecture, time_range)

                        # 🔥 إرسال التذكير
                        await context.bot.send_message(
                            chat_id=user["user_id"],
                            text=reminder_msg
                        )

                        # 🔥 تحديث حالة التذكير المرسل
                        if "sent_reminders" not in user:
                            user["sent_reminders"] = {}
                        user["sent_reminders"][lecture_key] = now.isoformat()

            except Exception as e:
                print(f"خطأ في إرسال تذكير للمستخدم {user.get('user_id')}: {e}")

        # 🔥 حفظ بيانات المستخدمين بعد التحديث
        UserManager.save_users()

    @staticmethod
    def create_reminder_message(lecture, time_range):
        """إنشاء رسالة التذكير"""
        subject = lecture.get("subject", "غير محدد")
        teacher = lecture.get("teacher", "غير محدد")
        room = lecture.get("room", "غير محدد")

        message = "⏰ **تذكير محاضرة قريباً!**\n\n"
        message += f"📚 **المادة:** {subject}\n"
        message += f"👨‍🏫 **الأستاذ:** {teacher}\n"
        message += f"🏫 **القاعة:** {room}\n"
        message += f"⏰ **الوقت:** {time_range}\n"
        message += f"🕐 **تبدأ بعد:** 30 دقيقة\n\n"
        message += "🎯 استعد للمحاضرة وحضر أدواتك!"

        return message

    @staticmethod
    def cleanup_old_reminders():
        """تنظيف التذكيرات القديمة"""
        now = datetime.now()
        today = now.date()

        for user in users:
            if "sent_reminders" not in user:
                continue

            # 🔥 إزالة التذكيرات الأقدم من اليوم
            user["sent_reminders"] = {
                key: timestamp for key, timestamp in user["sent_reminders"].items()
                if datetime.fromisoformat(timestamp).date() == today
            }


# ============ الأدوات التقنية ============
class TechUtils:
    @staticmethod
    def generate_qr_code(text: str):
        """إنشاء QR Code"""
        try:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(text)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")
            bio = BytesIO()
            img.save(bio, 'PNG')
            bio.seek(0)
            return bio
        except Exception:
            return None

    @staticmethod
    def advanced_url_check(url: str) -> str:
        """فحص الروابط المتقدم"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()

            checks = {
                'security_score': 100,
                'warnings': [],
                'details': {}
            }

            # فحص البروتوكول
            if url.startswith('https://'):
                checks['details']['protocol'] = '🟢 HTTPS - مشفر'
            else:
                checks['details']['protocol'] = '🔴 HTTP - غير مشفر'
                checks['security_score'] -= 30
                checks['warnings'].append('الرابط غير مشفر')

            # فحص النطاقات الموثوقة
            trusted_domains = {
                'wikipedia.org': 'موسوعة موثوقة',
                'islamweb.net': 'موقع إسلامي موثوق',
                'alukah.net': 'شبكة الألوكة الثقافية',
                '.edu.sa': 'مواقع التعليم السعودية'
            }

            domain_trusted = False
            for trusted_domain, description in trusted_domains.items():
                if trusted_domain in domain:
                    checks['details']['domain_trust'] = f'🟢 {description}'
                    checks['security_score'] += 20
                    domain_trusted = True
                    break

            if not domain_trusted:
                checks['details']['domain_trust'] = '🟡 نطاق عادي'

            # بناء التقرير
            security_score = max(0, min(100, checks['security_score']))

            if security_score >= 80:
                status = "🛡️ آمن جداً"
                color = "🟢"
            elif security_score >= 60:
                status = "👍 آمن"
                color = "🟡"
            elif security_score >= 40:
                status = "⚠️ حذر"
                color = "🟠"
            else:
                status = "🚫 خطر"
                color = "🔴"

            report = f"{color} **تقرير فحص الرابط**\n\n"
            report += f"🔗 **الرابط:** `{url}`\n"
            report += f"📊 **مستوى الأمان:** {security_score}% - {status}\n\n"

            report += "**تفاصيل الفحص:**\n"
            for key, value in checks['details'].items():
                report += f"• {value}\n"

            if checks['warnings']:
                report += f"\n**⚠️ تحذيرات:**\n"
                for warning in checks['warnings']:
                    report += f"• {warning}\n"

            report += f"\n{'✅' if security_score >= 60 else '❌'} **التوصية:** {'الرابط آمن للاستخدام' if security_score >= 60 else 'تجنب استخدام هذا الرابط'}"

            return report

        except Exception as e:
            return f"❌ **خطأ في الفحص:** {str(e)}"


# ============ الكيبوردات ============
class Keyboards:
    @staticmethod
    def main_menu():
        keyboard = [
            [KeyboardButton("📅 جدول المحاضرات"), KeyboardButton("📑 الامتحانات")],
            [KeyboardButton("📚 دورتي"), KeyboardButton("📚 المتون")],
            [KeyboardButton("🏢 مرافق الكلية"), KeyboardButton("❓ الأسئلة الشائعة")],
            [KeyboardButton("🛠️ تقنيتي")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    @staticmethod
    def technology_menu():
        keyboard = [
            [KeyboardButton("🔳 مولّد QR Code")],
            [KeyboardButton("🔗 فحص الروابط")],
            [KeyboardButton("🏠 الرجوع للرئيسية")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

    @staticmethod
    def back_only():
        return ReplyKeyboardMarkup([[KeyboardButton("🏠 الرجوع للرئيسية")]], resize_keyboard=True,
                                   one_time_keyboard=True)

    @staticmethod
    def year_selection():
        return ReplyKeyboardMarkup([[KeyboardButton("1"), KeyboardButton("2"), KeyboardButton("3")]],
                                   one_time_keyboard=True, resize_keyboard=True)

    @staticmethod
    def semester_selection(year: int):
        if year == 1:
            buttons = [[KeyboardButton("1"), KeyboardButton("2")]]
        elif year == 2:
            buttons = [[KeyboardButton("3"), KeyboardButton("4")]]
        else:
            buttons = [[KeyboardButton("5"), KeyboardButton("6")]]
        return ReplyKeyboardMarkup(buttons, one_time_keyboard=True, resize_keyboard=True)

    @staticmethod
    def major_selection():
        return ReplyKeyboardMarkup([
            [KeyboardButton("أصول الدين")],
            [KeyboardButton("الفكر الإسلامي والحوار الحضاري")]
        ], one_time_keyboard=True, resize_keyboard=True)

    @staticmethod
    def exam_menu():
        keyboard = [
            [KeyboardButton("📄 امتحانات سابقة")],
            [KeyboardButton("📚 كيفية المراجعة"), KeyboardButton("⏰ أوقات الامتحانات")],
            [KeyboardButton("🏠 الرجوع للرئيسية")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)


# ============ الدوال الأساسية ============
async def show_progress(update):
    """عرض شريط التقدم"""
    msg = await update.message.reply_text("⏳ جاري البحث: [□□□□□□□□] 0%")
    for i in range(1, 9):
        await asyncio.sleep(0.5)
        progress = "■" * i + "□" * (8 - i)
        percent = i * 12
        await msg.edit_text(f"⏳ جاري البحث: [{progress}] {percent}%")
    return msg


async def send_college_map(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إرسال خريطة الكلية"""
    text = "🏢 مرافق الكلية:\n<a href='https://www.google.com/maps/d/edit?mid=12HmTA4DmxkimkVSqNKroJCA5cfVAdBA&usp=sharing'>اضغط هنا لفتح مرافق الكلية</a>"
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=Keyboards.back_only())


async def send_faq_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض الأسئلة الشائعة"""
    faqs = texts.get("faqs", [])
    keyboard = []

    for i, q in enumerate(faqs):
        keyboard.append([InlineKeyboardButton(q['question'], callback_data=f"faq_{i}")])

    keyboard.append([InlineKeyboardButton("➕ إضافة سؤال جديد", callback_data="add_new_question")])
    keyboard.append([InlineKeyboardButton("👥 مجموعتي", callback_data="show_my_group")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("❓ اختر السؤال لمعرفة الجواب:", reply_markup=reply_markup)


async def exam_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """أمر الامتحانات"""
    await update.message.reply_text("اختر:", reply_markup=Keyboards.exam_menu())


# ============ دوال الجدول ============
WEEK_KEYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEK_AR = {
    "Monday": "الإثنين",
    "Tuesday": "الثلاثاء",
    "Wednesday": "الأربعاء",
    "Thursday": "الخميس",
    "Friday": "الجمعة",
    "Saturday": "السبت",
    "Sunday": "الأحد"
}


async def send_schedule_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, user):
    """قائمة الجدول الدراسي"""
    kb = [
        [KeyboardButton("📅 جدول اليوم"), KeyboardButton("📆 جدول الأسبوع")],
        [KeyboardButton("🔁 تغيير الشعبة/الفصل"), KeyboardButton("🏠 الرجوع للرئيسية")]
    ]
    await update.message.reply_text(
        "اختر الطريقة لعرض الجدول:",
        reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=True)
    )


# ============ دوال الجدول الدراسي ============
class ScheduleManager:
    @staticmethod
    def parse_time(time_str):
        """تحويل وقت النص إلى كائن datetime"""
        try:
            start_time_str = time_str.split('-')[0].strip()
            return datetime.strptime(start_time_str, "%H:%M")
        except:
            return None

    @staticmethod
    def debug_schedule_structure(schedule_data, user_info, verbose=False):
        """دالة تشخيصية لفحص هيكل الجدول"""
        debug_info = f"🔍 تشخيص الجدول:\n"
        debug_info += f"المستخدم: سنة {user_info.get('year')}, فصل {user_info.get('semester')}, مسلك {user_info.get('major')}\n"

        if not schedule_data:
            debug_info += "❌ الجدول فارغ أو غير موجود\n"
            return debug_info

        debug_info += f"مفاتيح الجدول الرئيسية: {list(schedule_data.keys())}\n"
        return debug_info

    @staticmethod
    def get_current_day_schedule(schedule_for_major, day_key):
        """الحصول على جدول اليوم مع معلومات الوقت الحي"""
        if not schedule_for_major:
            return "❌ الجدول غير متوفر"

        if day_key not in schedule_for_major:
            return f"⚠️ لا يوجد جدول ليوم {WEEK_AR.get(day_key, day_key)}"

        lst = schedule_for_major.get(day_key, [])

        if not isinstance(lst, list):
            if isinstance(lst, dict):
                lst = [lst]
            else:
                return "⚠️ تنسيق الجدول غير صحيح"

        if not lst:
            return f"📅 جدول {WEEK_AR.get(day_key, day_key)}:\nلا توجد محاضرات لهذا اليوم."

        now = datetime.now()
        current_time = now.time()
        result_lines = []

        for slot in lst:
            if isinstance(slot, str):
                try:
                    slot = {"time": slot, "subject": "مادة", "teacher": "أستاذ", "room": "قاعة"}
                except:
                    continue

            time_range = slot.get("time", "غير محدد")
            subject = slot.get("subject", "غير محدد")
            teacher = slot.get("teacher", "غير محدد")
            room = slot.get("room", "غير محدد")

            time_info = ""
            if time_range and isinstance(time_range, str) and '-' in time_range:
                times = time_range.split('-')
                if len(times) == 2:
                    start_time = ScheduleManager.parse_time(times[0])
                    end_time = ScheduleManager.parse_time(times[1])

                    if start_time and end_time:
                        start_time_obj = start_time.time()
                        end_time_obj = end_time.time()

                        if start_time_obj <= current_time <= end_time_obj:
                            time_info = "🟢 جاري الان"
                        elif current_time < start_time_obj:
                            time_diff = datetime.combine(now.date(), start_time_obj) - now
                            hours, remainder = divmod(time_diff.seconds, 3600)
                            minutes = remainder // 60

                            if hours > 0:
                                time_info = f"⏳ يبدأ بعد: {hours} ساعة {minutes} دقيقة"
                            else:
                                time_info = f"⏳ يبدأ بعد: {minutes} دقيقة"
                        else:
                            time_info = "✅ انتهت"

            line = f"⏰ {time_range} - {subject}\n👨‍🏫 {teacher} | 🏫 {room}"
            if time_info:
                line += f"\n{time_info}"
            result_lines.append(line)

        header = f"📅 جدول {WEEK_AR.get(day_key, day_key)}:\n\n"
        return header + "\n\n".join(result_lines)

    @staticmethod
    def get_next_class(schedule_for_major, day_key):
        """الحصول على المحاضرة التالية"""
        if not schedule_for_major or day_key not in schedule_for_major:
            return None

        lst = schedule_for_major.get(day_key, [])
        if not isinstance(lst, list) or not lst:
            return None

        now = datetime.now()
        current_time = now.time()

        for slot in lst:
            if isinstance(slot, str):
                continue

            time_range = slot.get("time", "")
            if time_range and '-' in time_range:
                start_time = ScheduleManager.parse_time(time_range.split('-')[0])
                if start_time and start_time.time() > current_time:
                    return slot
        return None

    @staticmethod
    def get_schedule_for_user_with_debug(user):
        """ترجع جدول كامل للفصل مع معلومات تشخيصية"""
        if not all(k in user for k in ("year", "semester", "major")):
            return None, "❌ معلومات المستخدم غير مكتملة"

        year_key = f"S{user['year']}"
        semester_key = str(user["semester"])
        major_key = user["major"]

        try:
            schedule_data = schedule.get(year_key, {}).get(major_key, {}).get(semester_key, {})
            if not schedule_data:
                return None, f"❌ لا يوجد جدول لـ {year_key}/{major_key}/{semester_key}"
            return schedule_data, "✅ الجدول موجود"
        except KeyError as e:
            return None, f"❌ خطأ في المفتاح: {e}"

    @staticmethod
    def create_weekly_table(schedule_for_major):
        """إنشاء جدول أسبوعي بتنسيق جدول منظم ومبسط"""
        if not schedule_for_major:
            return "❌ لا يوجد جدول"

        result = "📅 **جدول الأسبوع الدراسي**\n\n"
        days_english = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        for day_key in days_english:
            arabic_day = WEEK_AR.get(day_key, day_key)
            day_schedule = schedule_for_major.get(day_key, [])

            result += f"🔹 **{arabic_day}:**\n"

            if not day_schedule:
                result += "   • لا توجد محاضرات\n\n"
                continue

            # عرض المحاضرات في شكل قائمة منظمة
            for i, slot in enumerate(day_schedule, 1):
                if isinstance(slot, str):
                    continue

                time_range = slot.get("time", "غير محدد")
                subject = slot.get("subject", "غير محدد")
                teacher = slot.get("teacher", "غير محدد")
                room = slot.get("room", "غير محدد")

                result += f"   {i}️⃣ **⏰ {time_range}**\n"
                result += f"      📚 {subject}\n"
                result += f"      👨‍🏫 {teacher}\n"
                result += f"      🏫 {room}\n\n"

        result += "💡 *لرؤية الجدول مع الأوقات الحية، استخدم 'جدول اليوم'*"

        return result


# ============ دوال الجدول الدراسي (الواجهة) ============
async def send_today_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE, user):
    """إرسال جدول اليوم"""
    sch, debug_msg = ScheduleManager.get_schedule_for_user_with_debug(user)

    if not sch:
        debug_info = ScheduleManager.debug_schedule_structure(None, user)
        await update.message.reply_text(
            f"{debug_msg}\n\n{debug_info}\n\n⚠️ الرجاء التأكد من تسجيل المعلومات الصحيحة.",
            reply_markup=Keyboards.back_only()
        )
        return

    today_en = WEEK_KEYS[datetime.now().weekday()]
    arabic_day = WEEK_AR.get(today_en, today_en)

    body = ScheduleManager.get_current_day_schedule(sch, today_en)

    next_class = ScheduleManager.get_next_class(sch, today_en)
    if next_class:
        next_class_time = ScheduleManager.parse_time(next_class.get("time", "").split('-')[0])
        if next_class_time:
            time_diff = datetime.combine(datetime.now().date(), next_class_time.time()) - datetime.now()
            hours, remainder = divmod(time_diff.seconds, 3600)
            minutes = remainder // 60

            next_class_info = f"\n\n📋 المحاضرة التالية:\n{next_class.get('subject', 'غير محدد')}\n⏰ تبدأ بعد: {hours} ساعة {minutes} دقيقة"
            body += next_class_info

    await update.message.reply_text(body, reply_markup=Keyboards.back_only())


async def send_week_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE, user):
    """إرسال جدول الأسبوع"""
    sch = CourseManager.get_schedule_for_user(user)

    if not sch:
        await update.message.reply_text("❌ لا يوجد جدول لمسلكك وفصلك")
        return

    table_text = ScheduleManager.create_weekly_table(sch)
    await update.message.reply_text(table_text, parse_mode="Markdown")


# ============ دوال المساعدة ============
class HelperUtils:
    @staticmethod
    def save_new_question_entry(user_id, username, question_text):
        """تخزين السؤال الجديد"""
        entry = {
            "user_id": user_id,
            "question": question_text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return FileManager.append_to_json(Config.QUESTIONS_FILE, entry)

    @staticmethod
    async def handle_admin_faq_reply(update, context):
        """معالجة رد المشرف على الأسئلة"""
        answer = update.message.text
        data = context.user_data['pending_faq']

        texts_data = FileManager.safe_load_json(Config.TEXTS_FILE, {})
        if "faqs" not in texts_data:
            texts_data["faqs"] = []

        texts_data["faqs"].append({
            "question": data['question'],
            "answer": answer
        })

        # 🔥 الإضافة الجديدة: حفظ البيانات وتحديثها في الذاكرة مباشرة
        if FileManager.safe_save_json(Config.TEXTS_FILE, texts_data):
            # 🔥 تحديث البيانات في الذاكرة مباشرة
            global texts
            texts = FileManager.safe_load_json(Config.TEXTS_FILE, {})

            questions = FileManager.safe_load_json(Config.QUESTIONS_FILE, [])
            if data['index'] < len(questions):
                questions.pop(data['index'])
                FileManager.safe_save_json(Config.QUESTIONS_FILE, questions)

            del context.user_data['pending_faq']
            await update.message.reply_text("✅ تم إضافة السؤال إلى الأسئلة الشائعة وتحديثها مباشرة!")
        else:
            await update.message.reply_text("⚠️ حدث خطأ في حفظ السؤال.")


# ============ معالجة التسجيل ============
class RegistrationHandler:
    @staticmethod
    async def handle_year_selection(update, user, msg):
        """معالجة اختيار السنة"""
        if msg in ["1", "2", "3"]:
            user["year"] = int(msg)
            user["awaiting_semester"] = True
            UserManager.save_users()
            await update.message.reply_text("📌 اختر فصلك الدراسي:",
                                            reply_markup=Keyboards.semester_selection(user["year"]))
            return True
        else:
            await update.message.reply_text("⚠️ المرجو الضغط على زر من الأزرار فقط (السنة).",
                                            reply_markup=Keyboards.year_selection())
            return True

    @staticmethod
    async def handle_semester_selection(update, user, msg):
        """معالجة اختيار الفصل"""
        valid_semesters = {1: ["1", "2"], 2: ["3", "4"], 3: ["5", "6"]}
        if msg in valid_semesters.get(user["year"], []):
            user["semester"] = int(msg)
            user["awaiting_semester"] = False
            UserManager.save_users()
            await update.message.reply_text("📌 اختر مسلكك الدراسي:",
                                            reply_markup=Keyboards.major_selection())
            return True
        else:
            await update.message.reply_text("⚠️ المرجو الضغط على زر من الأزرار فقط (الفصل).",
                                            reply_markup=Keyboards.semester_selection(user["year"]))
            return True

    @staticmethod
    async def handle_major_selection(update, user, msg):
        """معالجة اختيار المسلك"""
        if msg in ["أصول الدين", "الفكر الإسلامي والحوار الحضاري"]:
            user["major"] = msg
            UserManager.save_users()
            await update.message.reply_text(
                f"✅ تم التسجيل بنجاح!\n📚 السنة: {user['year']}\n📖 الفصل: {user['semester']}\n🎓 المسلك: {user['major']}\n\n"
                "الآن يمكنك استعمال لوحة الأزرار.",
                reply_markup=Keyboards.main_menu()
            )
            return True
        else:
            await update.message.reply_text("⚠️ المرجو الضغط على زر من الأزرار فقط (المسلك).",
                                            reply_markup=Keyboards.major_selection())
            return True


# ============ معالجة الأزرار الرئيسية ============
class MainMenuHandler:
    @staticmethod
    async def handle_main_menu_buttons(update, context, user, msg):
        """معالجة أزرار القائمة الرئيسية"""
        handlers = {
            "📑 الامتحانات": exam_command,
            "/exam": exam_command,
            "🏢 مرافق الكلية": send_college_map,
            "❓ الأسئلة الشائعة": send_faq_buttons,
            "📚 المتون": MainMenuHandler.handle_pdf_texts,
            "📚 دورتي": MainMenuHandler.handle_my_course,
            "📅 جدول المحاضرات": lambda u, c: send_schedule_menu(u, c, user),
            "/schedule": lambda u, c: send_schedule_menu(u, c, user),
            "🛠️ تقنيتي": MainMenuHandler.handle_technology
        }

        if msg in handlers:
            await handlers[msg](update, context)
            return True
        return False

    @staticmethod
    async def handle_pdf_texts(update, context):
        """معالجة عرض المتون"""
        keyboard = [
            [InlineKeyboardButton(name, callback_data=f"pdf_{name}")]
            for name in pdf_texts.keys()
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("اختر المتن لتحميله:", reply_markup=reply_markup)

    @staticmethod
    async def handle_my_course(update, context):
        """معالجة عرض الدورة"""
        user_id = update.effective_user.id
        user = UserManager.find_user(user_id)
        if not user:
            await update.message.reply_text("⚠️ خاصك دير /start باش تسجل أولاً.",
                                            reply_markup=Keyboards.back_only())
            return

        course = CourseManager.get_course_for_user(user)
        if not course:
            await update.message.reply_text("🚫 ما كايناش دورة مرتبطة بالفصل ديالك.",
                                            reply_markup=Keyboards.back_only())
            return

        CourseManager.ensure_user_course_progress(user)
        UserManager.save_users()

        course_name = course.get("course_name", "دورة غير معروفة")
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton(f"📚 {course_name}", callback_data=f"course_open|{user_id}|{str(user.get('semester'))}")
        ]])
        await update.message.reply_text(f"الدورة المتاحة لك: {course_name}", reply_markup=kb)

    @staticmethod
    async def handle_technology(update, context):
        """معالجة القائمة التقنية"""
        await update.message.reply_text(
            "🛠️ **أدواتي التقنية المساعدة**\n\nاختر الأداة التي تحتاجها:",
            reply_markup=Keyboards.technology_menu(),
            parse_mode="Markdown"
        )


# ============ معالجة الأدوات التقنية ============
class TechnologyHandler:
    @staticmethod
    async def handle_qr_generation(update, context, msg):
        """معالجة إنشاء QR Code"""
        try:
            url = f"https://quickchart.io/qr?text={msg}&size=200"
            response = requests.get(url)

            if response.status_code == 200:
                bio = BytesIO(response.content)
                bio.seek(0)
                await update.message.reply_photo(
                    photo=bio,
                    caption=f"✅ تم إنشاء QR Code!\nالنص: {msg}",
                    reply_markup=Keyboards.technology_menu()
                )
            else:
                await update.message.reply_text(
                    "❌ فشل في إنشاء QR Code. حاول لاحقاً.",
                    reply_markup=Keyboards.technology_menu()
                )
        except Exception as e:
            await update.message.reply_text(
                f"❌ خطأ: {str(e)}",
                reply_markup=Keyboards.technology_menu()
            )
        context.user_data['awaiting_qr'] = False

    @staticmethod
    async def handle_link_check(update, context, msg):
        """معالجة فحص الروابط"""
        analyzing_msg = await update.message.reply_text("🔍 **جاري فحص الرابط...**\n\n⏳ قد يستغرق بضع ثوانٍ")

        safety_report = TechUtils.advanced_url_check(msg)

        await analyzing_msg.delete()
        await update.message.reply_text(
            safety_report,
            parse_mode="Markdown",
            reply_markup=Keyboards.technology_menu(),
            disable_web_page_preview=True
        )
        context.user_data['awaiting_link_check'] = False


# ============ معالجة أزرار الجدول ============
async def handle_schedule_buttons(update, context, user, msg):
    """معالجة أزرار الجدول"""
    if msg == "📅 جدول اليوم":
        await send_today_schedule(update, context, user)
    elif msg == "📆 جدول الأسبوع":
        await send_week_schedule(update, context, user)
    elif msg == "🔁 تغيير الشعبة/الفصل":
        user["year"] = None
        user["semester"] = None
        user["major"] = None
        UserManager.save_users()
        await update.message.reply_text("حسناً، عاود اختَر سنتك الدراسية:",
                                        reply_markup=Keyboards.year_selection())


# ============ معالجة القوائم الفرعية ============
async def handle_submenu_buttons(update, context, user, msg):
    """معالجة أزرار القوائم الفرعية"""

    # أزرار التقنيات الفرعية
    if msg == "🔳 مولّد QR Code":
        await update.message.reply_text(
            "🔳 **مولّد QR Code**\n\nأرسل النص أو الرابط الذي تريد تحويله:",
            reply_markup=Keyboards.back_only(),
            parse_mode="Markdown"
        )
        context.user_data['awaiting_qr'] = True
        return True

    elif msg == "🔗 فحص الروابط":
        await update.message.reply_text(
            "🔗 **فحص سلامة الروابط**\n\nأرسل الرابط الذي تريد فحصه:",
            reply_markup=Keyboards.back_only(),
            parse_mode="Markdown"
        )
        context.user_data['awaiting_link_check'] = True
        return True

    # أزرار الامتحانات الفرعية
    elif msg == "📄 امتحانات سابقة":
        await update.message.reply_text(
            "📄 **الامتحانات السابقة**\n\nسيتم إضافة الامتحانات السابقة قريباً...",
            reply_markup=Keyboards.exam_menu()
        )
        return True

    elif msg == "📚 كيفية المراجعة":
        await update.message.reply_text(
            "📚 **نصائح للمراجعة:**\n\n"
            "1. نظم وقتك واجعل جدولاً للمراجعة\n"
            "2. ركز على النقاط الأساسية في كل مادة\n"
            "3. استخدم التلخيص والخرائط الذهنية\n"
            "4. حل تمارين وامتحانات سابقة\n"
            "5. خذ فترات راحة منتظمة",
            reply_markup=Keyboards.exam_menu()
        )
        return True

    elif msg == "⏰ أوقات الامتحانات":
        await update.message.reply_text(
            "⏰ **مواعيد الامتحانات**\n\n"
            "سيتم الإعلان عن مواعيد الامتحانات قريباً...\n"
            "تابع الإعلانات الرسمية للكلية.",
            reply_markup=Keyboards.exam_menu()
        )
        return True

    return False


# ============ معالجة قاعدة البيانات ============
async def handle_database_answer(update, answer):
    """معالجة الإجابة من قاعدة البيانات"""
    response_type = answer.get("type")
    if response_type == "text":
        await update.message.reply_text(f"🕌 كلية أصول الدين\n{answer['content']}",
                                        reply_markup=Keyboards.main_menu())
    elif response_type == "file":
        try:
            await update.message.reply_document(open(f"files/{answer['content']}", "rb"),
                                                caption="🕌 كلية أصول الدين\n📂 الملف جاهز للتحميل:")
        except Exception:
            await update.message.reply_text("⚠️ الملف غير متوفر حالياً.",
                                            reply_markup=Keyboards.main_menu())
    elif response_type == "image_text":
        try:
            await update.message.reply_photo(open(f"files/{answer['content']['image']}", "rb"),
                                             caption=f"🕌 كلية أصول الدين\n{answer['content']['text']}")
        except Exception:
            await update.message.reply_text(f"🕌 كلية أصول الدين\n{answer['content']['text']}",
                                            reply_markup=Keyboards.main_menu())


# ============ الدالة الرئيسية لمعالجة الرسائل ============
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message.text.strip() if update.message.text else ""
    user_id = update.effective_user.id
    username = update.effective_user.username or f"user{user_id}"

    user = UserManager.find_user(user_id)

    # 🔥 تحديث نشاط المستخدم
    UserManager.update_user_activity(user_id)

    # 🔥 الإصلاح: عدم تنظيف البيانات قبل التحقق من حالة السؤال الجديد
    # if user:
    #     user = UserManager.clean_user_data(user)

    if not user:
        user = UserManager.create_new_user(user_id, username)
        users.append(user)
        UserManager.save_users()
        await update.message.reply_text("👋 مرحباً! اختر سنتك الدراسية:",
                                        reply_markup=Keyboards.year_selection())
        return

    if user_id in Config.ADMINS and 'pending_faq' in context.user_data:
        await HelperUtils.handle_admin_faq_reply(update, context)
        return

    # 🔥 الإصلاح: معالجة السؤال الجديد في البداية قبل أي شيء آخر
    if user.get("awaiting_new_question"):
        question_text = msg.strip()

        # 🔥 منع الأسئلة الفارغة
        if not question_text:
            await update.message.reply_text("⚠️ الرجاء كتابة سؤال صحيح.", reply_markup=Keyboards.back_only())
            return

        # 🔥 حفظ السؤال في ملف الأسئلة الجديدة
        success = HelperUtils.save_new_question_entry(user_id, username, question_text)

        # تنظيف الحالة
        user["awaiting_new_question"] = False
        UserManager.save_users()

        if success:
            await update.message.reply_text(
                "✅ شكراً، تم تسجيل سؤالك بنجاح! سيظهر للمشرف للمراجعة.",
                reply_markup=Keyboards.main_menu()
            )
        else:
            await update.message.reply_text(
                "⚠️ حدث خطأ في حفظ السؤال. حاول مرة أخرى.",
                reply_markup=Keyboards.main_menu()
            )
        return

    # 🔥 تنظيف البيانات بعد معالجة السؤال الجديد
    if user:
        user = UserManager.clean_user_data(user)

    if not user.get("year"):
        if await RegistrationHandler.handle_year_selection(update, user, msg):
            return

    if user.get("awaiting_semester"):
        if await RegistrationHandler.handle_semester_selection(update, user, msg):
            return

    if not user.get("major"):
        if await RegistrationHandler.handle_major_selection(update, user, msg):
            return

    # 🔥 معالجة الأزرار الفرعية أولاً
    if await handle_submenu_buttons(update, context, user, msg):
        return

    # معالجة الأزرار الرئيسية
    if await MainMenuHandler.handle_main_menu_buttons(update, context, user, msg):
        return

    if msg in ["📅 جدول اليوم", "📆 جدول الأسبوع", "🔁 تغيير الشعبة/الفصل"]:
        await handle_schedule_buttons(update, context, user, msg)
        return

    if msg == "🏠 الرجوع للرئيسية":
        await update.message.reply_text("🏠 عدت إلى القائمة الرئيسية.",
                                        reply_markup=Keyboards.main_menu())
        return

    # معالجة المدخلات النصية للتقنيات
    if context.user_data.get('awaiting_qr'):
        await TechnologyHandler.handle_qr_generation(update, context, msg)
        return

    if context.user_data.get('awaiting_link_check'):
        await TechnologyHandler.handle_link_check(update, context, msg)
        return

    # البحث في الردود التفاعلية
    interactive = TextUtils.interactive_replies(msg)
    if interactive:
        await update.message.reply_text(interactive, reply_markup=Keyboards.main_menu())
        return

    # البحث في قاعدة البيانات
    answer = TextUtils.search_summary(msg, db)
    if answer:
        await handle_database_answer(update, answer)
        return

    # 🔥 فقط إذا لم يتم التعرف على الرسالة، نرسلها إلى Gemini AI
    loading_msg = await show_progress(update)
    ai_service = AIService()
    gemini_answer = ai_service.ask_gemini(update.message.text)
    await loading_msg.delete()
    await update.message.reply_text(f"🕌 جوابي حسب علمي:\n{gemini_answer}")


# ============ أوامر المشرفين المحسنة ============
class AdminManager:
    @staticmethod
    async def admin_poll(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إنشاء استطلاع من قبل المشرف"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرف فقط.")
            return

        raw = update.message.text or ""
        raw = re.sub(r'^/poll(@\w+)?\s*', '', raw, count=1).strip()

        parts = []
        try:
            if raw:
                parts = shlex.split(raw)
        except ValueError:
            parts = [p.strip() for p in raw.split('|') if p.strip()]

        if len(parts) < 3:
            await update.message.reply_text(
                "⚠️ الاستعمال الصحيح:\n"
                "/poll \"السؤال\" \"خيار1\" \"خيار2\" [\"خيار3\" ...]\nأو:\n"
                "/poll السؤال | خيار1 | خيار2 | خيار3\n(خاص على الأقل جوج خيارات)."
            )
            return

        question = parts[0]
        options = parts[1:]

        if len(options) > 10:
            options = options[:10]

        sent = 0
        failed = 0
        for u in users:
            try:
                await context.bot.send_poll(
                    chat_id=u["user_id"],
                    question=question,
                    options=options,
                    is_anonymous=True
                )
                sent += 1
            except Exception:
                failed += 1

        await update.message.reply_text(f"✅ تم إرسال الاستطلاع: نجح {sent}، فشل {failed}.")

    @staticmethod
    async def admin_questions(update, context):
        """عرض الأسئلة الجديدة للمشرف"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرف فقط.")
            return

        questions = FileManager.safe_load_json(Config.QUESTIONS_FILE, default=[])
        if not questions:
            await update.message.reply_text("⚠️ لا توجد أسئلة جديدة حالياً.")
            return

        for i, q in enumerate(questions):
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ اعتماد", callback_data=f"approve_{i}"),
                 InlineKeyboardButton("❌ رفض", callback_data=f"reject_{i}")]
            ])
            await update.message.reply_text(f"{i + 1}. {q['question']}\n⏰ {q['timestamp']}", reply_markup=keyboard)

    @staticmethod
    async def admin_announce(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إرسال إعلان للمستخدمين"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرف فقط.")
            return

        if not context.args and not update.message.reply_to_message:
            await update.message.reply_text("⚠️ استعمل: /announce <النص> أو رد على رسالة تحتوي صورة/ملف.")
            return

        if context.args:
            text = " ".join(context.args)
            formatted = f"📢 إعلان:\n\n{text}"
            await AdminManager.send_to_all_users(context, formatted, "text")
            await update.message.reply_text("✅ تم إرسال الإعلان.")
            return

        if update.message.reply_to_message:
            reply = update.message.reply_to_message
            if reply.photo:
                await AdminManager.send_to_all_users(context, reply, "photo")
                await update.message.reply_text("✅ تم إرسال إعلان الصورة.")
                return
            if reply.document:
                await AdminManager.send_to_all_users(context, reply, "document")
                await update.message.reply_text("✅ تم إرسال ملف للجميع.")
                return

        await update.message.reply_text("⚠️ نوع الرسالة غير مدعوم للإعلان.")

    @staticmethod
    async def send_to_all_users(context, content, content_type):
        """إرسال محتوى لجميع المستخدمين"""
        for u in users:
            try:
                if content_type == "text":
                    await context.bot.send_message(chat_id=u["user_id"], text=content)
                elif content_type == "photo":
                    await context.bot.send_photo(
                        chat_id=u["user_id"],
                        photo=content.photo[-1].file_id,
                        caption=content.caption or ""
                    )
                elif content_type == "document":
                    await context.bot.send_document(
                        chat_id=u["user_id"],
                        document=content.document.file_id,
                        caption=content.caption or ""
                    )
            except Exception:
                continue

    @staticmethod
    async def admin_stats(update, context):
        """إحصائيات المستخدمين"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        if not users:
            await update.message.reply_text("⚠️ لا يوجد أي طالب مسجّل.")
            return

        stats = {}
        for u in users:
            major = u.get("major", "غير محدد")
            sem = str(u.get("semester", "؟"))
            if major not in stats:
                stats[major] = {str(i): 0 for i in range(1, 7)}
            if sem in stats[major]:
                stats[major][sem] += 1

        table_data = []
        for major, sems in stats.items():
            for s in range(1, 7):
                cnt = sems.get(str(s), 0)
                table_data.append([major, s, cnt])

        table_text = tabulate(table_data, headers=["الشعبة", "الفصل", "عدد الطلبة"], tablefmt="pretty")
        final_text = "📊 إحصائيات الطلبة حسب الشعبة والفصول\n\n" + table_text

        await update.message.reply_text(f"<pre>{final_text}</pre>", parse_mode="HTML")

    @staticmethod
    async def add_faq(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إضافة سؤال شائع مباشرة"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                "⚠️ الاستعمال الصحيح:\n"
                "/add_faq \"السؤال\" \"الجواب\"\n\n"
                "مثال:\n"
                "/add_faq \"ما هي مواعيد الدوام؟\" \"الدوام من الأحد إلى الخميس من 8 صباحاً إلى 2 ظهراً\""
            )
            return

        try:
            # استخراج السؤال والجواب من الأقواس
            text = " ".join(context.args)
            if '"' in text:
                parts = shlex.split(text)
                if len(parts) >= 2:
                    question = parts[0]
                    answer = " ".join(parts[1:])
                else:
                    await update.message.reply_text("❌ يجب وضع السؤال والجواب بين أقواس")
                    return
            else:
                # إذا لم تكن هناك أقواس، نأخذ أول كلمتين كسؤال والباقي كجواب
                parts = text.split(" ", 1)
                if len(parts) >= 2:
                    question = parts[0]
                    answer = parts[1]
                else:
                    await update.message.reply_text("❌ يجب تقديم سؤال وجواب")
                    return

            # تحميل البيانات الحالية
            texts_data = FileManager.safe_load_json(Config.TEXTS_FILE, {})
            if "faqs" not in texts_data:
                texts_data["faqs"] = []

            # إضافة السؤال الجديد
            texts_data["faqs"].append({
                "question": question,
                "answer": answer
            })

            # حفظ البيانات
            if FileManager.safe_save_json(Config.TEXTS_FILE, texts_data):
                global texts
                texts = FileManager.safe_load_json(Config.TEXTS_FILE, {})
                await update.message.reply_text(
                    f"✅ تم إضافة السؤال الشائع بنجاح!\n\n"
                    f"❓ السؤال: {question}\n"
                    f"💡 الجواب: {answer}"
                )
            else:
                await update.message.reply_text("❌ حدث خطأ في حفظ السؤال")

        except Exception as e:
            await update.message.reply_text(f"❌ حدث خطأ: {str(e)}")

    @staticmethod
    async def edit_faq(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """تعديل سؤال شائع موجود"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        if not context.args or len(context.args) < 3:
            await update.message.reply_text(
                "⚠️ الاستعمال الصحيح:\n"
                "/edit_faq <رقم_السؤال> \"السؤال_الجديد\" \"الجواب_الجديد\"\n\n"
                "مثال:\n"
                "/edit_faq 1 \"ما هي المواعيد الجديدة؟\" \"المواعيد الجديدة من 8 إلى 3\"\n\n"
                "لرؤية قائمة الأسئلة استخدم: /list_faqs"
            )
            return

        try:
            faq_id = int(context.args[0]) - 1  # تحويل إلى index (يبدأ من 0)

            # استخراج السؤال والجواب الجديدين
            text = " ".join(context.args[1:])
            if '"' in text:
                parts = shlex.split(text)
                if len(parts) >= 2:
                    new_question = parts[0]
                    new_answer = " ".join(parts[1:])
                else:
                    await update.message.reply_text("❌ يجب وضع السؤال والجواب بين أقواس")
                    return
            else:
                parts = text.split(" ", 1)
                if len(parts) >= 2:
                    new_question = parts[0]
                    new_answer = parts[1]
                else:
                    await update.message.reply_text("❌ يجب تقديم سؤال وجواب جديدين")
                    return

            # تحميل البيانات الحالية
            texts_data = FileManager.safe_load_json(Config.TEXTS_FILE, {})
            if "faqs" not in texts_data or faq_id >= len(texts_data["faqs"]) or faq_id < 0:
                await update.message.reply_text("❌ رقم السؤال غير صحيح")
                return

            # الحصول على السؤال القديم
            old_question = texts_data["faqs"][faq_id]["question"]
            old_answer = texts_data["faqs"][faq_id]["answer"]

            # تحديث السؤال
            texts_data["faqs"][faq_id] = {
                "question": new_question,
                "answer": new_answer
            }

            # حفظ البيانات
            if FileManager.safe_save_json(Config.TEXTS_FILE, texts_data):
                global texts
                texts = FileManager.safe_load_json(Config.TEXTS_FILE, {})
                await update.message.reply_text(
                    f"✅ تم تعديل السؤال بنجاح!\n\n"
                    f"📝 **السؤال القديم:**\n{old_question}\n💡 {old_answer}\n\n"
                    f"📝 **السؤال الجديد:**\n{new_question}\n💡 {new_answer}"
                )
            else:
                await update.message.reply_text("❌ حدث خطأ في حفظ التعديلات")

        except ValueError:
            await update.message.reply_text("❌ رقم السؤال يجب أن يكون رقماً صحيحاً")
        except Exception as e:
            await update.message.reply_text(f"❌ حدث خطأ: {str(e)}")

    @staticmethod
    async def delete_faq(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """حذف سؤال شائع"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        if not context.args or len(context.args) != 1:
            await update.message.reply_text(
                "⚠️ الاستعمال الصحيح:\n"
                "/delete_faq <رقم_السؤال>\n\n"
                "مثال:\n"
                "/delete_faq 1\n\n"
                "لرؤية قائمة الأسئلة استخدم: /list_faqs"
            )
            return

        try:
            faq_id = int(context.args[0]) - 1  # تحويل إلى index (يبدأ من 0)

            # تحميل البيانات الحالية
            texts_data = FileManager.safe_load_json(Config.TEXTS_FILE, {})
            if "faqs" not in texts_data or faq_id >= len(texts_data["faqs"]) or faq_id < 0:
                await update.message.reply_text("❌ رقم السؤال غير صحيح")
                return

            # الحصول على السؤال المراد حذفه
            deleted_question = texts_data["faqs"][faq_id]["question"]
            deleted_answer = texts_data["faqs"][faq_id]["answer"]

            # حذف السؤال
            texts_data["faqs"].pop(faq_id)

            # حفظ البيانات
            if FileManager.safe_save_json(Config.TEXTS_FILE, texts_data):
                global texts
                texts = FileManager.safe_load_json(Config.TEXTS_FILE, {})
                await update.message.reply_text(
                    f"✅ تم حذف السؤال بنجاح!\n\n"
                    f"🗑️ **السؤال المحذوف:**\n{deleted_question}\n💡 {deleted_answer}"
                )
            else:
                await update.message.reply_text("❌ حدث خطأ في حذف السؤال")

        except ValueError:
            await update.message.reply_text("❌ رقم السؤال يجب أن يكون رقماً صحيحاً")
        except Exception as e:
            await update.message.reply_text(f"❌ حدث خطأ: {str(e)}")

    @staticmethod
    async def list_faqs(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض قائمة الأسئلة الشائعة"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        texts_data = FileManager.safe_load_json(Config.TEXTS_FILE, {})
        faqs = texts_data.get("faqs", [])

        if not faqs:
            await update.message.reply_text("⚠️ لا توجد أسئلة شائعة حالياً.")
            return

        message = "📋 قائمة الأسئلة الشائعة:\n\n"
        for i, faq in enumerate(faqs, 1):
            question = faq['question']
            answer_preview = faq['answer'][:50] + "..." if len(faq['answer']) > 50 else faq['answer']

            message += f"{i}. {question}\n"
            message += f"   💡 {answer_preview}\n\n"

        message += "\n💡 الأوامر المتاحة:\n"
        message += "/edit_faq <رقم> \"سؤال\" \"جواب\" - تعديل سؤال\n"
        message += "/delete_faq <رقم> - حذف سؤال\n"
        message += "/add_faq \"سؤال\" \"جواب\" - إضافة سؤال جديد"

        await update.message.reply_text(message)

    @staticmethod
    async def admin_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض جميع الأوامر الإدارية المتاحة في جدول احترافي"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        # إنشاء جدول الأوامر بشكل احترافي
        help_sections = [
            {
                "emoji": "📢",
                "category": "الإعلانات والاتصال",
                "commands": [
                    ("/announce", "إرسال إعلان نصي أو بملف/صورة"),
                    ("/poll", "إنشاء استطلاع للرأي"),
                    ("/questions", "عرض الأسئلة الجديدة")
                ]
            },
            {
                "emoji": "📊",
                "category": "الإحصائيات والتقارير",
                "commands": [
                    ("/stats", "إحصائيات المستخدمين"),
                    ("/weekly_report", "تقرير أسبوعي مفصل"),
                    ("/activity", "نشاط المستخدمين"),
                    ("/growth", "معدل نمو المستخدمين")
                ]
            },
            {
                "emoji": "❓",
                "category": "إدارة الأسئلة الشائعة",
                "commands": [
                    ("/add_faq", "إضافة سؤال جديد"),
                    ("/edit_faq", "تعديل سؤال موجود"),
                    ("/delete_faq", "حذف سؤال"),
                    ("/list_faqs", "عرض قائمة الأسئلة")
                ]
            },
            {
                "emoji": "🛡️",
                "category": "إدارة المشرفين",
                "commands": [
                    ("/admins_list", "عرض قائمة المشرفين"),
                    ("/add_admin", "إضافة مشرف جديد"),
                    ("/remove_admin", "إزالة مشرف")
                ]
            },
            {
                "emoji": "ℹ️",
                "category": "المساعدة",
                "commands": [
                    ("/admin_help", "عرض هذه القائمة")
                ]
            }
        ]

        # بناء الرسالة بشكل جدول احترافي
        help_text = "👑 **لوحة تحكم المشرفين - جميع الأوامر المتاحة**\n\n"
        help_text += "═" * 50 + "\n\n"

        for section in help_sections:
            help_text += f"{section['emoji']} **{section['category']}**\n"
            help_text += "─" * 30 + "\n"

            for command, description in section['commands']:
                help_text += f"• `{command}` - {description}\n"

            help_text += "═" * 50 + "\n\n"

        help_text += "💡 **لنسخ أي أمر:** اضغط على الأمر ليتم نسخه تلقائياً"

        await update.message.reply_text(
            help_text,
            parse_mode="Markdown"
        )


# ============ معالجة نسخ الأوامر ============
async def handle_copy_command_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة نسخ الأوامر"""
    query = update.callback_query
    await query.answer()

    data = query.data
    if data.startswith("copy_command_"):
        command = data.replace("copy_command_", "")
        full_command = f"/{command}"

        # نسخ الأمر إلى الحافظة
        await query.message.reply_text(
            f"✅ تم نسخ الأمر: `{full_command}`\n\n"
            f"📋 **يمكنك الآن لصقه واستخدامه مباشرة**\n\n"
            f"💡 **نصيحة:** استخدم الأمر في الدردشة الرئيسية للبوت",
            parse_mode="Markdown"
        )


# ============ أوامر الأمان الجديدة ============
class SecurityManager:
    @staticmethod
    async def admins_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض قائمة المشرفين"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        if not Config.ADMINS:
            await update.message.reply_text("⚠️ لا يوجد مشرفين حالياً.")
            return

        message = "🛡️ **قائمة المشرفين:**\n\n"
        for i, admin_id in enumerate(Config.ADMINS, 1):
            try:
                user = await context.bot.get_chat(admin_id)
                username = f"@{user.username}" if user.username else "بدون معرف"
                message += f"{i}. {user.first_name} {username} (`{admin_id}`)\n"
            except Exception:
                message += f"{i}. مستخدم غير متاح (`{admin_id}`)\n"

        message += f"\n👑 **إجمالي المشرفين:** {len(Config.ADMINS)}"

        await update.message.reply_text(message, parse_mode="Markdown")

    @staticmethod
    async def add_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إضافة مشرف جديد"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        if not context.args:
            await update.message.reply_text(
                "⚠️ الاستعمال الصحيح:\n"
                "/add_admin <معرف_المستخدم>\n\n"
                "مثال:\n"
                "/add_admin 123456789"
            )
            return

        try:
            new_admin_id = int(context.args[0])

            # التحقق من عدم إضافة المستخدم نفسه
            if new_admin_id == user_id:
                await update.message.reply_text("❌ لا يمكنك إضافة نفسك كمشرف.")
                return

            # التحقق من عدم وجود المستخدم مسبقاً في القائمة
            if new_admin_id in Config.ADMINS:
                await update.message.reply_text("❌ هذا المستخدم مشرف بالفعل.")
                return

            # إضافة المشرف الجديد
            Config.ADMINS.append(new_admin_id)

            # تحديث ملف البيئة
            env_file = ".env"
            if os.path.exists(env_file):
                with open(env_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                with open(env_file, "w", encoding="utf-8") as f:
                    for line in lines:
                        if line.startswith("ADMINS="):
                            admins_str = ",".join(map(str, Config.ADMINS))
                            f.write(f"ADMINS={admins_str}\n")
                        else:
                            f.write(line)

            # الحصول على معلومات المستخدم الجديد
            try:
                new_admin_user = await context.bot.get_chat(new_admin_id)
                admin_name = f"{new_admin_user.first_name} (@{new_admin_user.username})" if new_admin_user.username else new_admin_user.first_name
            except:
                admin_name = f"المستخدم ({new_admin_id})"

            await update.message.reply_text(
                f"✅ تم إضافة المشرف بنجاح!\n\n"
                f"👤 **المشرف الجديد:** {admin_name}\n"
                f"🆔 **المعرف:** `{new_admin_id}`\n\n"
                f"👑 **إجمالي المشرفين الآن:** {len(Config.ADMINS)}"
            )

        except ValueError:
            await update.message.reply_text("❌ المعرف يجب أن يكون رقماً صحيحاً.")
        except Exception as e:
            await update.message.reply_text(f"❌ حدث خطأ: {str(e)}")

    @staticmethod
    async def remove_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إزالة مشرف"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        if not context.args:
            await update.message.reply_text(
                "⚠️ الاستعمال الصحيح:\n"
                "/remove_admin <معرف_المستخدم>\n\n"
                "مثال:\n"
                "/remove_admin 123456789\n\n"
                "لرؤية قائمة المشرفين استخدم: /admins_list"
            )
            return

        try:
            admin_to_remove = int(context.args[0])

            # التحقق من عدم إزالة النفس
            if admin_to_remove == user_id:
                await update.message.reply_text("❌ لا يمكنك إزالة نفسك من قائمة المشرفين.")
                return

            # التحقق من وجود المشرف في القائمة
            if admin_to_remove not in Config.ADMINS:
                await update.message.reply_text("❌ هذا المستخدم ليس مشرفاً.")
                return

            # إزالة المشرف
            Config.ADMINS.remove(admin_to_remove)

            # تحديث ملف البيئة
            env_file = ".env"
            if os.path.exists(env_file):
                with open(env_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                with open(env_file, "w", encoding="utf-8") as f:
                    for line in lines:
                        if line.startswith("ADMINS="):
                            admins_str = ",".join(map(str, Config.ADMINS))
                            f.write(f"ADMINS={admins_str}\n")
                        else:
                            f.write(line)

            await update.message.reply_text(
                f"✅ تم إزالة المشرف بنجاح!\n\n"
                f"🆔 **المعرف:** `{admin_to_remove}`\n\n"
                f"👑 **إجمالي المشرفين الآن:** {len(Config.ADMINS)}"
            )

        except ValueError:
            await update.message.reply_text("❌ المعرف يجب أن يكون رقماً صحيحاً.")
        except Exception as e:
            await update.message.reply_text(f"❌ حدث خطأ: {str(e)}")


# ============ الأوامر الإدارية الجديدة ============
class AdminReports:
    @staticmethod
    async def weekly_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """تقرير أسبوعي مفصل"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        # حساب التاريخ قبل أسبوع
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)

        # تحليل البيانات
        total_users = len(users)
        new_users_week = 0
        active_users_week = 0
        total_messages = 0

        # إحصائيات حسب التخصص
        major_stats = {}
        year_stats = {}

        for user in users:
            # المستخدمين الجدد
            if "last_active" in user:
                last_active = datetime.fromisoformat(user["last_active"])
                if last_active >= week_ago:
                    active_users_week += 1

                    # عدد الرسائل
                    total_messages += user.get("message_count", 0)

            # إحصائيات التخصص
            major = user.get("major", "غير محدد")
            if major not in major_stats:
                major_stats[major] = 0
            major_stats[major] += 1

            # إحصائيات السنة
            year = user.get("year", "غير محدد")
            if year not in year_stats:
                year_stats[year] = 0
            year_stats[year] += 1

        # حساب النشاط
        activity_percentage = (active_users_week / total_users * 100) if total_users > 0 else 0

        # بناء التقرير
        report = "📊 **التقرير الأسبوعي**\n\n"
        report += f"👥 **إجمالي المستخدمين:** {total_users}\n"
        report += f"🆕 **المستخدمين النشطين (أسبوع):** {active_users_week}\n"
        report += f"📈 **نسبة النشاط:** {activity_percentage:.1f}%\n"
        report += f"💬 **إجمالي الرسائل:** {total_messages}\n\n"

        report += "🎓 **التوزيع حسب التخصص:**\n"
        for major, count in major_stats.items():
            percentage = (count / total_users * 100) if total_users > 0 else 0
            report += f"  • {major}: {count} ({percentage:.1f}%)\n"

        report += "\n📚 **التوزيع حسب السنة:**\n"
        for year, count in year_stats.items():
            percentage = (count / total_users * 100) if total_users > 0 else 0
            report += f"  • السنة {year}: {count} ({percentage:.1f}%)\n"

        report += f"\n⏰ **آخر تحديث:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        await update.message.reply_text(report, parse_mode="Markdown")

    @staticmethod
    async def user_activity(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نشاط المستخدمين"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        # فرز المستخدمين حسب آخر نشاط
        active_users = []
        for user in users:
            if "last_active" in user:
                last_active = datetime.fromisoformat(user["last_active"])
                message_count = user.get("message_count", 0)
                active_users.append({
                    "user": user,
                    "last_active": last_active,
                    "message_count": message_count
                })

        # ترتيب حسب آخر نشاط (الأحدث أولاً)
        active_users.sort(key=lambda x: x["last_active"], reverse=True)

        # عرض أفضل 10 مستخدمين نشاطاً
        report = "🏆 **أكثر المستخدمين نشاطاً**\n\n"

        for i, data in enumerate(active_users[:10], 1):
            user = data["user"]
            last_active = data["last_active"]
            message_count = data["message_count"]

            # حساب الوقت منذ آخر نشاط
            time_diff = datetime.now(timezone.utc) - last_active
            hours = int(time_diff.total_seconds() // 3600)

            if hours < 1:
                last_seen = "الآن"
            elif hours < 24:
                last_seen = f"قبل {hours} ساعة"
            else:
                days = hours // 24
                last_seen = f"قبل {days} يوم"

            username = user.get("username", "بدون معرف")
            report += f"{i}. @{username}\n"
            report += f"   💬 {message_count} رسالة | 🕐 {last_seen}\n\n"

        report += f"📊 **إجمالي المستخدمين النشطين:** {len(active_users)}"

        await update.message.reply_text(report, parse_mode="Markdown")

    @staticmethod
    async def growth_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معدل نمو المستخدمين"""
        user_id = update.effective_user.id
        if user_id not in Config.ADMINS:
            await update.message.reply_text("🚫 هذا الأمر مخصص للمشرفين فقط.")
            return

        # تحليل النمو (نفترض أن تاريخ الإنشاء هو أول ظهور في النظام)
        today = datetime.now(timezone.utc).date()
        growth_data = {}

        for user in users:
            if "last_active" in user:
                join_date = datetime.fromisoformat(user["last_active"]).date()
                days_since_join = (today - join_date).days

                # تجميع حسب الأسبوع
                week_key = join_date.isocalendar()[1]  # رقم الأسبوع
                year_key = join_date.year
                key = f"{year_key}-W{week_key}"

                if key not in growth_data:
                    growth_data[key] = 0
                growth_data[key] += 1

        # ترتيب البيانات
        sorted_weeks = sorted(growth_data.keys())

        report = "📈 **معدل نمو المستخدمين**\n\n"

        if not sorted_weeks:
            report += "⚠️ لا توجد بيانات كافية لتحليل النمو."
        else:
            # عرض آخر 8 أسابيع
            recent_weeks = sorted_weeks[-8:]

            total_growth = 0
            for week in recent_weeks:
                count = growth_data[week]
                total_growth += count
                report += f"📅 **أسبوع {week}:** {count} مستخدم جديد\n"

            # حساب المتوسط
            avg_growth = total_growth / len(recent_weeks) if recent_weeks else 0

            report += f"\n📊 **المتوسط الأسبوعي:** {avg_growth:.1f} مستخدم"
            report += f"\n👥 **إجمالي المستخدمين:** {len(users)}"

            # توقعات النمو
            if avg_growth > 0:
                weekly_growth_rate = (avg_growth / len(users)) * 100 if users else 0
                report += f"\n📈 **معدل النمو الأسبوعي:** {weekly_growth_rate:.1f}%"

        report += f"\n\n⏰ **تاريخ التقرير:** {today.strftime('%Y-%m-%d')}"

        await update.message.reply_text(report, parse_mode="Markdown")


# ============ معالجة الـ Callbacks ============
class CallbackHandler:
    @staticmethod
    async def course_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالجة كولباك الدورة"""
        query = update.callback_query
        await query.answer()
        data = query.data
        parts = data.split("|")
        action = parts[0]

        if action == "course_open":
            await CallbackHandler.handle_course_open(query, parts)
        elif action == "course_modules":
            await CallbackHandler.handle_course_modules(query, parts)
        elif action == "course_select_module":
            await CallbackHandler.handle_course_select_module(query, parts)
        elif action == "course_next":
            await CallbackHandler.handle_course_next(query, parts)
        elif action == "course_final_exam":
            await CallbackHandler.handle_course_final_exam(query, parts)
        elif action == "course_final_answer":
            await CallbackHandler.handle_course_final_answer(query, parts)
        elif action == "course_start":
            await CallbackHandler.handle_course_start(query, parts)
        elif action == "course_quiz":
            await CallbackHandler.handle_course_quiz(query, parts)
        elif action == "course_answer":
            await CallbackHandler.handle_course_answer(query, parts)
        elif action == "course_back":
            await query.message.reply_text("🏠 رجوع للرئيسية.", reply_markup=Keyboards.main_menu())
        elif action == "noop":
            await query.answer()

    @staticmethod
    async def handle_course_open(query, parts):
        """فتح الدورة"""
        _, uid_s, sem = parts
        uid = int(uid_s)
        user, idx = UserManager.get_user_and_index(uid)
        if not user:
            await query.edit_message_text("⚠️ المستخدم غير مسجل.")
            return

        course = CourseManager.get_course_for_user(user)
        if not course:
            await query.edit_message_text("🚫 الدورة غير موجودة لفصلك.")
            return

        CourseManager.ensure_user_course_progress(user)
        UserManager.save_users()

        cm = user.get("current_module", 1)
        total_modules = len(course.get("modules", []))
        completed_modules = user.get("completed_modules", [])

        module_title = course["modules"][cm - 1]["title"] if cm - 1 < len(course["modules"]) else "محور غير معروف"

        progress_text = f"📘 الدورة: {course.get('course_name')}\n"
        progress_text += f"المحور الحالي: {cm}/{total_modules} — {module_title}\n"
        progress_text += f"المحاور المكتملة: {len(completed_modules)}/{total_modules}"

        kb = [
            [InlineKeyboardButton(f"▶️ بدأ المحور الحالي ({cm})", callback_data=f"course_start|{uid}|{sem}|{cm}")],
            [InlineKeyboardButton("📋 عناوين المحاور", callback_data=f"course_modules|{uid}|{sem}")],
        ]

        if len(completed_modules) >= total_modules:
            kb.append([InlineKeyboardButton("🎓 الاختبار النهائي", callback_data=f"course_final_exam|{uid}|{sem}")])
        else:
            kb.append([InlineKeyboardButton("⏭️ المحور التالي", callback_data=f"course_next|{uid}|{sem}|{cm}")])

        kb.append([InlineKeyboardButton("🏠 رجوع للرئيسية", callback_data=f"course_back|{uid}")])

        await query.edit_message_text(progress_text, reply_markup=InlineKeyboardMarkup(kb))

    @staticmethod
    async def handle_course_modules(query, parts):
        """عرض محاور الدورة"""
        _, uid_s, sem = parts
        uid = int(uid_s)
        user, idx = UserManager.get_user_and_index(uid)
        if not user:
            await query.edit_message_text("⚠️ المستخدم غير مسجل.")
            return

        course = CourseManager.get_course_for_user(user)
        if not course:
            await query.edit_message_text("🚫 الدورة غير موجودة لفصلك.")
            return

        modules = course.get("modules", [])
        completed_modules = user.get("completed_modules", [])

        kb = []
        for i, module in enumerate(modules, 1):
            status = "✅" if i in completed_modules else "⏳"
            kb.append([InlineKeyboardButton(f"{status} {i}. {module['title']}",
                                            callback_data=f"course_select_module|{uid}|{sem}|{i}")])

        kb.append([InlineKeyboardButton("⬅️ رجوع", callback_data=f"course_open|{uid}|{sem}")])

        text = f"📋 محاور الدورة: {course.get('course_name')}\n\n✅ = مكتمل | ⏳ = لم يكتمل\n\nاختر المحور الذي تريد دراسته:"
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb))

    @staticmethod
    async def handle_course_select_module(query, parts):
        """اختيار محور معين"""
        _, uid_s, sem, module_no_s = parts
        uid = int(uid_s)
        module_no = int(module_no_s)
        user, idx = UserManager.get_user_and_index(uid)
        if not user:
            await query.edit_message_text("⚠️ المستخدم غير مسجل.")
            return

        course = CourseManager.get_course_for_user(user)
        if not course:
            await query.edit_message_text("🚫 الدورة غير موجودة لفصلك.")
            return

        user["current_module"] = module_no
        UserManager.save_users()

        module_title = course["modules"][module_no - 1]["title"] if module_no - 1 < len(
            course["modules"]) else "محور غير معروف"
        text = f"📘 الدورة: {course.get('course_name')}\nالمحور الحالي: {module_no} — {module_title}"

        completed_modules = user.get("completed_modules", [])
        total_modules = len(course.get("modules", []))

        kb = [
            [InlineKeyboardButton(f"▶️ بدأ المحور الحالي ({module_no})",
                                  callback_data=f"course_start|{uid}|{sem}|{module_no}")],
            [InlineKeyboardButton("📋 عناوين المحاور", callback_data=f"course_modules|{uid}|{sem}")],
        ]

        if len(completed_modules) >= total_modules:
            kb.append([InlineKeyboardButton("🎓 الاختبار النهائي", callback_data=f"course_final_exam|{uid}|{sem}")])
        else:
            kb.append([InlineKeyboardButton("⏭️ المحور التالي", callback_data=f"course_next|{uid}|{sem}|{module_no}")])

        kb.append([InlineKeyboardButton("🏠 رجوع للرئيسية", callback_data=f"course_back|{uid}")])

        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb))

    @staticmethod
    async def handle_course_next(query, parts):
        """المحور التالي"""
        _, uid_s, sem, current_module_s = parts
        uid = int(uid_s)
        current_module = int(current_module_s)
        user, idx = UserManager.get_user_and_index(uid)
        if not user:
            await query.edit_message_text("⚠️ المستخدم غير مسجل.")
            return

        course = CourseManager.get_course_for_user(user)
        if not course:
            await query.edit_message_text("🚫 الدورة غير موجودة لفصلك.")
            return

        total_modules = len(course.get("modules", []))
        new_module = current_module + 1

        if new_module > total_modules:
            user["current_module"] = new_module
            UserManager.save_users()

            kb = [
                [InlineKeyboardButton("🎓 الاختبار النهائي", callback_data=f"course_final_exam|{uid}|{sem}")],
                [InlineKeyboardButton("📋 عناوين المحاور", callback_data=f"course_modules|{uid}|{sem}")],
                [InlineKeyboardButton("🏠 رجوع للرئيسية", callback_data=f"course_back|{uid}")]
            ]
            text = f"📘 الدورة: {course.get('course_name')}\n✅ لقد انتهيت من جميع المحاور ({total_modules})\n\nيمكنك الآن اجتياز الاختبار النهائي للحصول على الشهادة."
            await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb))
            return

        user["current_module"] = new_module
        UserManager.save_users()

        module_title = course["modules"][new_module - 1]["title"] if new_module - 1 < len(
            course["modules"]) else "محور غير معروف"
        text = f"📘 الدورة: {course.get('course_name')}\nالمحور الحالي: {new_module}/{total_modules} — {module_title}"

        completed_modules = user.get("completed_modules", [])

        kb = [
            [InlineKeyboardButton(f"▶️ بدأ المحور الحالي ({new_module})",
                                  callback_data=f"course_start|{uid}|{sem}|{new_module}")],
            [InlineKeyboardButton("📋 عناوين المحاور", callback_data=f"course_modules|{uid}|{sem}")],
        ]

        if len(completed_modules) >= total_modules:
            kb.append([InlineKeyboardButton("🎓 الاختبار النهائي", callback_data=f"course_final_exam|{uid}|{sem}")])
        else:
            kb.append([InlineKeyboardButton("⏭️ المحور التالي", callback_data=f"course_next|{uid}|{sem}|{new_module}")])

        kb.append([InlineKeyboardButton("🏠 رجوع للرئيسية", callback_data=f"course_back|{uid}")])

        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb))

    @staticmethod
    async def handle_course_start(query, parts):
        """بدء المحور"""
        _, uid_s, sem, module_no_s = parts
        uid = int(uid_s)
        module_no = int(module_no_s)
        user, idx = UserManager.get_user_and_index(uid)
        if not user:
            await query.edit_message_text("⚠️ المستخدم غير مسجل.")
            return

        course = CourseManager.get_course_for_user(user)
        if not course:
            await query.edit_message_text("🚫 الدورة غير موجودة لفصلك.")
            return

        modules = course.get("modules", [])
        if module_no - 1 >= len(modules):
            await query.edit_message_text("⚠️ المحور غير موجود.")
            return

        mod = modules[module_no - 1]
        text = f"*{mod.get('title')}*\n\n{mod.get('content', '(لا يوجد محتوى)')}"

        kb = [
            [InlineKeyboardButton("📝 بدء الاختبار", callback_data=f"course_quiz|{uid}|{sem}|{module_no}")],
            [InlineKeyboardButton("⬅️ رجوع لقائمة الدورة", callback_data=f"course_open|{uid}|{sem}")]
        ]

        try:
            await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
        except:
            await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb))

    @staticmethod
    async def handle_course_quiz(query, parts):
        """بدء اختبار المحور"""
        _, uid_s, sem, module_no_s = parts
        uid = int(uid_s)
        module_no = int(module_no_s)
        user, idx = UserManager.get_user_and_index(uid)
        if not user:
            await query.edit_message_text("⚠️ المستخدم غير مسجل.")
            return

        course = CourseManager.get_course_for_user(user)
        if not course:
            await query.edit_message_text("🚫 الدورة غير موجودة لفصلك.")
            return

        modules = course.get("modules", [])
        if module_no - 1 >= len(modules):
            await query.edit_message_text("⚠️ المحور غير موجود.")
            return

        module = modules[module_no - 1]
        questions = module.get("quiz", {}).get("questions", [])
        if not questions:
            await query.edit_message_text(
                "⚠️ لا يوجد اختبار لهذا المحور.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("⬅️ رجوع لقائمة الدورة",
                                           callback_data=f"course_open|{uid}|{sem}")]]
                )
            )
            return

        user["quiz_session"] = {
            "course_sem": sem,
            "module_no": module_no,
            "q_index": 0,
            "score": 0
        }
        UserManager.save_users()

        q = questions[0]
        qtext = f"سؤال 1/{len(questions)}:\n{q.get('question')}"
        kb = [[InlineKeyboardButton(opt, callback_data=f"course_answer|{uid}|{sem}|{module_no}|0|{i}")]
              for i, opt in enumerate(q.get("options", []))]
        await query.edit_message_text(qtext, reply_markup=InlineKeyboardMarkup(kb))

    @staticmethod
    async def handle_course_answer(query, parts):
        """معالجة إجابة السؤال"""
        _, uid_s, sem, module_no_s, qidx_s, optidx_s = parts
        uid = int(uid_s)
        module_no = int(module_no_s)
        qidx = int(qidx_s)
        optidx = int(optidx_s)
        user, idx = UserManager.get_user_and_index(uid)
        if not user:
            await query.edit_message_text("⚠️ المستخدم غير مسجل.")
            return

        course = CourseManager.get_course_for_user(user)
        if not course:
            await query.edit_message_text("🚫 الدورة غير موجودة لفصلك.")
            return

        modules = course.get("modules", [])
        if module_no - 1 >= len(modules):
            await query.edit_message_text("⚠️ المحور غير موجود.")
            return

        module = modules[module_no - 1]
        questions = module.get("quiz", {}).get("questions", [])
        if qidx >= len(questions):
            await query.edit_message_text("⚠️ خطأ فالسؤال.")
            return

        sess = user.get("quiz_session") or {}
        if sess.get("module_no") != module_no:
            await query.edit_message_text("⚠️ الجلسة منتهية أو غير متوافقة، افتح الاختبار من جديد.")
            return

        if optidx == questions[qidx].get("answer"):
            sess["score"] = sess.get("score", 0) + 1

        sess["q_index"] += 1
        user["quiz_session"] = sess
        UserManager.save_users()

        if sess["q_index"] < len(questions):
            next_q = questions[sess['q_index']]
            qtext = f"سؤال {sess['q_index'] + 1}/{len(questions)}:\n{next_q.get('question')}"
            kb = [[InlineKeyboardButton(opt,
                                        callback_data=f"course_answer|{uid}|{sem}|{module_no}|{sess['q_index']}|{i}")]
                  for i, opt in enumerate(next_q.get("options", []))]
            await query.edit_message_text(qtext, reply_markup=InlineKeyboardMarkup(kb))
        else:
            score = sess.get("score", 0)
            user["quiz_session"] = None

            if score >= (len(questions) * 0.7):
                if "completed_modules" not in user:
                    user["completed_modules"] = []
                if module_no not in user["completed_modules"]:
                    user["completed_modules"].append(module_no)

                next_module = module_no + 1
                if next_module <= len(course.get("modules", [])):
                    user["current_module"] = next_module

            UserManager.save_users()

            message_text = f"✅ انتهيت من الاختبار. النتيجة: {score}/{len(questions)}"

            if score >= (len(questions) * 0.7):
                message_text += "\n🎉 لقد أكملت هذا المحور بنجاح!"

                if module_no == len(course.get("modules", [])):
                    message_text += "\n\n🎓 لقد أكملت جميع المحاور! يمكنك الآن اجتياز الاختبار النهائي."
                    kb = [
                        [InlineKeyboardButton("🎓 الاختبار النهائي", callback_data=f"course_final_exam|{uid}|{sem}")],
                        [InlineKeyboardButton("📋 عناوين المحاور", callback_data=f"course_modules|{uid}|{sem}")],
                        [InlineKeyboardButton("🏠 رجوع للرئيسية", callback_data=f"course_back|{uid}")]
                    ]
                else:
                    kb = [
                        [InlineKeyboardButton("⏭️ المحور التالي",
                                              callback_data=f"course_start|{uid}|{sem}|{next_module}")],
                        [InlineKeyboardButton("📋 عناوين المحاور", callback_data=f"course_modules|{uid}|{sem}")],
                        [InlineKeyboardButton("🏠 رجوع للرئيسية", callback_data=f"course_back|{uid}")]
                    ]
            else:
                message_text += "\n❌ لم تحقق النسبة المطلوبة (70%) لإكمال المحور. يمكنك المحاولة مرة أخرى."
                kb = [
                    [InlineKeyboardButton("🔄 إعادة الاختبار", callback_data=f"course_quiz|{uid}|{sem}|{module_no}")],
                    [InlineKeyboardButton("📋 عناوين المحاور", callback_data=f"course_modules|{uid}|{sem}")],
                    [InlineKeyboardButton("🏠 رجوع للرئيسية", callback_data=f"course_back|{uid}")]
                ]

            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup(kb))

    @staticmethod
    async def handle_course_final_exam(query, parts):
        """الاختبار النهائي"""
        _, uid_s, sem = parts
        uid = int(uid_s)
        user, idx = UserManager.get_user_and_index(uid)
        if not user:
            await query.edit_message_text("⚠️ المستخدم غير مسجل.")
            return

        course = CourseManager.get_course_for_user(user)
        if not course:
            await query.edit_message_text("🚫 الدورة غير موجودة لفصلك.")
            return

        final_exam = course.get("final_exam", {})
        questions = final_exam.get("questions", [])

        if not questions:
            await query.edit_message_text(
                "⚠️ لا يوجد اختبار نهائي لهذه الدورة بعد.\n\n"
                "سيتم إضافة الاختبار النهائي قريباً بإذن الله.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("⬅️ رجوع لقائمة الدورة", callback_data=f"course_open|{uid}|{sem}")],
                    [InlineKeyboardButton("🏠 الرئيسية", callback_data=f"course_back|{uid}")]
                ])
            )
            return

        total_modules = len(course.get("modules", []))
        completed_modules = user.get("completed_modules", [])

        if len(completed_modules) < total_modules:
            modules_left = total_modules - len(completed_modules)
            await query.edit_message_text(
                f"⏳ لم تنتهِ من جميع المحاور بعد!\n\n"
                f"المحاور المتبقية: {modules_left}\n"
                f"يجب إنهاء جميع المحاور أولاً قبل التقدم للاختبار النهائي.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("➡️ أكمل المحاور المتبقية",
                                          callback_data=f"course_modules|{uid}|{sem}")],
                    [InlineKeyboardButton("⬅️ رجوع", callback_data=f"course_open|{uid}|{sem}")]
                ])
            )
            return

        user["final_exam_session"] = {
            "course_sem": sem,
            "q_index": 0,
            "score": 0,
            "total_questions": len(questions)
        }
        UserManager.save_users()

        q = questions[0]
        qtext = f"🎓 الاختبار النهائي\nسؤال 1/{len(questions)}:\n{q.get('question')}"
        kb = [[InlineKeyboardButton(opt, callback_data=f"course_final_answer|{uid}|{sem}|0|{i}")]
              for i, opt in enumerate(q.get("options", []))]
        await query.edit_message_text(qtext, reply_markup=InlineKeyboardMarkup(kb))

    @staticmethod
    async def handle_course_final_answer(query, parts):
        """معالجة إجابة الاختبار النهائي"""
        _, uid_s, sem, qidx_s, optidx_s = parts
        uid = int(uid_s)
        qidx = int(qidx_s)
        optidx = int(optidx_s)
        user, idx = UserManager.get_user_and_index(uid)
        if not user:
            await query.edit_message_text("⚠️ المستخدم غير مسجل.")
            return

        course = CourseManager.get_course_for_user(user)
        if not course:
            await query.edit_message_text("🚫 الدورة غير موجودة لفصلك.")
            return

        final_exam = course.get("final_exam", {})
        questions = final_exam.get("questions", [])

        if qidx >= len(questions):
            await query.edit_message_text("⚠️ خطأ في السؤال.")
            return

        sess = user.get("final_exam_session") or {}
        if sess.get("course_sem") != sem:
            await query.edit_message_text("⚠️ الجلسة منتهية أو غير متوافقة، افتح الاختبار من جديد.")
            return

        if optidx == questions[qidx].get("answer"):
            sess["score"] = sess.get("score", 0) + 1

        sess["q_index"] += 1
        user["final_exam_session"] = sess
        UserManager.save_users()

        if sess["q_index"] < len(questions):
            next_q = questions[sess['q_index']]
            qtext = f"🎓 الاختبار النهائي\nسؤال {sess['q_index'] + 1}/{len(questions)}:\n{next_q.get('question')}"
            kb = [[InlineKeyboardButton(opt, callback_data=f"course_final_answer|{uid}|{sem}|{sess['q_index']}|{i}")]
                  for i, opt in enumerate(next_q.get("options", []))]
            await query.edit_message_text(qtext, reply_markup=InlineKeyboardMarkup(kb))
        else:
            score = sess.get("score", 0)
            total_questions = len(questions)
            percentage = (score / total_questions) * 100

            if "final_exam_session" in user:
                del user["final_exam_session"]

            if percentage >= 70:
                user["completed_courses"] = user.get("completed_courses", [])
                if sem not in user["completed_courses"]:
                    user["completed_courses"].append(sem)

                user["certificates"] = user.get("certificates", [])
                user["certificates"].append(course.get("course_name"))

                if "final_exam" in course:
                    course["final_exam"]["passed"] = True

                pdf_link = course.get("pdf_file")

                if pdf_link:
                    text = f"🎉 مبروك! لقد نجحت في الاختبار النهائي\n\n" \
                           f"النتيجة: {score}/{total_questions} ({percentage:.1f}%)\n\n" \
                           f"✅ هده جائزتك ايها الطالب المجد.\n\n" \
                           f"📚 رابط الكتاب: {pdf_link}"
                else:
                    text = f"🎉 مبروك! لقد نجحت في الاختبار النهائي\n\n" \
                           f"النتيجة: {score}/{total_questions} ({percentage:.1f}%)\n\n" \
                           f"✅ هده جائزتك ايها الطالب المجد.\n\n" \
                           f"⚠️ لا يوجد كتاب مرتبط بهذه الدورة"
            else:
                text = f"⚠️ لم تحقق النسبة المطلوبة للنجاح\n\n" \
                       f"النتيجة: {score}/{total_questions} ({percentage:.1f}%)\n\n" \
                       f"🔁 تحتاج إلى 70% على الأقل للنجاح. يمكنك إعادة المحاولة لاحقاً."

            UserManager.save_users()

            kb = [
                [InlineKeyboardButton("⬅️ رجوع لقائمة الدورة", callback_data=f"course_open|{uid}|{sem}")],
                [InlineKeyboardButton("🏠 الرئيسية", callback_data=f"course_back|{uid}")]
            ]
            await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb))


# ============ معالجة الـ Callbacks الأخرى ============
async def handle_faq_callback(update, context):
    """معالجة الأسئلة الشائعة"""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "add_new_question":
        user_id = query.from_user.id
        user = UserManager.find_user(user_id)
        if user:
            user["awaiting_new_question"] = True
            UserManager.save_users()
            await query.message.reply_text("✍️ اكتب سؤالك هنا:", reply_markup=Keyboards.back_only())
        return

    elif data == "show_my_group":
        user_id = query.from_user.id
        user = UserManager.find_user(user_id)
        if user:
            year = str(user.get("year", ""))
            sem = str(user.get("semester", ""))
            major = str(user.get("major", ""))

            link = texts.get("groups", {}).get(year, {}).get(sem, {}).get(major)
            if link:
                await query.message.reply_text(f"👥 رابط مجموعتك:\n{link}")
            else:
                await query.message.reply_text("⚠️ لا يوجد رابط لمجموعتك حالياً.")
        return

    if data.startswith("faq_"):
        idx = int(data.split("_")[1])
        faqs = texts.get("faqs", [])
        if 0 <= idx < len(faqs):
            answer = faqs[idx]["answer"]
            await query.edit_message_text(f"❓ السؤال: {faqs[idx]['question']}\n\n💡 الجواب: {answer}")
        else:
            await query.edit_message_text("⚠️ السؤال لم يعد متوفر.")


async def handle_question_callback(update, context):
    """معالجة أسئلة المشرفين"""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("pdf_"):
        await handle_pdf_callback(update, context)
        return

    action, index = data.split("_")
    index = int(index)

    questions = FileManager.safe_load_json(Config.QUESTIONS_FILE, default=[])

    if index >= len(questions):
        await query.edit_message_text("⚠️ السؤال لم يعد متوفر.")
        return

    question = questions[index]

    if action == "approve":
        context.user_data['pending_faq'] = {
            'index': index,
            'question': question['question']
        }
        await query.edit_message_text(f"✍️ اكتب الجواب للسؤال:\n{question['question']}")

    elif action == "reject":
        questions.pop(index)
        FileManager.safe_save_json(Config.QUESTIONS_FILE, questions)
        await query.edit_message_text(f"❌ تم رفض السؤال")


async def handle_pdf_callback(update, context):
    """معالجة تحميل الملفات PDF"""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("pdf_"):
        pdf_name = data[4:]
        pdf_url = pdf_texts.get(pdf_name)

        if pdf_url:
            await query.message.reply_text(f"📚 {pdf_name}\n\n🔗 رابط التحميل: {pdf_url}")
        else:
            await query.message.reply_text("⚠️ الرابط غير متوفر حالياً.")


# ============ إعداد التطبيق الرئيسي ============
def setup_handlers(app):
    """إعداد جميع الـ handlers"""

    # أوامر المشرفين المحسنة
    app.add_handler(CommandHandler("announce", AdminManager.admin_announce))
    app.add_handler(CommandHandler("stats", AdminManager.admin_stats))
    app.add_handler(CommandHandler("poll", AdminManager.admin_poll))
    app.add_handler(CommandHandler("questions", AdminManager.admin_questions))
    app.add_handler(CommandHandler("add_faq", AdminManager.add_faq))
    app.add_handler(CommandHandler("edit_faq", AdminManager.edit_faq))
    app.add_handler(CommandHandler("delete_faq", AdminManager.delete_faq))
    app.add_handler(CommandHandler("list_faqs", AdminManager.list_faqs))
    app.add_handler(CommandHandler("admin_help", AdminManager.admin_help))

    # أوامر الأمان الجديدة
    app.add_handler(CommandHandler("admins_list", SecurityManager.admins_list))
    app.add_handler(CommandHandler("add_admin", SecurityManager.add_admin))
    app.add_handler(CommandHandler("remove_admin", SecurityManager.remove_admin))

    # الأوامر الإدارية الجديدة
    app.add_handler(CommandHandler("weekly_report", AdminReports.weekly_report))
    app.add_handler(CommandHandler("activity", AdminReports.user_activity))
    app.add_handler(CommandHandler("growth", AdminReports.growth_stats))

    # الـ Callbacks
    app.add_handler(CallbackQueryHandler(handle_faq_callback, pattern=r"^faq_"))
    app.add_handler(CallbackQueryHandler(handle_faq_callback, pattern=r"^(add_new_question|show_my_group)"))
    app.add_handler(CallbackQueryHandler(handle_question_callback, pattern=r"^(approve_|reject_)"))
    app.add_handler(CallbackQueryHandler(CallbackHandler.course_callback_handler, pattern=r"^course_"))
    app.add_handler(CallbackQueryHandler(handle_pdf_callback, pattern=r"^pdf_"))
    app.add_handler(CallbackQueryHandler(handle_copy_command_callback, pattern=r"^copy_command_"))

    # معالجة الرسائل
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))


def initialize_data():
    """تهيئة البيانات"""
    global users
    # تنظيف بيانات المستخدمين من الحقول غير الضرورية
    cleaned_users = []
    for user in users:
        cleaned_user = {
            "user_id": user.get("user_id"),
            "username": user.get("username"),
            "year": user.get("year"),
            "semester": user.get("semester"),
            "major": user.get("major"),
            "current_module": user.get("current_module", 1),
            "completed_modules": user.get("completed_modules", []),
            "completed_courses": user.get("completed_courses", []),
            "certificates": user.get("certificates", []),
            "sent_reminders": user.get("sent_reminders", {}),
            "last_active": user.get("last_active", datetime.now(timezone.utc).isoformat()),
            "message_count": user.get("message_count", 0)
        }
        cleaned_users.append(cleaned_user)

    users = cleaned_users
    UserManager.save_users()

    print(f"👑 عدد المشرفين: {len(Config.ADMINS)}")


async def scheduled_reminders(context: ContextTypes.DEFAULT_TYPE):
    """الدالة المجدولة لإرسال التذكيرات"""
    try:
        # 🔥 تنظيف التذكيرات القديمة أولاً
        ReminderSystem.cleanup_old_reminders()

        # 🔥 إرسال تذكيرات المحاضرات
        await ReminderSystem.send_lecture_reminders(context)

    except Exception as e:
        print(f"خطأ في التذكيرات المجدولة: {e}")


def main():
    """الدالة الرئيسية لتشغيل البوت"""

    # تهيئة البيانات
    initialize_data()

    # إنشاء التطبيق مع JobQueue
    app = Application.builder().token(Config.TOKEN).build()

    # 🔥 إعداد التذكيرات المجدولة
    job_queue = app.job_queue
    if job_queue:
        # 🔥 تشغيل كل دقيقة للتحقق من التذكيرات
        job_queue.run_repeating(
            scheduled_reminders,
            interval=60,  # كل دقيقة
            first=10  # بعد 10 ثواني من التشغيل
        )
        print("✅ نظام التذكيرات التلقائية مفعل")

    # إعداد الـ handlers
    setup_handlers(app)

    print("🤖 البوت شغال...")

    # تشغيل البوت
    app.run_polling()


if __name__ == "__main__":
    main()
