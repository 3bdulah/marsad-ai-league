import random

RESPONSES = {
    "food": ["🍔 يمكنك التوجه إلى البوابة 3 لشراء الطعام.", "🍟 الطعام متوفر عند البوابة 7."],
    "bathroom": ["🚻 دورات المياه قرب الممر الشرقي.", "🧼 ستجد الحمام خلف الصف A."],
    "exit": ["🚪 أقرب مخرج هو المخرج الجنوبي.", "🚨 استخدم المخرج رقم 4 في حالات الطوارئ."],
    "score": ["📊 النتيجة الحالية 2 - 1 لصالح الفريق الأحمر.", "⚽ آخر هدف تم تسجيله قبل 5 دقائق."],
    "hi": ["أهلاً بك في ملعب مرصد الذكي! كيف يمكنني مساعدتك؟"],
    "default": ["❓ لم أفهم طلبك، هل يمكنك التوضيح؟"]
}

def get_response(user_input):
    user_input = user_input.lower()
    for key in RESPONSES:
        if key in user_input:
            return random.choice(RESPONSES[key])
    return random.choice(RESPONSES["default"])

if __name__ == "__main__":
    print("🤖 مَرْصَد بوت: أهلاً بك! اسألني عن الطعام، الحمام، الخروج أو النتيجة.")
    while True:
        query = input("أنت: ")
        if query.lower() in ["خروج", "exit", "quit"]:
            print("👋 إلى اللقاء!")
            break
        print("مَرْصَد بوت:", get_response(query))