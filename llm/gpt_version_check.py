import openai

client = openai.OpenAI(api_key="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")  # ← 본인의 API 키 입력

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": ""}],
    )
    print("✅ GPT-3.5-turbo 사용 가능합니다!")
except Exception as e:
    print("❌ GPT-3.5-turbo 사용 불가:", e)
