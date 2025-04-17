import openai
import streamlit as st
import PyPDF2

# 定義 talk 方法，用於文案撰寫與優化
def HR_talk(api_key, msg_ar, company_name, job_description, required_skills, experience_required, email, selected_job):

    # 設置 API Key
    openai.api_key = api_key
    client = openai.OpenAI(api_key=api_key)
    # 使用 OpenAI API 進行對話生成，並將生成的結果返回
    response = client.chat.completions.create(

        model="gpt-3.5-turbo",

        messages=[
            {"role": "user", "content": f"生成一個徵才貼文,公司名稱為{company_name}，工作職稱為{selected_job}\
                                        1. 工作能力需求：{required_skills}，\
                                        2. 工作內容為：{job_description}的風格撰寫\
                                        3. 經驗需求為：{experience_required}，\
                                        4. 求職信投地處為：{email}\
                                        5. 其他公司介紹內容或貼文生成需求：{msg_ar}\
                                        利用以上條件生成。"}],
        temperature=0.4,  # 生成文本的溫度，控制模型生成多樣性，0.0 為完全固定，1.0 為完全隨機
        max_tokens=1024,  # 產生的文本長度上限，控制模型生成的長度
        top_p=1,  # top-p 生成文本，控制模型生成文本的機率分布
        frequency_penalty=0.6,  # 重複文本的懲罰程度，控制模型避免重複生成相似的文本
        presence_penalty=0.6,  # 未出現過的文本的懲罰程度，控制模型傾向於生成已經出現過的文本
    )

    return response.choices[0].message.content

def talk(api_key, msg_ar, selected_job, name, company_name, skills, uploaded_file):
    import openai
    # 設置 API Key
    openai.api_key = api_key
    client = openai.OpenAI(api_key=api_key)

    # 擷取 PDF 的內容（如果有上傳）
    txt_data = '無'
    if uploaded_file is not None:
        txt_data = extract_text_from_pdf(uploaded_file)

    # 建立對話訊息
    user_message = f"""
    請幫我生成求職信:
    姓名：{name}
    公司名稱：{company_name}
    應徵職缺：{selected_job}

    1. 自身技能和過往經驗：{skills}
    2. 求職信需求為：{msg_ar}
    或參考以下 履歷關鍵字生成：{txt_data}
    
    綜合以上資訊產出一封求職信，其中包含自我簡介與投遞動機等，
    
    """

    # 呼叫 GPT API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_message}],
        temperature=0.4,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0.6,
        presence_penalty=0.6,
    )

    return response.choices[0].message.content


def calculate(data, year, employment_type, selected_location, company_size, remote_ratio):
    return 10

def extract_text_from_pdf(file):
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text