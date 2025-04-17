# 導入必要的 library
import openai
import streamlit as st
from function import talk, HR_talk

from function import calculate
import pandas as pd

import joblib
import os
import numpy as np
# 定義主函數

import streamlit as st
import joblib

@st.cache_resource
def load_pipeline():
    return joblib.load("salary_interval_pred_lgb.pkl")

pipeline = load_pipeline()

model_lower = pipeline['model_lower']
model_upper = pipeline['model_upper']
tfidf = pipeline['tfidf']
svd = pipeline['svd']
encoders = pipeline['encoders']
cat_features = pipeline['cat_features']
tfidf_cols = pipeline['tfidf_cols']


# 預測函數
def predict_salary_interval(input_data: pd.DataFrame) -> pd.DataFrame:
    input_data = input_data.copy()

    # remote_ratio 類別化處理
    conditions = [
        input_data['remote_ratio'] == 0,
        input_data['remote_ratio'] == 50,
        input_data['remote_ratio'] == 100
    ]
    choices = ['Onsite', 'Hybrid', 'Remote']
    input_data['remote_ratio'] = np.select(conditions, choices, default='Unknown')

    # 文字欄位處理與特徵新增
    input_data['residence_location'] = input_data['employee_residence'] + '/' + input_data['company_location']
    input_data['work_year'] = input_data['work_year'].astype(str)
    input_data['job_title'].replace('System Engineer', 'Systems Engineer', inplace=True)
    input_data['job_title'].replace('Solution Engineer', 'Solutions Engineer', inplace=True)
    input_data['job_title_raw'] = input_data['job_title']

    # 處理分類欄位
    for col in cat_features:
        if col in encoders:
            input_data[col] = input_data[col].fillna('None').astype(str)
            input_data[col] = encoders[col].transform(input_data[[col]])

    # 處理 TF-IDF + SVD 特徵
    job_title_tfidf = tfidf.transform(input_data['job_title_raw'])
    job_title_svd = svd.transform(job_title_tfidf)
    tfidf_df = pd.DataFrame(job_title_svd, columns=tfidf_cols, index=input_data.index)

    # 移除未用欄位
    cols2drop = ['employment_type', 'salary_currency', 'employee_residence', 'company_location', 'job_title', 'job_title_raw']
    input_data.drop(columns=cols2drop, inplace=True, errors='ignore')

    # 合併 TF-IDF 特徵
    input_data = pd.concat([input_data, tfidf_df], axis=1)

    # 類別特徵轉型
    for col in cat_features:
        if col in input_data.columns:
            input_data[col] = input_data[col].astype('category')

    # 預測上下限（log 還原）
    y_lower = np.expm1(model_lower.predict(input_data))
    y_upper = np.expm1(model_upper.predict(input_data))
    result = pd.DataFrame({
        '預測下限薪資': y_lower,
        '預測上限薪資': y_upper
    }, index=input_data.index)

    return result




def main():
    data = pd.read_csv("salaries.csv")
    menu = ["人資","求職者"]
    choice = st.sidebar.selectbox("Menu",menu)
    st.subheader("資料科學從業人員的薪資查詢平台​")
    st.write("---") 

    if choice == "人資":
         # 分隔線

        # 如果選擇「人資」，顯示工作內容與需求的填寫表單
        st.subheader("填寫工作內容與需求")

        encoded_result = []  # 初始化空 list
        job_titles = data["job_title"].unique().tolist()
        job_titles.sort()
        selected_job = st.multiselect("你選擇的職缺是：", job_titles, default=None, max_selections=1)
        if selected_job:
            st.write("你選擇的職缺是：", selected_job[0])
            encoded_result.append(selected_job[0])
        # 創建文案輸入區域，請求用戶輸入文案
        
        col1,col2 = st.columns(2)
        with col1:
            year = st.number_input("就業年份",2020,2030)
            encoded_result.append(year)

            ##
            company_locations = data["company_location"].unique().tolist()
            company_locations.sort()
            employee_residence = data["employee_residence"].unique().tolist()
            employee_residence.sort()
            selected_location = st.multiselect("選擇公司地點：", company_locations, default=None, max_selections=1)
            if selected_location:
                st.write("你選擇的地點是：", selected_location[0]) 
                encoded_result.append(selected_location[0])
            ##
            selected_residence = st.multiselect("選擇居住地：", employee_residence , default=None, max_selections=1)
            if selected_residence:
                st.write("你選擇的居住地是：", selected_residence[0]) 
                encoded_result.append(selected_residence[0])

            remote_ratio = st.select_slider("遠端比例",["0","50","100"]) 
            encoded_result.append(remote_ratio)
        with col2:
            ##
            experience_level = st.radio("工作經驗與能力",["EN","MI","SE","EX"]) 
            ##
            encoded_result.append(experience_level)

            
            company_size = st.radio("公司規模",["S","M","L"]) 
            encoded_result.append(company_size)
        if st.button("預測薪資區間"):
            if selected_job and selected_location:
                input_dict = {
                    'work_year': str(year),
                    'experience_level': experience_level.split(" ")[0],

                    'job_title': selected_job[0],
                    'employee_residence': selected_location[0],
                    'remote_ratio': int(remote_ratio),
                    'company_location': selected_location[0],
                    'company_size': company_size
                }

                input_df = pd.DataFrame([input_dict])
                prediction = predict_salary_interval(input_df)



                with st.container():
                    st.markdown("### 📥 輸入資料")
                    st.dataframe(input_df, use_container_width=True, hide_index=True)
                
                    st.markdown("---")  # 分隔線
                
                    st.markdown("### 📊 預測薪資區間")
                    st.dataframe(prediction, use_container_width=True, hide_index=True)

            else:
                st.warning("請選擇職缺、公司地點或居住地。")

        st.write("---") 

        company_name = st.text_input("公司名稱", value = 'ex:台灣行銷研究')
        job_titles1 = data["job_title"].unique().tolist()
        job_titles1.sort()
        selected_job1 = st.multiselect("你選擇的職缺是：", job_titles1, default=None, max_selections=1, key="job_select_1")
        if selected_job1:
            st.write("你選擇的職缺是：", selected_job1[0])

        job_description = st.text_area("工作內容", value = 'ex:資料與研究整理、硬體測試與驗證、演算法開發...')
        required_skills = st.text_area("所需技能（請用逗號分隔）", value = 'ex:Python, 學習能力強、適應力佳，樂於探索新技術, 具備良好的報告撰寫與文件整理能力...')
        experience_required = st.selectbox("所需經驗年數", ["不限", "1-3 年", "3-5 年", "5 年以上"])
        email = st.text_input("投遞履歷的信箱或網址",value = 'ex:xxxx@gmail.com')
        msg_ar = st.text_area("請輸入職缺相關內容或貼文需求",
                            value="ex:彈性上下班時間,扁平化管理,良好的學習與成長環境、條列式內容, 內容輕鬆活潑...")
        if st.button("文案產生"):

            api_key = "自己的API"  # 請輸入您的 API KEY

            # 使用 talk() 函數產生文案
            response = HR_talk(api_key, msg_ar, company_name, job_description, required_skills, experience_required, email, selected_job1)#還需要改request

            # 在 Streamlit 中顯示文案結果 (st.text_area)
            results = st.text_area('文案結果', response, height=300)

    else:
        st.subheader("填寫自身能力與經驗")
        encoded_result = []  # 初始化空 list
        job_titles = data["job_title"].unique().tolist()
        job_titles.sort()
        selected_job = st.multiselect("你選擇的職缺是：", job_titles, default=None, max_selections=1)
        if selected_job:
            st.write("你選擇的職缺是：", selected_job[0])
            encoded_result.append(selected_job[0])
        # 創建文案輸入區域，請求用戶輸入文案
        
        col1,col2 = st.columns(2)
        with col1:
            year = st.number_input("就業年份",2020,2030)
            encoded_result.append(year)

            ##
            company_locations = data["company_location"].unique().tolist()
            company_locations.sort()
            employee_residence = data["employee_residence"].unique().tolist()
            employee_residence.sort()
            selected_location = st.multiselect("選擇公司地點：", company_locations, default=None, max_selections=1)
            if selected_location:
                st.write("你選擇的地點是：", selected_location[0]) 
                encoded_result.append(selected_location[0])
            ##
            selected_residence = st.multiselect("選擇居住地：", employee_residence , default=None, max_selections=1)
            if selected_residence:
                st.write("你選擇的居住地是：", selected_residence[0]) 
                encoded_result.append(selected_residence[0])

            remote_ratio = st.select_slider("遠端比例",["0","50","100"]) 
            encoded_result.append(remote_ratio)
        with col2:
            ##
            experience_level = st.radio("工作經驗與能力",["EN","MI","SE","EX"]) 
            ##
            encoded_result.append(experience_level)

            
            company_size = st.radio("公司規模",["S","M","L"]) 
            encoded_result.append(company_size)
        if st.button("預測薪資區間"):
            if selected_job and selected_location:
                input_dict = {
                    'work_year': str(year),
                    'experience_level': experience_level.split(" ")[0],

                    'job_title': selected_job[0],
                    'employee_residence': selected_location[0],
                    'remote_ratio': int(remote_ratio),
                    'company_location': selected_location[0],
                    'company_size': company_size
                }

                input_df = pd.DataFrame([input_dict])
                prediction = predict_salary_interval(input_df)



                with st.container():
                    st.markdown("### 📥 輸入資料")
                    st.dataframe(input_df, use_container_width=True, hide_index=True)
                
                    st.markdown("---")  # 分隔線
                
                    st.markdown("### 📊 預測薪資區間")
                    st.dataframe(prediction, use_container_width=True, hide_index=True)

            else:
                st.warning("請選擇職缺、公司地點或居住地。")
        st.write("---")
        name = st.text_input("姓名", value = '請填姓名')
        company_name = st.text_input("想投遞履歷的公司", value = 'ex:台灣行銷研究')
        job_titles2 = data["job_title"].unique().tolist()
        job_titles2.sort()
        selected_job2 = st.multiselect("你選擇的職缺是：", job_titles2, default=None, max_selections=1, key="job_select_2")
        if selected_job2:
            st.write("你選擇的職缺是：", selected_job2[0])
        skills = st.text_area("個人資訊、能力和過往經驗等", value = 'ex:畢業於XX大學, 專精Python, SQL, VBA...')
        msg_ar = st.text_area("請輸入您求職信的需求",
                            value="ex:有禮貌且自信, 詳述自身經歷與動機並結合先前經歷...")
        # 檔案上傳（限制檔案類型）
        uploaded_file = st.file_uploader("可選擇上傳履歷", type=["pdf"])
        
        if st.button("文案產生"):

            api_key = "自己的API"  # 請輸入您的 API KEY

            # 使用 talk() 函數產生文案
            response = talk(api_key, msg_ar, selected_job2, name, company_name, skills, uploaded_file)#還需要改request

            # 在 Streamlit 中顯示文案結果 (st.text_area)
            results = st.text_area('文案結果', response, height=300)

if __name__ == '__main__':
    main()
