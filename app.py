"""Giao diện demo dùng checkpoint DistilBERT đã fine-tune."""
import json
from pathlib import Path
import streamlit as st
from src.common import ROOT
from src.predict import load_model, predict

st.set_page_config(page_title='COVID-19 | Nhóm 5', page_icon='📄')
st.title('Phát hiện tin giả về COVID-19')
st.caption('Xử lý ngôn ngữ tự nhiên • CONSTRAINT@AAAI2021 • Nhóm 5')
st.write('Nhập một bài đăng tiếng Anh để thử mô hình DistilBERT.')
model_dir = st.sidebar.text_input('Thư mục checkpoint', str(ROOT/'outputs/distilbert/seed_42/model'))
st.sidebar.caption('Phụ trách DistilBERT: Lương Hoàng Bảo Quân — 24DH111520')
st.info('Đây là phân loại văn bản theo dữ liệu nghiên cứu, không phải công cụ xác minh thông tin y tế.')

@st.cache_resource
def get_model(path):
    return load_model(path)

available = (Path(model_dir)/'config.json').exists()
if not available:
    st.warning('Chưa có mô hình đã train. Hãy tải thư mục model từ Kaggle về và nhập đúng đường dẫn bên trái.')
text = st.text_area('Nội dung tiếng Anh', height=180, placeholder='Nhập văn bản cần phân loại...')
if st.button('Dự đoán', disabled=not available):
    if not text.strip():
        st.warning('Bạn chưa nhập nội dung.')
    else:
        try:
            cfg_path = Path(model_dir).parent/'run_config.json'
            max_length = json.loads(cfg_path.read_text())['max_length'] if cfg_path.exists() else 128
            tokenizer, model = get_model(model_dir)
            probabilities = predict(text, tokenizer, model, max_length)
            label = max(probabilities, key=probabilities.get)
            st.subheader('Nhãn dự đoán: '+label.upper())
            st.bar_chart(probabilities)
            st.caption('Điểm softmax thể hiện mức ưu tiên giữa hai nhãn; không phải xác suất thông tin đúng ngoài thực tế.')
        except Exception as error:
            st.error(str(error))
