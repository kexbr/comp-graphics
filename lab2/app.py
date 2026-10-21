import streamlit as st
import numpy as np
from PIL import Image
import processing as pr

st.set_page_config(page_title="Лабораторная 2: Вариант 11", layout="wide")

st.title("Обработка изображений (Написано с нуля, без OpenCV)")
st.sidebar.title("Меню и настройки")

mode = st.sidebar.radio(
    "Выберите метод обработки:",
    ("1. Поэлементные операции и контраст", "2. Локальная пороговая обработка")
)

uploaded_file = st.sidebar.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    image_np = np.array(image_pil)
    
    st.markdown("### Исходное изображение")
    st.image(image_np, width=400)

    if mode == "1. Поэлементные операции и контраст":
        st.markdown("### Результаты обработки")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Линейное контрастирование (Manual)**")
            st.info("Формула: (pixel - min) * (255 / (max - min)). Применена к каждому каналу.")
            res_contrast = pr.manual_linear_contrast(image_np)
            st.image(res_contrast, use_container_width=True)
            
        with col2:
            st.markdown("**Гамма-коррекция (Manual)**")
            gamma = st.slider("Значение Гаммы", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            res_gamma = pr.manual_gamma_correction(image_np, gamma)
            st.image(res_gamma, use_container_width=True)

    elif mode == "2. Локальная пороговая обработка":
        st.markdown("### Сравнение 2-х методов локальной пороговой обработки")
        
        block_size = st.sidebar.slider("Размер блока", min_value=3, max_value=51, value=11, step=2)
        c_val = st.sidebar.slider("Константа C", min_value=-100, max_value=100, value=2, step=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Метод 1: Adaptive Mean (Через интегральное изображение)**")
            res_mean = pr.manual_adaptive_mean(image_np, block_size, c_val)
            st.image(res_mean, use_container_width=True)
            
        with col2:
            st.markdown("**Метод 2: Adaptive Gaussian (Сепарабельная 1D свертка)**")
            res_gauss = pr.manual_adaptive_gaussian(image_np, block_size, c_val)
            st.image(res_gauss, use_container_width=True)
            
else:
    st.info("Пожалуйста, загрузите изображение.")