"""Streamlit приложение — фронтенд для предсказания цен на квартиры."""

import requests
import streamlit as st

import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Kwork ML — Предсказание цен", page_icon="🏠", layout="centered")
st.title("🏠 Предсказание цен на квартиры в Москве")

# --- Auth ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.user_id = None

if not st.session_state.authenticated:
    st.header("Вход")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Пароль", type="password", key="login_password")

    if st.button("Войти"):
        try:
            resp = requests.post(f"{API_BASE}/auth/login", json={"email": email, "password": password})
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.token = data["access_token"]
                st.session_state.user_id = data["user_id"]
                st.session_state.authenticated = True
                st.success("Успешный вход!")
                st.rerun()
            else:
                st.error(f"Ошибка: {resp.json()['detail']}")
        except Exception as e:
            st.error(f"Не удалось подключиться к API: {e}")

    st.divider()
    st.header("Регистрация")
    reg_email = st.text_input("Email", key="reg_email")
    reg_password = st.text_input("Пароль", type="password", key="reg_password")

    if st.button("Зарегистрироваться"):
        try:
            resp = requests.post(f"{API_BASE}/auth/register", json={"email": reg_email, "password": reg_password})
            if resp.status_code == 201:
                st.success("Регистрация успешна! Теперь войдите.")
            else:
                st.error(f"Ошибка: {resp.json()['detail']}")
        except Exception as e:
            st.error(f"Не удалось подключиться к API: {e}")

else:
    st.sidebar.success(f"Войден как: {st.session_state.token[:20]}...")
    if st.sidebar.button("Выйти"):
        st.session_state.authenticated = False
        st.session_state.token = None
        st.session_state.user_id = None
        st.rerun()

    st.header("Новое предсказание")

    col1, col2 = st.columns(2)
    with col1:
        rooms = st.number_input("Комнаты", min_value=1, max_value=10, value=2)
        area = st.number_input("Площадь (м²)", min_value=10.0, max_value=500.0, value=50.0)
        floor = st.number_input("Этаж", min_value=1, max_value=50, value=5)
    with col2:
        total_floors = st.number_input("Всего этажей", min_value=1, max_value=50, value=10)
        year_built = st.number_input("Год постройки", min_value=1900, max_value=2030, value=2005)
        metro_min = st.number_input("Минут до метро", min_value=0.0, max_value=120.0, value=10.0)

    if st.button("Предсказать цену", type="primary"):
        features = {
            "rooms": rooms,
            "area": area,
            "floor": floor,
            "total_floors": total_floors,
            "year_built": year_built,
            "distance_to_metro_min": metro_min,
        }

        headers = {"Authorization": f"Bearer {st.session_state.token}"}

        try:
            resp = requests.post(f"{API_BASE}/predict", json=features, headers=headers)
            if resp.status_code == 201:
                data = resp.json()
                price = data["predicted_price"]
                st.success(f"💰 Предсказанная цена: {price:,.0f} ₽")
                st.info(f"📊 Фичи: {data['input_features']}")
            else:
                st.error(f"Ошибка: {resp.json()['detail']}")
        except Exception as e:
            st.error(f"Не удалось подключиться к API: {e}")

    st.divider()

    st.header("История предсказаний")
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        resp = requests.get(f"{API_BASE}/predict/history", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            if data["predictions"]:
                for pred in data["predictions"]:
                    st.write(f"- **{pred['predicted_price']:,.0f} ₽** — {pred['input_features']}")
            else:
                st.info("История пуста")
        else:
            st.error(f"Ошибка загрузки истории: {resp.status_code}")
    except Exception as e:
        st.error(f"Не удалось загрузить историю: {e}")
