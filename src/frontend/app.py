"""Streamlit приложение — фронтенд для предсказания цен на квартиры."""

import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")
TIMEOUT = 10  # секунд; без таймаута зависший API подвешивает всю страницу

st.set_page_config(page_title="Kwork ML — Предсказание цен", page_icon="🏠", layout="centered")
st.title("🏠 Предсказание цен на квартиры в Москве")


def error_message(resp: requests.Response) -> str:
    """Достаёт понятный текст ошибки из ответа API.

    detail бывает строкой (наши ошибки), списком (422 валидации Pydantic) или
    вообще отсутствовать (например, 502 от прокси с HTML вместо JSON).
    """
    try:
        payload = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, list):
        return "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
    return str(detail) if detail else f"HTTP {resp.status_code}"


def logout() -> None:
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.user_id = None


# --- Auth ---
if "authenticated" not in st.session_state:
    logout()

if not st.session_state.authenticated:
    st.header("Вход")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Пароль", type="password", key="login_password")

    if st.button("Войти"):
        try:
            resp = requests.post(
                f"{API_BASE}/auth/login",
                json={"email": email, "password": password},
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.token = data["access_token"]
                st.session_state.user_id = data["user_id"]
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error(f"Ошибка: {error_message(resp)}")
        except requests.RequestException as e:
            st.error(f"Не удалось подключиться к API: {e}")

    st.divider()
    st.header("Регистрация")
    reg_email = st.text_input("Email", key="reg_email")
    reg_password = st.text_input("Пароль (минимум 8 символов)", type="password", key="reg_password")

    if st.button("Зарегистрироваться"):
        try:
            resp = requests.post(
                f"{API_BASE}/auth/register",
                json={"email": reg_email, "password": reg_password},
                timeout=TIMEOUT,
            )
            if resp.status_code == 201:
                st.success("Регистрация успешна! Теперь войдите.")
            else:
                st.error(f"Ошибка: {error_message(resp)}")
        except requests.RequestException as e:
            st.error(f"Не удалось подключиться к API: {e}")

else:
    st.sidebar.success(f"Вы вошли (пользователь №{st.session_state.user_id})")
    if st.sidebar.button("Выйти"):
        logout()
        st.rerun()

    headers = {"Authorization": f"Bearer {st.session_state.token}"}

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
        if floor > total_floors:
            st.error("Этаж не может быть выше, чем этажность дома")
        else:
            features = {
                "rooms": rooms,
                "area": area,
                "floor": floor,
                "total_floors": total_floors,
                "year_built": year_built,
                "distance_to_metro_min": metro_min,
            }
            try:
                resp = requests.post(
                    f"{API_BASE}/predict", json=features, headers=headers, timeout=TIMEOUT
                )
                if resp.status_code == 201:
                    data = resp.json()
                    st.success(f"💰 Предсказанная цена: {data['predicted_price']:,.0f} ₽")
                    st.info(f"📊 Фичи: {data['input_features']}")
                elif resp.status_code == 401:
                    # Токен истёк — возвращаем на форму входа
                    logout()
                    st.rerun()
                else:
                    st.error(f"Ошибка: {error_message(resp)}")
            except requests.RequestException as e:
                st.error(f"Не удалось подключиться к API: {e}")

    st.divider()

    st.header("История предсказаний")
    try:
        resp = requests.get(f"{API_BASE}/predict/history", headers=headers, timeout=TIMEOUT)
        if resp.status_code == 200:
            predictions = resp.json()["predictions"]
            if predictions:
                for pred in predictions:
                    st.write(f"- **{pred['predicted_price']:,.0f} ₽** — {pred['input_features']}")
            else:
                st.info("История пуста")
        elif resp.status_code == 401:
            logout()
            st.rerun()
        else:
            st.error(f"Ошибка загрузки истории: {error_message(resp)}")
    except requests.RequestException as e:
        st.error(f"Не удалось загрузить историю: {e}")
