import streamlit as st
chosen_option = st.sidebar.selectbox('Hello', ['A', 'B', 'C'])
st.write(chosen_option)