import streamlit as st


def apply_global_styles() -> None:
    """Apply shared styles used by the UI screens."""
    st.markdown(
        """
        <style>
            .main > div {
                padding-top: 0;
            }
            .hero-title {
                text-align: center;
                font-size: 2.5rem;
                font-weight: 700;
                margin-top: 2rem;
                margin-bottom: 2rem;
            }
            .center-shell {
                min-height: 70vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .block-container {
                max-width: 1000px;
            }
            div.stButton > button {
                width: 100%;
                min-height: 4.5rem;
                font-size: 1.2rem;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
