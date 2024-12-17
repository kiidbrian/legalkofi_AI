import streamlit as st

if __name__ == "__main__":
    # Document Upload Area
    with st.sidebar:
        st.set_page_config(page_title="LegalKofi")
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button(
            "⚡️ Process",
        )
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
    # Question and Answer Area
    st.header("⚖️ LegalKofi")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
        "🔥 Ask",
    )

    if ask and prompt:
        pass