
import os 
import streamlit as st
from transformer_question_answering.pipeline.prediction import PredictionPipeline


def main():
    st.set_page_config(layout="wide", page_title="Transformer model for Question Answering using Natural Language Processing")
    st.header("Question Answering using Natural Language Processing")

    input_text = st.text_area("Enter the text: ", key="input_text", height=200)

    input_question = st.text_input("Enter the question: ", key="input_question")

    # prediction button
    submit = st.button("Ask the question")

    if submit and input_text and input_question:
        response = PredictionPipeline().predict(text=input_text, question=input_question)
        st.subheader("The Response is")
        st.write(response)



if __name__ == "__main__":
    main()