from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt

import streamlit as st

load_dotenv()

# Initialize the ChatGoogleGenerativeAI model (replace 'your-api-key' with your actual API key if needed)
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template=load_prompt('template.json')

prompt=template.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
}
)


st.header('Research tool')

if st.button("Submit"):
    chain=template|model
    result=chain.invoke({
            'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)
    st.text('Some random text')