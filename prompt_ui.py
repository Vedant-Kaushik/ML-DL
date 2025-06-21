import streamlit as st
from chatmodel import client,system_prompt

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]
messages = st.session_state.messages

st.title("ChatGPT-4o")

user_input = st.text_area(
    """Go ahead and ask me anything!
You can also ask follow up questions to the previous question"""
)

messages.append({"role": "user", "content": user_input})
if st.button("Generate"): 
    with st.spinner("Generating response..."):
        result = client.chat.completions.create(
            model='provider-5/gpt-4o',
            messages=messages,
            temperature=0.7
        )
        messages.append({"role": "assistant", "content": result.choices[0].message.content})
    st.write(result.choices[0].message.content) 