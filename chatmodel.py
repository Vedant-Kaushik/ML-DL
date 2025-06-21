from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from custom_chat_ctransformers import CustomChatCTransformers


llm = CustomChatCTransformers(
    model_path='models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    model_type='llama',
    temperature=0.5,
    gpu_layers=50 
)
model = llm


messages = [
    SystemMessage(content="You are a helpful AI assistant. When given a review, summarize it and provide the sentiment (positive, negative, or neutral).")
]
while True:
    user_input = input("Enter your review: ")
    if user_input.lower() == "exit":
        break
    messages.append(HumanMessage(content=user_input))

    
    result = model.invoke(messages)
   

    print("Review:", result.content)

    messages.append(AIMessage(content=result.content))



 