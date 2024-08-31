
import google.generativeai as ai

API_KEY = "AIzaSyCbp44uFpmTAD1Iws_r1nxeWkmHMBVzOr0"

ai.configure(api_key = API_KEY)

model = ai.GenerativeModel(model_name="gemini-pro")

chat = model.start_chat()

while True:
    message = input('\nYou: ')
    if message.lower()=='exit':
        print('Chatbot: Goodbye!')
        break


    response = chat.send_message(message)
    print("Chatbot: ", response.text)
