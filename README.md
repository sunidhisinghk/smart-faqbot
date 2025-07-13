''' Updated it from OpenAI API to Gemeni API due to cost issue

# 💼 Smart FAQBot

An AI-powered HR assistant chatbot built with **Streamlit**, **FAISS**, and **Google Gemini**, designed to answer queries based on an organization's HR policy documents.

Demo:
> Run locally with:
bash
streamlit run web_chatbot.py

Project structure:
smart-faqbot/
├── data/
│   └── hr_policy.txt         # Raw HR policy document
├── .env                      # Environment variables (API keys)
├── .gitignore
├── requirements.txt
├── embedder.py              # Preprocessing script (indexing)
├── chatbot.py               # Response generator logic
├── web_chatbot.py           # Streamlit UI

Create a virtual enviornment:
pip install -r requirements.txt

Set up .env:
GEMINI_API_KEY=your-api-key-here

Run embedder.py to generate:
faq_chunks.pkl
faq_index.faiss

Run the application:
streamlit run web_chatbot.py

