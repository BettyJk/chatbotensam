import streamlit as st
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import fitz
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect
from keybert import KeyBERT
from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import logging
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from typing import List, Tuple

from PIL import Image
import base64
import time

st.set_page_config(
    page_title="ENSAM Chatbot",
    page_icon="🤖",
    layout="wide",
)

ensam_logo = r"C:\Users\admin\new bot\ensam.png"
chatbot_logo = r"C:\Users\admin\new bot\logoL7erf.png"

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image(chatbot_logo, width=200)
with col2:
    st.markdown("<h1 style='text-align: center; color: #007bff;'>ENSAM Chatbot</h1>", unsafe_allow_html=True)
with col3:
    st.image(ensam_logo, width=200)

st.markdown(
    """
    <style>
    @keyframes fade-in {
        0% {opacity: 0; transform: translateY(-10px);}
        100% {opacity: 1; transform: translateY(0);}
    }
    .animated-card {
        animation: fade-in 2s ease;
    }
    .stApp {
        background-color: #141f2b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='animated-card'>", unsafe_allow_html=True)
st.write("Welcome to the ENSAM chatbot! Please select a feature to start:")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #2c3e50; 
            color: #ecf0f1;
            border-radius: 15px;
            padding: 20px;
        }

        /* Title Styling */
        .sidebar .sidebar-content .title {
            font-size: 22px;
            color: #f39c12;
            font-weight: bold;
            text-align: center;
        }

        /* Radio Buttons */
        .stRadio>label {
            font-size: 16px;
            color: #ecf0f1;
            font-weight: 500;
            display: block;
            margin-top: 10px;
        }

        .stRadio>div>div>label {
            background: #34495e;
            padding: 10px;
            border-radius: 5px;
            color: #ecf0f1;
        }

        /* Hover Effect for Radio Options */
        .stRadio>div>div>label:hover {
            background-color: #f39c12;
            color: #2c3e50;
            transform: scale(1.05);
            transition: 0.3s;
        }

        /* Sidebar Animation */
        @keyframes slideIn {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(0); }
        }
        
        .sidebar .sidebar-content {
            animation: slideIn 0.5s ease-out;
        }

        /* Add a subtle shadow for a 3D effect */
        .sidebar .sidebar-content {
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar title and radio button selection
st.sidebar.title("L7erf Bot")
choice = st.sidebar.radio(
    "Choose your chatbot:",
    ("Summarization Bot", "General Info Bot", "Courses Bot")
)

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if choice == "Summarization Bot":
   
    def load_summarization_models():
        try:
            summarizer_en = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
            summarizer_fr = pipeline("summarization", model="moussaKam/mbarthez-dialogue-summarization", device=-1)
            return summarizer_en, summarizer_fr
        except Exception as e:
            logger.error(f"Error loading summarization models: {e}")
            return None, None

    summarizer_en, summarizer_fr = load_summarization_models()
    kw_model = KeyBERT()
    flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    def is_valid_pdf(file_path: str) -> bool:
        try:
            with fitz.open(file_path) as pdf:
                return pdf.page_count > 0
        except Exception as e:
            logger.error(f"PDF validation error: {e}")
            return False

    def extract_text_from_pdf(pdf_path: str) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return "".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return ""

    def extract_images_and_ocr(pdf_path: str, max_pages: int = 5) -> str:
        try:
            images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=max_pages)
            return "".join(pytesseract.image_to_string(img, lang="eng+fra") for img in images)
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return ""

    def get_summarizer(text: str):
        language = detect(text)
        return (summarizer_fr, "French") if language == "fr" else (summarizer_en, "English")

    def summarize_text(text: str, length: str = "medium") -> str:
        summarizer, language = get_summarizer(text)
        if summarizer is None:
            return "Error: Summarizer model not available."

        length_map = {"short": (50, 100), "medium": (100, 150), "long": (150, 200)}
        min_len, max_len = length_map.get(length, (100, 150))

        chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
        summaries = [
            summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
            for chunk in chunks
        ]
        return " ".join(summaries)

    def extract_keywords(text: str) -> List[str]:
        language = detect(text)
        stop_words = "french" if language == "fr" else "english"
        return [kw[0] for kw in kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=stop_words, top_n=10)]

    def generate_structured_notes(summary: str, language: str) -> str:
        prompt = (
            f"Organize the following summary into structured notes with sections and bullet points in {language}:\n\n"
            f"{summary}\n\nStructure:\n- **Introduction**\n- **Key Points**\n- **Conclusion**"
        )
        inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = flan_model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1, temperature=0.7)
        return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_content_from_pdf(pdf_file) -> Tuple[str, str]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_path = temp_file.name

        if not is_valid_pdf(temp_path):
            return "", ""

        with ThreadPoolExecutor() as executor:
            future_text = executor.submit(extract_text_from_pdf, temp_path)
            future_ocr = executor.submit(extract_images_and_ocr, temp_path, max_pages=5)

            extracted_text = future_text.result()
            extracted_ocr = future_ocr.result()

        os.remove(temp_path)
        return extracted_text or extracted_ocr, extracted_ocr

    st.title("Multilingual PDF Summarizer & Note Organizer")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        st.info("Processing PDF...")
        text, ocr_text = extract_content_from_pdf(uploaded_file)

        if not text:
            st.error("Failed to extract content from the PDF.")
        else:
            language = "French" if detect(text) == "fr" else "English"
            st.success(f"Content extracted! Detected Language: {language}")

            summary_length = st.radio("Summarization Depth:", ["short", "medium", "long"])
            summary = summarize_text(text, length=summary_length)
            keywords = extract_keywords(text)

            st.subheader("Summary")
            st.write(summary)

            st.subheader("Keywords")
            st.write(", ".join(keywords))

            with st.spinner("Generating structured notes..."):
                notes = generate_structured_notes(summary, language)
            st.subheader("Organized Notes")
            st.write(notes)

            if ocr_text:
                st.subheader("OCR Text")
                st.text_area("Extracted Text from Images", ocr_text)


elif choice == "Courses Bot":
   
    model_name = r"C:\Users\admin\l7erf-bot\model"

    with st.spinner("Chargement des modèles..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
        model.resize_token_embeddings(len(tokenizer))

    def generate_gpt_response(query):
        """
        Generate a response to a given query using the fine-tuned GPT model.
        
        Parameters:
        query (str): The user input question.
        
        Returns:
        str: The generated response from the model.
        """
        prompt = f"Question: {query}\nRéponse:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_length=200,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Réponse:")[-1].strip()

    st.title("Chatbot IA en Français")
  
    query = st.text_input("Posez votre question ici:")
    
    if query:
        with st.spinner("Recherche en cours..."):
            response = generate_gpt_response(query)

        st.write("### Réponse :", response)

elif choice == "General Info Bot":
 
    os.environ["GROQ_API_KEY"] = "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"

    def main():
        """
        Streamlit chatbot interface using LangChain and Groq API.
        """

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY is not set. Please export it in your environment.")
            return

        model = "llama3-8b-8192"
        groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

        system_prompt = ("You are a friendly and knowledgeable chatbot specialized in providing "
    "information about ENSAM Meknès, including its academic programs and extracurricular activities.\n\n"
    "### Academic Tracks (Filières):\n\n"
    "#### A. Parcours Mécanique (Mechanical Track):\n"
    "- **Génie Mécanique Structures et Ingénierie des Produits**:\n"
    "  Focus: Product design, structural analysis, and material properties.\n"
    "  Key Topics: Finite Element Analysis (FEA), stress analysis, fatigue testing, and product lifecycle management.\n"
    "- **Génie Mécanique Procédés de Fabrication Industrielle**:\n"
    "  Focus: Manufacturing processes, CNC machining, and production optimization.\n"
    "  Key Topics: Lean manufacturing, Six Sigma, process automation, quality control, and industrial robotics.\n"
    "- **Génie Mécanique Energétique**:\n"
    "  Focus: Energy systems, thermodynamics, and renewable energy.\n"
    "  Key Topics: HVAC, renewable energy technologies, power plant engineering, and energy management.\n"
    "- **Génie Mécanique Mobilité (Aéronautique et Automobile)**:\n"
    "  Focus: Vehicle and aerospace engineering, propulsion systems, and aerodynamics.\n"
    "  Key Topics: Vehicle design, aerodynamics, propulsion systems, and lightweight materials.\n\n"
    "#### B. Parcours Electromécanique et Industriel (Electromechanical and Industrial Track):\n"
    "- **Génie Industriel et Productique**:\n"
    "  Focus: Optimizing production processes, supply chains, and operations.\n"
    "  Key Topics: Operations research, supply chain management, logistics, and inventory management.\n"
    "- **Génie Electromécanique Energie et Maintenance Electromécanique**:\n"
    "  Focus: Energy management, system maintenance, and reliability.\n"
    "  Key Topics: Predictive maintenance, energy systems, mechatronics, and fault diagnosis.\n"
    "- **Génie Electromécanique et Digitalisation Industrielle**:\n"
    "  Focus: Combining electromechanics with IoT and Industry 4.0 technologies.\n"
    "  Key Topics: IoT, automation, PLC programming, and digital twins.\n"
    "- **Génie Industriel et Excellence Opérationnelle**:\n"
    "  Focus: Lean manufacturing, continuous improvement, and operational efficiency.\n"
    "  Key Topics: Lean Six Sigma, TQM, process reengineering, and Kaizen methodologies.\n\n"
    "#### C. Parcours Informatique et IA (Computer Science and AI Track):\n"
    "- **Génie Informatique**:\n"
    "  Focus: Software development, system design, and IT infrastructure.\n"
    "  Key Topics: Programming, database management, network security, and software engineering.\n"
    "- **Génie Intelligence Artificielle et Data Science**:\n"
    "  Focus: AI, machine learning, and data analytics.\n"
    "  Key Topics: Machine learning algorithms, neural networks, big data, and natural language processing.\n\n"
    "#### D. Parcours Civil (Civil Track):\n"
    "- **Génie Civil**:\n"
    "  Focus: Structural analysis, construction management, and urban planning.\n"
    "  Key Topics: Structural design, geotechnical engineering, construction materials, and environmental engineering.\n\n"
    "### Extracurricular Activities (Nos Clubs):\n"
    "ENSAM Meknès hosts a wide variety of student clubs, each with its unique focus and activities:\n"
    "- **CLUB SOCIAL A&M**: Engages in volunteering and social responsibility projects to assist underprivileged communities.\n"
    "- **Caravane Alhayat**: Focuses on organizing charitable and humanitarian events, including community support initiatives and outreach activities.\n"
    "- **Club Culturel A&M**: Celebrates cultural diversity through events, workshops, and discussions.\n"
    "- **Club ENSAM Express**: Focused on debating, creative writing, and public speaking to hone communication skills.\n"
    "- **Club ENSAM Events**: Organizes large-scale events such as marathons, comedy shows, and hiking trips.\n"
    "- **Club WeArt**: A hub for artistic expression, including drawing, fashion design, and theater performances.\n"
    "- **Club K-Otaku**: Celebrates Japanese pop culture, including anime, manga, and cosplay.\n"
    "- **Club Musique A&M**: Brings music enthusiasts together for singing, performing, and organizing parties.\n"
    "- **Club Enactus A&M**: Focuses on social entrepreneurship, empowering students to create projects with a positive impact.\n"
    "- **Club GadzIt A&M**: A technology-focused club specializing in coding, programming, and software development.\n"
    "- **Space Club ENSAM Meknès**: Explores the wonders of space science and cosmology.\n"
    "- **Club Robotique et Innovation A&M**: Dives into robotics, automation, and innovative engineering projects.\n"
    "- **Club Energétique**: Dedicated to energy systems and sustainable energy solutions.\n"
    "- **Club Mécanique A&M**: Explores the mechanics of machines, structures, and their applications.\n"
    "- **Club Industriel A&M**: Organizes field trips, guided visits to companies, and career development events like CV correction workshops, interview simulations, and the famous 'Industrial Day' celebrating ENSAM alumni entrepreneurs.\n"
    "- **Club Mat'ion Process**: Focused on manufacturing processes like forging, foundry work, and material science.\n"
    "- **Club Civil A&M**: Engages in activities related to civil engineering, including site visits and technical workshops.\n"
    "- **Ultras Gadz'Arts**: A passionate supporters' club dedicated to cheering for ENSAM's sports teams and promoting team spirit.\n\n"
    "Feel free to ask about academic programs, clubs, or anything else related to ENSAM Meknès!")
        conversational_memory_length = 5 
        memory = ConversationBufferWindowMemory(
            k=conversational_memory_length, memory_key="chat_history", return_messages=True
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        with st.sidebar:
            st.header("Session Management")
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")

        st.title("Groq Chatbot")
        user_input = st.text_input("Ask me anything:", placeholder="Type your message here...")

        if user_input:
  
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}"),
                ]
            )

            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                memory=memory,
            )

            try:
                
                response = conversation.predict(human_input=user_input)
              
                st.session_state.chat_history.append({"user": user_input, "bot": response})
                st.success(response)
            except Exception as e:
                st.error(f"Error: {e}")

       
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for message in st.session_state.chat_history:
                st.markdown(f"**You**: {message['user']}")
                st.markdown(f"**Bot**: {message['bot']}")

    if __name__ == "__main__":
        main()
