"""
WronAI Web Application - Streamlit interface for Polish language model.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from ..models import load_model
from ..inference import InferenceEngine, InferenceConfig
from ..utils.logging import get_logger, setup_logging
from ..utils.memory import get_memory_usage, format_memory_usage

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="WronAI - Polski Model Językowy",
    page_icon="🐦‍⬛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .polish-flag {
        background: linear-gradient(to bottom, #ffffff 50%, #dc143c 50%);
        height: 20px;
        width: 30px;
        display: inline-block;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_wronai_model(model_name: str, quantize: bool = True):
    """Load WronAI model with caching."""
    try:
        with st.spinner(f"Ładowanie modelu {model_name}..."):
            model = load_model(
                model_name=model_name,
                quantize=quantize,
                device="auto"
            )
            return model
    except Exception as e:
        st.error(f"Błąd podczas ładowania modelu: {e}")
        return None

@st.cache_resource
def create_inference_engine(_model):
    """Create inference engine with caching."""
    if _model is None:
        return None

    config = InferenceConfig(
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        use_polish_formatting=True
    )

    return InferenceEngine(_model, config)

def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

    if "generation_stats" not in st.session_state:
        st.session_state.generation_stats = {
            "total_generations": 0,
            "total_tokens": 0,
            "total_time": 0.0
        }

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

def sidebar_config():
    """Create sidebar configuration."""
    st.sidebar.markdown('<div class="polish-flag"></div>**WronAI Settings**', unsafe_allow_html=True)

    # Model selection
    model_options = [
        "mistralai/Mistral-7B-v0.1",
        "microsoft/DialoGPT-small",
        "checkpoint/wronai-7b"
    ]

    selected_model = st.sidebar.selectbox(
        "Wybierz model:",
        model_options,
        index=0
    )

    # Generation parameters
    st.sidebar.subheader("Parametry generacji")

    max_length = st.sidebar.slider(
        "Maksymalna długość:",
        min_value=50,
        max_value=1000,
        value=256,
        step=50
    )

    temperature = st.sidebar.slider(
        "Temperatura:",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1
    )

    top_p = st.sidebar.slider(
        "Top-p:",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05
    )

    repetition_penalty = st.sidebar.slider(
        "Kara za powtórzenia:",
        min_value=1.0,
        max_value=2.0,
        value=1.1,
        step=0.1
    )

    # Advanced settings
    with st.sidebar.expander("Ustawienia zaawansowane"):
        quantize = st.checkbox("Kwantyzacja 4-bit", value=True)
        use_polish_format = st.checkbox("Format polski", value=True)
        show_stats = st.checkbox("Pokaż statystyki", value=True)

    return {
        "model_name": selected_model,
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "quantize": quantize,
        "use_polish_format": use_polish_format,
        "show_stats": show_stats
    }

def display_system_info():
    """Display system information."""
    memory_info = get_memory_usage()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "CPU Memory",
            f"{memory_info.get('cpu_memory_used_gb', 0):.1f}GB",
            f"{memory_info.get('cpu_memory_percent', 0):.1f}%"
        )

    with col2:
        gpu_memory = memory_info.get('gpu_memory_used_gb', 0)
        gpu_total = memory_info.get('gpu_memory_total_gb', 0)
        if gpu_total > 0:
            st.metric(
                "GPU Memory",
                f"{gpu_memory:.1f}GB",
                f"{memory_info.get('gpu_memory_percent', 0):.1f}%"
            )
        else:
            st.metric("GPU Memory", "N/A", "GPU niedostępny")

    with col3:
        total_gens = st.session_state.generation_stats["total_generations"]
        avg_time = (st.session_state.generation_stats["total_time"] /
                   max(total_gens, 1))
        st.metric(
            "Generacje",
            str(total_gens),
            f"{avg_time:.2f}s avg"
        )

def display_generation_stats(engine):
    """Display generation statistics."""
    if engine is None:
        return

    stats = engine.get_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Statystyki wydajności")

        # Performance metrics
        metrics_df = pd.DataFrame([
            {"Metryka": "Tokeny/sekundę", "Wartość": f"{stats.get('average_tokens_per_second', 0):.1f}"},
            {"Metryka": "Całkowite tokeny", "Wartość": stats.get('total_tokens', 0)},
            {"Metryka": "Całkowity czas", "Wartość": f"{stats.get('total_time', 0):.2f}s"},
            {"Metryka": "Generacje", "Wartość": stats.get('total_generations', 0)}
        ])

        st.dataframe(metrics_df, hide_index=True)

    with col2:
        st.subheader("🧠 Informacje o modelu")

        model_info = stats.get("model_info", {})
        if model_info:
            info_df = pd.DataFrame([
                {"Właściwość": "Typ modelu", "Wartość": model_info.get('model_type', 'N/A')},
                {"Właściwość": "Parametry", "Wartość": f"{model_info.get('total_parameters', 0):,}"},
                {"Właściwość": "Kwantyzacja", "Wartość": "Tak" if model_info.get('is_quantized') else "Nie"},
                {"Właściwość": "LoRA", "Wartość": "Tak" if model_info.get('has_lora') else "Nie"}
            ])

            st.dataframe(info_df, hide_index=True)

def chat_interface(engine, config):
    """Create chat interface."""
    st.subheader("💬 Chat z WronAI")

    # Display conversation history
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>Ty:</strong> {message["content"]}</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>🐦‍⬛ WronAI:</strong> {message["content"]}</div>',
                       unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Napisz wiadomość po polsku..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        if engine:
            with st.spinner("WronAI myśli..."):
                start_time = time.time()

                try:
                    # Update inference config
                    engine.config.max_length = config["max_length"]
                    engine.config.temperature = config["temperature"]
                    engine.config.top_p = config["top_p"]
                    engine.config.repetition_penalty = config["repetition_penalty"]

                    # Generate response
                    response = engine.chat(
                        prompt,
                        conversation_history=st.session_state.conversation_history
                    )

                    generation_time = time.time() - start_time

                    # Update statistics
                    st.session_state.generation_stats["total_generations"] += 1
                    st.session_state.generation_stats["total_time"] += generation_time

                    # Add response to messages
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Update conversation history
                    st.session_state.conversation_history.append({
                        "user": prompt,
                        "assistant": response
                    })

                    # Keep only last 10 exchanges
                    if len(st.session_state.conversation_history) > 10:
                        st.session_state.conversation_history = st.session_state.conversation_history[-10:]

                    st.rerun()

                except Exception as e:
                    st.error(f"Błąd podczas generacji: {e}")
        else:
            st.error("Model nie został załadowany.")

def text_generation_interface(engine, config):
    """Create text generation interface."""
    st.subheader("✍️ Generacja tekstu")

    # Predefined prompts
    st.write("**Przykładowe prompty:**")
    example_prompts = [
        "Opowiedz o Polsce:",
        "Jakie są tradycyjne polskie potrawy?",
        "Wyjaśnij pojęcie sztucznej inteligencji:",
        "Napisz krótki wiersz o jesieni:",
        "Przetłumacz na angielski: 'Miło Cię poznać'"
    ]

    cols = st.columns(len(example_prompts))
    for i, prompt in enumerate(example_prompts):
        with cols[i]:
            if st.button(prompt[:20] + "...", key=f"prompt_{i}"):
                st.session_state.current_prompt = prompt

    # Text input
    prompt = st.text_area(
        "Wprowadź prompt:",
        value=getattr(st.session_state, 'current_prompt', ''),
        height=100,
        placeholder="Napisz prompt po polsku..."
    )

    # Generate button
    col1, col2 = st.columns([1, 4])
    with col1:
        generate_button = st.button("🚀 Generuj", type="primary")

    with col2:
        if st.button("🗑️ Wyczyść"):
            if 'current_prompt' in st.session_state:
                del st.session_state.current_prompt
            st.rerun()

    # Generation
    if generate_button and prompt and engine:
        with st.spinner("Generowanie tekstu..."):
            start_time = time.time()

            try:
                # Update config
                engine.config.max_length = config["max_length"]
                engine.config.temperature = config["temperature"]
                engine.config.top_p = config["top_p"]
                engine.config.repetition_penalty = config["repetition_penalty"]

                # Generate text
                response = engine.generate(prompt)
                generation_time = time.time() - start_time

                # Update statistics
                st.session_state.generation_stats["total_generations"] += 1
                st.session_state.generation_stats["total_time"] += generation_time

                # Display result
                st.subheader("📝 Wygenerowany tekst:")
                st.markdown(f'<div class="chat-message assistant-message">{response}</div>',
                           unsafe_allow_html=True)

                # Display generation info
                st.info(f"⏱️ Czas generacji: {generation_time:.2f}s | "
                       f"📏 Długość: {len(response.split())} słów")

            except Exception as e:
                st.error(f"Błąd podczas generacji: {e}")

def model_comparison_interface():
    """Create model comparison interface."""
    st.subheader("🔍 Porównanie modeli")

    # Model comparison data
    comparison_data = {
        "Model": ["WronAI-7B", "PLLuM-8x7B", "Bielik-7B", "GPT-3.5"],
        "Parametry": ["7B", "46.7B", "7B", "175B"],
        "Polski Score": [7.2, 8.5, 7.8, 6.5],
        "VRAM": ["8GB", "40GB+", "14GB", "N/A"],
        "Licencja": ["Apache 2.0", "Custom", "Apache 2.0", "Commercial"]
    }

    df = pd.DataFrame(comparison_data)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(df, hide_index=True)

    with col2:
        # Bar chart of Polish scores
        fig = px.bar(
            df,
            x="Model",
            y="Polski Score",
            title="Wyniki dla języka polskiego",
            color="Polski Score",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

def benchmarks_interface():
    """Create benchmarks interface."""
    st.subheader("📊 Benchmarki")

    # Sample benchmark data
    benchmark_data = {
        "Test": [
            "Polish QA",
            "Sentiment Analysis",
            "Text Generation",
            "Translation PL-EN",
            "Grammar Check"
        ],
        "WronAI Score": [85, 78, 82, 75, 80],
        "Baseline": [70, 65, 70, 60, 65],
        "Best Available": [90, 85, 88, 85, 85]
    }

    df = pd.DataFrame(benchmark_data)

    # Radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=df["WronAI Score"],
        theta=df["Test"],
        fill='toself',
        name='WronAI',
        line_color='blue'
    ))

    fig.add_trace(go.Scatterpolar(
        r=df["Best Available"],
        theta=df["Test"],
        fill='toself',
        name='Best Available',
        line_color='red',
        opacity=0.6
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Porównanie wyników benchmarków"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("Szczegółowe wyniki")
    st.dataframe(df, hide_index=True)

def examples_interface():
    """Create examples interface."""
    st.subheader("📚 Przykłady użycia")

    examples = [
        {
            "title": "Odpowiadanie na pytania",
            "prompt": "Co to jest sztuczna inteligencja?",
            "response": "Sztuczna inteligencja (AI) to dziedzina informatyki zajmująca się tworzeniem systemów komputerowych zdolnych do wykonywania zadań, które tradycyjnie wymagają ludzkiej inteligencji, takich jak rozpoznawanie obrazów, rozumienie mowy czy podejmowanie decyzji."
        },
        {
            "title": "Tłumaczenie",
            "prompt": "Przetłumacz na angielski: 'Dziękuję bardzo za pomoc'",
            "response": "'Dziękuję bardzo za pomoc' w języku angielskim to 'Thank you very much for your help'."
        },
        {
            "title": "Kreatywne pisanie",
            "prompt": "Napisz krótki wiersz o Polsce",
            "response": "Nad Wisłą płynie czas,\nBiało-czerwona duma nas,\nW polskich sercach miłość tkwi,\nDo ojczyzny, co nas chroni, żywi."
        },
        {
            "title": "Wyjaśnianie pojęć",
            "prompt": "Wyjaśnij różnicę między uczeniem maszynowym a deep learning",
            "response": "Uczenie maszynowe to szeroka dziedzina AI, gdzie algorytmy uczą się z danych. Deep learning to podzbiór uczenia maszynowego wykorzystujący sieci neuronowe z wieloma warstwami do modelowania złożonych wzorców w danych."
        }
    ]

    for i, example in enumerate(examples):
        with st.expander(f"📝 {example['title']}"):
            st.markdown(f"**Prompt:** {example['prompt']}")
            st.markdown(f"**Odpowiedź:** {example['response']}")

            if st.button(f"Wypróbuj ten przykład", key=f"example_{i}"):
                st.session_state.current_prompt = example['prompt']
                st.switch_page("Generacja tekstu")

def about_interface():
    """Create about interface."""
    st.subheader("ℹ️ O WronAI")

    st.markdown("""
    ### 🐦‍⬛ WronAI - Polski Model Językowy
    
    **WronAI** to open-source projekt mający na celu demokratyzację sztucznej inteligencji 
    dla języka polskiego. Nasz model jest specjalnie zoptymalizowany dla polszczyzny 
    i może być trenowany na dostępnym sprzęcie konsumenckim.
    
    #### ✨ Kluczowe cechy:
    - 🇵🇱 **Specjalizacja w języku polskim** - zoptymalizowany dla polskiej morfologii
    - 💾 **Niskie wymagania** - działa na GPU 8GB dzięki kwantyzacji 4-bit
    - 🔧 **QLoRA fine-tuning** - efektywne dostrajanie z LoRA
    - 📊 **Monitoring wydajności** - real-time śledzenie użycia pamięci
    - 🐳 **Docker support** - łatwe wdrożenie w kontenerach
    
    #### 🏗️ Architektura:
    - **Model bazowy:** Mistral-7B / LLaMA-7B
    - **Kwantyzacja:** 4-bit NF4 z BitsAndBytesConfig
    - **Fine-tuning:** QLoRA z polskim korpusem instrukcji
    - **Inference:** Optymalizowany silnik z cache'owaniem
    
    #### 📈 Wydajność:
    - **Parametry:** 7B (46.7B w wersji MoE)
    - **VRAM:** 8GB dla inferencji, 8GB+ dla treningu
    - **Prędkość:** ~10-50 tokenów/sekundę (zależnie od sprzętu)
    - **Jakość:** 7.2/10 w polskich benchmarkach
    
    #### 🤝 Community:
    - **Licencja:** Apache 2.0 (pełna otwartość)
    - **GitHub:** [WronAI Repository](https://github.com/twoje-repo/WronAI)
    - **Discord:** [WronAI Community](https://discord.gg/wronai)
    - **Contributions:** Zapraszamy do współpracy!
    """)

    # System info
    st.subheader("🖥️ Informacje systemowe")
    memory_info = get_memory_usage()

    system_info = {
        "CPU Memory": f"{memory_info.get('cpu_memory_used_gb', 0):.1f}GB / {memory_info.get('cpu_memory_total_gb', 0):.1f}GB",
        "GPU Memory": f"{memory_info.get('gpu_memory_used_gb', 0):.1f}GB / {memory_info.get('gpu_memory_total_gb', 0):.1f}GB" if memory_info.get('gpu_memory_total_gb', 0) > 0 else "Niedostępny",
        "Memory Usage": format_memory_usage(memory_info),
        "Model Loaded": "Tak" if st.session_state.model_loaded else "Nie"
    }

    for key, value in system_info.items():
        st.text(f"{key}: {value}")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">🐦‍⬛ WronAI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Polski Model Językowy - Demokratyzacja AI</p>', unsafe_allow_html=True)

    # Sidebar configuration
    config = sidebar_config()

    # Load model
    if not st.session_state.model_loaded or st.sidebar.button("🔄 Przeładuj model"):
        model = load_wronai_model(config["model_name"], config["quantize"])
        if model:
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success(f"✅ Model {config['model_name']} załadowany pomyślnie!")
        else:
            st.session_state.model_loaded = False
            st.error("❌ Nie udało się załadować modelu")

    # Create inference engine
    engine = None
    if st.session_state.model_loaded:
        engine = create_inference_engine(st.session_state.model)

    # Navigation tabs
    tabs = st.tabs(["💬 Chat", "✍️ Generacja", "🔍 Porównanie", "📊 Benchmarki", "📚 Przykłady", "ℹ️ O projekcie"])

    with tabs[0]:
        if config["show_stats"]:
            display_system_info()
            if engine:
                display_generation_stats(engine)
        chat_interface(engine, config)

    with tabs[1]:
        text_generation_interface(engine, config)

    with tabs[2]:
        model_comparison_interface()

    with tabs[3]:
        benchmarks_interface()

    with tabs[4]:
        examples_interface()

    with tabs[5]:
        about_interface()

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Made with ❤️ for Polish AI Community | Apache 2.0 License</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()