# LLM Sandbox

This repo is for experimenting with llm and related infrastructure. Goal is to learn as much about LLM chatbots, RAG, Langchain, NeMo, Triton and other tools.

Progress

- [x] Base chat interface
- [x] OpenAI API Integration
- [x] Safe-guarding for unrelated questions
- [ ] Document and Code repository upload
- [ ] Local model serve using Triton
- [ ] Finetune on custom dataset
- [x] Realtime natural language currency converter


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/deval-maker/llm-sandbox.git
   ```

2. Navigate to the project directory:

   ```bash
   cd llm-sandbox
   ```

3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:

   - On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to [http://localhost:8501](http://localhost:8501) to view the app.



## License

This project is licensed under the [MIT License](LICENSE).
```
