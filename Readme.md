# Personal Financial Advisor App

This is a chatbot app that does tracks all your expences and provides insights.

Main Features
1. **Expense Tracking and Categorization**
   - **Description:** Allow the user to input daily expenses in natural language and/or PDF format from bank/credit card companies. The chatbot will interpret the input, categorize and store them for future reference. Provide a summary of daily, weekly, or monthly expenses.
   - **Prompt Example:** "I spent $50 on groceries yesterday."
   - **Prompt Example:** Upload monthly statement PDF from chase bank.


2. **Currency Conversion:**
   - **Description:** Convert currencies based on real-time exchange rates. Users can inquire about currency conversions between different currencies.
   - **Prompt Example:** "How much is 1000 INR in US dollars?"


Progress

- [x] Basic chat interface 
- [x] OpenAI API Integration
- [x] Safe-guarding for unrelated questions
- [ ] Realtime currency converter
- [ ] Financial document upload
- [ ] Support for changing models
- [ ] NeMo API Integration
- [ ] Everything with Ray framework
- [ ] Finetune on custom catagory dataset
- [ ] Ray Model Serve
- [ ] Triton Serve


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/deval-maker/finance-chatbot.git
   ```

2. Navigate to the project directory:

   ```bash
   cd finance-chatbot
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
