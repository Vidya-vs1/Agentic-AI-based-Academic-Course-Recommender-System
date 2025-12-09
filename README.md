# SmartAdmissions - AI Course Recommender

An intelligent university and course recommendation system powered by AI agents. This application helps students find the best academic programs based on their profile, career goals, budget, and preferences.

## Features

- 🤖 **AI-Powered Analysis**: Uses specialized AI agents for comprehensive recommendations
- 📊 **Real-time Processing**: See results as each agent completes its analysis
- 🎯 **Personalized Recommendations**: Tailored suggestions based on your profile
- 💰 **Scholarship Finder**: Automatic scholarship and funding opportunity discovery
- ⭐ **Student Reviews**: Authentic reviews from multiple sources
- 💬 **Interactive Q&A**: Ask follow-up questions about recommendations

## Prerequisites

- Python 3.8 or higher
- OpenRouter API key (for AI agents)
- Serper API key (for web search)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd smart_academic_recommender
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (for PDF text extraction)
- **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

### 5. Set Up API Keys

Create a `.env` file in the project root:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

## Getting API Keys

### OpenRouter API Key
1. Visit [OpenRouter](https://openrouter.ai/keys)
2. Sign up for a free account
3. Generate an API key
4. Copy the key to your `.env` file

### Serper API Key
1. Visit [Serper.dev](https://serper.dev/api-key)
2. Sign up and get your API key
3. Copy the key to your `.env` file

## Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Enter API Keys**: Input your OpenRouter and Serper API keys
2. **Describe Yourself**: Share your educational background, goals, and preferences in natural language
3. **Optional LOR**: Upload a Letter of Recommendation PDF for additional context
4. **AI Analysis**: Watch as specialized agents analyze your profile and generate recommendations
5. **Explore Results**: Browse through personalized program recommendations, scholarships, and reviews
6. **Ask Questions**: Use the Q&A feature to get more details about recommendations

## How It Works

The system uses multiple AI agents that work sequentially:

1. **Normalizer Agent**: Cleans and standardizes your profile data
2. **Course Matcher Agent**: Finds universities and programs matching your criteria
3. **Course Specialist Agent**: Ranks and evaluates the top programs
4. **Scholarship Agent**: Discovers relevant funding opportunities
5. **Reviews Agent**: Gathers authentic student feedback

Each agent provides natural language output that's easy to understand.

## Project Structure

```
smart_academic_recommender/
├── app.py                 # Main Streamlit application
├── agents.py              # AI agent definitions and tasks
├── utils.py               # Utility functions for text extraction and validation
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                  # Environment variables (create this)
└── e/                    # Virtual environment directory
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`

2. **API Key Issues**: Verify your API keys are correctly set in the `.env` file

3. **Tesseract Not Found**: Ensure Tesseract OCR is installed and accessible in your PATH

4. **Streamlit Issues**: Try clearing Streamlit cache with `streamlit cache clear`

### Error Messages

- **"LLM import failed"**: Check your OpenRouter API key and internet connection
- **"PDF extraction failed"**: Ensure Tesseract is properly installed
- **"Agent execution failed"**: Verify Serper API key and try again

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.

---

**Note**: This application uses AI models and web search APIs. API usage may incur costs based on your usage patterns. Always review the terms of service for the APIs you use.