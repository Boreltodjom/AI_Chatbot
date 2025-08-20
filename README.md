# AI Chatbot with Image Analysis

A powerful Flask-based AI chatbot that supports conversation memory, image analysis, and multilingual interactions.

## Features

- 🧠 **Conversation Memory** - Remembers previous questions and context
- 🖼️ **Image Analysis** - Advanced image recognition and description
- 🌐 **Multilingual Support** - English and French language support
- 💬 **Real-time Chat** - Modern, responsive chat interface
- 🔄 **Session Management** - Each user gets their own conversation history
- 🗑️ **Clear History** - Option to reset conversation
- 📱 **Mobile Responsive** - Works perfectly on all devices

## Technologies Used

- **Backend**: Flask (Python)
- **AI Models**: OpenRouter API (Mistral-7B, Claude-3-Haiku)
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: PIL (Python Imaging Library)
- **Session Management**: Flask Sessions

## Prerequisites

- Python 3.8 or higher
- OpenRouter API account and key
- Internet connection for AI model access

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-chatbot.git
   cd ai-chatbot
   ```

2. **Install required packages**
   ```bash
   pip install flask requests pillow python-dotenv langdetect transformers
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

4. **Create templates directory**
   ```bash
   mkdir templates
   ```
   Save the HTML template as `templates/index.html`

## Usage

1. **Start the application**
   ```bash
   python advanced_app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Start chatting!**
   - Type messages for text conversations
   - Upload images for visual analysis
   - Use the 🗑️ button to clear conversation history

## API Configuration

This chatbot uses OpenRouter API which provides access to multiple AI models:

- **Text Model**: Mistral-7B-Instruct (fast, efficient for conversations)
- **Vision Model**: Claude-3-Haiku (excellent for image analysis)

To get an API key:
1. Visit [openrouter.ai](https://openrouter.ai)
2. Sign up for an account
3. Add credits to your account
4. Generate an API key
5. Add it to your `.env` file

## Project Structure

```
ai-chatbot/
├── advanced_app.py          # Main Flask application
├── templates/
│   └── index.html           # Chat interface template
├── uploads/                 # Temporary image storage (auto-created)
├── .env                     # Environment variables (create this)
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## Features in Detail

### Conversation Memory
- Maintains context across multiple messages
- Remembers previous topics and references
- Smart context management to avoid token limits

### Image Analysis
- Supports multiple image formats (JPG, PNG, GIF, etc.)
- Automatic image resizing for optimal processing
- Detailed image descriptions and analysis
- Fallback error handling for problematic images

### User Interface
- Modern, clean chat design
- Real-time message updates
- Mobile-responsive layout
- Visual feedback for user actions
- Typing indicators and loading states

## Planned Features (Coming Soon)

- 🎙️ Voice message support
- 📄 Document analysis (PDF, Word)
- 👤 User authentication
- 💾 Persistent conversation storage
- 🎨 Custom themes and personalities
- 🔌 Integration with external APIs

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/ai-chatbot/issues) page
2. Create a new issue if your problem isn't already reported
3. Provide detailed information about the error and your setup

## Acknowledgments

- OpenRouter for providing access to multiple AI models
- Anthropic (Claude) and Mistral AI for their excellent language models
- Flask community for the robust web framework

---

Made with ❤️ by  Todjom Borel 