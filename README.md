# JARVIS AI Assistant

JARVIS is an advanced AI assistant that combines voice and text interfaces, task and reminder management, and intelligent conversation using state-of-the-art LLMs (Google Gemini, Groq Llama3) and vector search for context. It features a modern web UI and can also run in continuous voice mode.

## Features

- **Conversational AI**: Chat with JARVIS using natural language (text or voice).
- **Task Management**: Add, list, and update tasks with priorities and due dates.
- **Reminders**: Set and receive reminders.
- **Contextual Memory**: Vector search for previous conversations and context.
- **Web Interface**: Modern, interactive web UI with voice support.
- **Voice Mode**: Wake-word detection and continuous listening (desktop only).
- **Agentic Reasoning**: Uses LangChain agents for advanced task handling.

## Architecture

JARVIS is built with a modular, agentic architecture that enables flexible, context-aware, and extensible AI assistance:

- **LLM Backbone**: Uses Google Gemini and Groq Llama3 models for natural language understanding and generation.
- **LangChain Agents & Tools**: The assistant leverages [LangChain](https://python.langchain.com/) agents, which can invoke a set of "tools" (functions) for structured operations such as task management, reminders, and context retrieval. This enables the AI to reason and act, not just chat.
- **Vector Database (FAISS)**: All conversations and knowledge snippets are embedded using Sentence Transformers and stored in a FAISS vector database. This allows for fast semantic search and retrieval of relevant context for every user query.
- **Knowledge Base**: The assistant builds a knowledge base from user interactions, storing conversation history, tasks, reminders, and extracted user preferences/topics. This knowledge base is used to personalize responses and provide continuity.
- **Web & Voice Interface**: The backend is a Flask server exposing REST APIs, while the frontend is a modern HTML/JS UI supporting both text and voice input/output.
- **Background Services**: Reminders and scheduled tasks are managed in background threads, ensuring timely alerts and persistent memory.

**High-Level Flow:**

1. **User Input** (text/voice) → 
2. **Frontend** (Web UI) → 
3. **Flask Backend** → 
4. **Agent Decision** (LLM + LangChain tools) → 
5. **Tool Execution** (task/reminder/search/context) → 
6. **LLM Response Generation** (with context from vector DB & knowledge base) → 
7. **Frontend Output** (text/voice) → 
8. **Data Storage** (Excel, Pickle, FAISS)

## Requirements

- Python 3.8+
- Node.js (optional, for frontend development)
- API keys for [Google Gemini](https://aistudio.google.com/apikey) and [Groq](https://console.groq.com/keys)

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/gamkers/Project_Jarvis.git
   cd Project_Jarvis
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set API Keys**

   - Get your Gemini API key here: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
   - Get your Groq API key here: [https://console.groq.com/keys](https://console.groq.com/keys)

   Export your API keys as environment variables:

   ```bash
   export GEMINI_API_KEY='your_gemini_api_key'
   export GROQ_API_KEY='your_groq_api_key'
   ```

   Or set them in your shell profile (`.bashrc`, `.zshrc`, etc).

4. **Run the Backend**

   **Standalone Mode**
   ```bash
   python jarvis_ai.py
   ```
   **API Mode**
   ```bash
   python app.py
   ```

   The Flask server will start at [http://localhost:4000](http://localhost:4000).

5. **Access the Web Interface**

   Open your browser and go to [http://localhost:4000](http://localhost:4000).

## Usage

- **Web UI**: Use the chat box to interact with JARVIS. You can type or use the microphone button for voice input.
- **Voice Mode**: (Desktop only) Run `jarvis_ai.py` directly for continuous wake-word listening.
- **APIs**: The backend exposes REST endpoints for conversation, tasks, reminders, and search.

## Project Structure

```
Jarvis_Project/
├── app.py                # Flask backend server
├── jarvis_ai.py          # Main Jarvis AI logic (agents, tools, vector DB, etc.)
├── requirements.txt      # Python dependencies
├── static/               # Frontend static files (HTML, CSS, JS)
├── jarvis_data/          # Data storage (created at runtime)
└── README.md
```
# Jarvis AI Assistant - System Architecture Document

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Patterns](#architecture-patterns)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [External Dependencies](#external-dependencies)
7. [Security Considerations](#security-considerations)
8. [Scalability & Performance](#scalability--performance)
9. [Deployment Architecture](#deployment-architecture)
10. [API Specifications](#api-specifications)
11. [Error Handling Strategy](#error-handling-strategy)
12. [Future Enhancements](#future-enhancements)

## Executive Summary

The Jarvis AI Assistant is a sophisticated, multi-modal AI system that combines natural language processing, voice interaction, task management, and cybersecurity testing capabilities. Built using a modular architecture with agent-based processing, the system provides both conversational AI capabilities and specialized security testing tools through an ESP32 Marauder device.

### Key Features
- **Voice-activated AI assistant** with wake word detection
- **Agentic AI processing** using LangChain and ReAct framework
- **Vector-based conversation memory** with FAISS indexing
- **Task and reminder management** with persistent storage
- **Cybersecurity testing capabilities** via ESP32 Marauder integration
- **Multi-modal interaction** supporting both voice and text interfaces

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Jarvis AI Assistant                         │
├─────────────────────────────────────────────────────────────────┤
│  User Interface Layer                                          │
│  ┌─────────────────┐  ┌─────────────────┐                     │
│  │  Voice Interface │  │  Text Interface │                     │
│  │  (Speech-to-Text)│  │  (Interactive)  │                     │
│  └─────────────────┘  └─────────────────┘                     │
├─────────────────────────────────────────────────────────────────┤
│  Processing Layer                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Wake Word     │  │  Agent Router   │  │  Response Gen   │ │
│  │   Detection     │  │   (LangChain)    │  │   (Gemini)      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Agent Tools Layer                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │Task Manager │ │Reminder Mgr │ │Vector Search│ │Marauder   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │Excel Storage│ │Vector DB    │ │FAISS Index  │              │
│  │(Tasks/Conv) │ │(Embeddings) │ │(Similarity) │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  External Integration Layer                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │ESP32        │ │Google APIs  │ │Groq API     │              │
│  │Marauder     │ │(Gemini/STT) │ │(LLama3)     │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```
<img width="2844" height="2673" alt="diagram-export-26-7-2025-4_04_38-PM" src="https://github.com/user-attachments/assets/43a18345-9814-4092-8931-1ce4a3fc47b3" />


## Architecture Patterns

### 1. Agent-Based Architecture
The system implements an **Agent-Based Architecture** using LangChain's ReAct (Reasoning and Acting) framework:
- **Central Agent Router**: Determines which tool to use based on user input
- **Specialized Tools**: Each capability (tasks, reminders, WiFi attacks) is encapsulated as a tool
- **Tool Orchestration**: Agents can chain multiple tools together to complete complex requests

### 2. Event-Driven Architecture
- **Wake Word Events**: Trigger state transitions from sleep to active mode
- **Reminder Events**: Background scheduler triggers time-based reminders
- **Voice Events**: Continuous listening loop processes audio input events

### 3. Layered Architecture
- **Presentation Layer**: Voice/Text interfaces
- **Business Logic Layer**: AI processing and agent routing
- **Data Access Layer**: File I/O and vector database operations
- **External Services Layer**: API integrations and hardware interfaces

## Core Components

### 1. JarvisAI (Main Controller)
**Purpose**: Central orchestrator and main entry point
**Responsibilities**:
- Initialize all subsystems and dependencies
- Manage conversation state and context
- Route requests between components
- Handle persistent data storage

**Key Methods**:
```python
def process_user_input(user_input: str) -> str
def should_use_agent(user_input: str) -> bool
def generate_response_with_context(user_input: str) -> str
```

### 2. MarauderController (Hardware Interface)
**Purpose**: Interface with ESP32 Marauder for cybersecurity testing
**Responsibilities**:
- Serial communication with ESP32 device
- WiFi network scanning and attacks
- Bluetooth spam/flooding operations
- Voice-guided network selection

**Key Methods**:
```python
def scan_wifi() -> List[str]
def deauth(ssid_index: int)
def stop_attack()
def send_cmd(cmd: str, wait: float)
```

### 3. Agent Tools System
**Purpose**: Modular capabilities accessible to the AI agent
**Components**:
- **Task Management Tools**: Add, list, update tasks
- **Reminder Tools**: Schedule and manage time-based reminders
- **Search Tools**: Query conversation history and context
- **Security Tools**: WiFi/Bluetooth attack capabilities

### 4. Vector Database System
**Purpose**: Semantic search and conversation memory
**Components**:
- **Sentence Transformer**: Converts text to embeddings
- **FAISS Index**: Fast similarity search
- **Persistent Storage**: Pickle-based vector storage

**Architecture**:
```
Text Input → Sentence Transformer → 384D Vector → FAISS Index
                                                      ↓
Query Vector → FAISS Search → Top-K Results → Context Enhancement
```

### 5. Voice Processing Pipeline
**Purpose**: Handle speech-to-text and text-to-speech operations
**Components**:
- **Speech Recognition**: Google Speech-to-Text via SpeechRecognition library
- **Text-to-Speech**: Platform-specific TTS with JARVIS-like voice configuration
- **Wake Word Detection**: Continuous listening with keyword detection

## Data Flow

### Primary Interaction Flow
```
1. User Input (Voice/Text)
   ↓
2. Wake Word Detection (if in sleep mode)
   ↓
3. Speech-to-Text Conversion (if voice)
   ↓
4. Agent Decision Router
   ↓
5a. Agent Tool Execution    OR    5b. Context-Aware Response Generation
    ↓                              ↓
6a. Tool Result Processing         6b. Gemini API Call with Context
   ↓                              ↓
7. Response Synthesis
   ↓
8. Text-to-Speech Conversion
   ↓
9. Audio Output to User
   ↓
10. Conversation Storage & Vector Indexing
```

### Data Persistence Flow
```
User Interaction
   ↓
Conversation Object Creation
   ↓
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│Excel File Save  │  │Vector DB Update │  │FAISS Index Add │
│(Structured Data)│  │(Embeddings)     │  │(Search Index)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## External Dependencies

### AI/ML Services
- **Google Gemini API**: Primary language model for response generation
- **Groq API**: LLama3-70B model for agent processing
- **Sentence Transformers**: Local embedding generation (all-MiniLM-L6-v2)

### Hardware Interfaces
- **ESP32 Marauder**: Cybersecurity testing device via serial communication
- **Microphone/Speakers**: Audio I/O for voice interaction

### Data Storage
- **Excel Files**: Structured storage for tasks, reminders, conversations
- **Pickle Files**: Vector database and FAISS index persistence
- **Local File System**: All data stored locally for privacy

### Python Libraries
```
Core AI: google-generativeai, langchain, langchain-groq
ML/Vector: sentence-transformers, faiss-cpu, numpy
Voice: speech-recognition, pyttsx3, pyaudio
Data: pandas, openpyxl
Hardware: pyserial
Utilities: schedule, threading, pathlib
```

## Security Considerations

### Data Privacy
- **Local Storage**: All conversation data stored locally, no cloud sync
- **API Security**: API keys stored in code (should be moved to environment variables)
- **Voice Data**: Speech-to-text processed via Google APIs (privacy consideration)

### Cybersecurity Testing
- **Controlled Environment**: Marauder operations should only be used in authorized environments
- **Attack Logging**: All security testing operations are logged
- **Manual Authorization**: User must explicitly select targets for attacks

### Recommendations
1. **Environment Variables**: Move API keys to environment variables
2. **Encryption**: Encrypt stored conversation data
3. **Access Control**: Add user authentication for sensitive operations
4. **Audit Logging**: Enhanced logging for all security operations

## Scalability & Performance

### Current Limitations
- **Single User**: Designed for single-user operation
- **Local Processing**: All vector operations performed locally
- **Memory Usage**: FAISS index and vectors stored in memory
- **Serial Communication**: Single ESP32 device connection

### Performance Optimizations
- **Lazy Loading**: Marauder controller initialized on first use
- **Conversation Pruning**: Conversation chain limited to last 5 exchanges
- **Vector Batching**: Efficient FAISS batch operations
- **Background Processing**: Reminder scheduler runs in separate thread

### Scalability Improvements
1. **Database Migration**: Replace Excel with SQLite/PostgreSQL
2. **Vector Database**: Use dedicated vector DB (Pinecone, Weaviate)
3. **Microservices**: Split components into separate services
4. **Load Balancing**: Support multiple Marauder devices

## Deployment Architecture

### Development Environment
```
Local Machine
├── Python 3.8+ Runtime
├── Required Python Packages
├── ESP32 Marauder Device (USB connected)
├── Audio I/O Devices
├── Data Directory (jarvis_data/)
│   ├── conversations.xlsx
│   ├── tasks.xlsx
│   ├── reminders.xlsx
│   ├── vector_db.pkl
│   └── faiss_index.idx
└── Configuration Files
```

### Production Deployment Options
1. **Standalone Desktop Application**
   - Package with PyInstaller
   - Include all dependencies
   - Hardware driver bundling

2. **Docker Container**
   - Containerized deployment
   - Hardware device mapping
   - Volume mounts for persistence

3. **Raspberry Pi Deployment**
   - Edge computing deployment
   - Local voice processing
   - IoT integration capabilities

## API Specifications

### Agent Tool Interface
Each agent tool follows this interface:
```python
def agent_tool_function(input_string: str) -> str:
    """
    Agent tool function interface
    
    Args:
        input_string: Comma-separated parameters or command
        
    Returns:
        str: Human-readable result message
    """
```

### Marauder Command Interface
```python
# WiFi Operations
scan_wifi() -> List[str]
deauth(ssid_index: int) -> None
send_cmd(command: str, wait: float) -> None

# Bluetooth Operations  
bluetooth_flood() -> None
stop_attack() -> None
```

### Voice Interface
```python
# Speech Processing
listen(timeout: int, phrase_time_limit: int) -> Optional[str]
speak(text: str) -> None
detect_wake_word() -> bool
```

## Error Handling Strategy

### Hierarchical Error Handling
1. **Component Level**: Each component handles its own exceptions
2. **Service Level**: Service-specific error recovery (API timeouts, hardware failures)
3. **Application Level**: Global exception handling with graceful degradation

### Error Categories
- **Hardware Errors**: Marauder connection failures, audio device issues
- **API Errors**: Network timeouts, authentication failures, rate limits
- **Data Errors**: File I/O failures, corrupted data, missing files
- **Processing Errors**: Speech recognition failures, agent parsing errors

### Recovery Strategies
- **Retry Logic**: Automatic retry for transient failures
- **Fallback Modes**: Graceful degradation (text mode if voice fails)
- **User Feedback**: Clear error messages and suggested actions
- **Logging**: Comprehensive error logging for debugging

## Future Enhancements

### Short-term Improvements
1. **Enhanced Security**
   - Environment variable configuration
   - Data encryption
   - User authentication

2. **Better Voice Processing**
   - Local STT/TTS options
   - Multi-language support
   - Noise cancellation

3. **Extended Marauder Support**
   - Additional attack types
   - Multiple device support
   - Enhanced logging

### Medium-term Features
1. **Web Interface**
   - Browser-based control panel
   - Remote operation capabilities
   - Mobile-responsive design

2. **Plugin Architecture**
   - Third-party tool integration
   - Custom agent development
   - Marketplace for extensions

3. **Cloud Integration**
   - Optional cloud backup
   - Multi-device synchronization
   - Collaborative features

### Long-term Vision
1. **Multi-Agent System**
   - Specialized agent roles
   - Agent collaboration
   - Complex task orchestration

2. **Advanced Analytics**
   - Usage pattern analysis
   - Performance optimization
   - Predictive capabilities

3. **Enterprise Features**
   - Multi-user support
   - Role-based access control
   - Compliance reporting

## Conclusion

The Jarvis AI Assistant represents a sophisticated integration of modern AI capabilities with practical cybersecurity testing tools. The modular, agent-based architecture provides flexibility for future enhancements while maintaining clear separation of concerns. The system's focus on local processing and storage ensures user privacy while delivering powerful AI-assisted functionality.

The architecture successfully balances complexity with maintainability, providing a solid foundation for both current operations and future expansion into more advanced AI assistant capabilities.

## License

MIT License

