# Conversational AI Service with Memory Management

![Project Demo](./demo-video.webm) <!-- Replace with actual video file name -->

A sophisticated conversational AI service that mimics human-like memory patterns with both short-term and long-term memory capabilities, complete with chat history management and user preference controls.

## 🎯 Project Overview

This project implements a production-ready conversational AI system that can:
- Remember user conversations intelligently
- Extract and store important conversation points
- Delete data based on user preferences
- Provide context-aware responses using memory and web search

## 🏗️ Architecture

![Architecture Diagram](./architecture-diagram.png)

The system is designed to mirror human conversation patterns where we remember recent interactions clearly and retain only important points from older conversations.

### Core Architecture Components

- **Short-Term Memory**: Redis-based storage for recent conversations
- **Long-Term Memory**: MongoDB for persistent data and important conversation points
- **Agentic Workflow**: LangGraph-powered decision making
- **Dual Tool System**: Database search + Web search capabilities

## 🧠 Memory Management System

### Short-Term Memory Strategy
- **Recent Chat Storage**: Last 4 conversations stored in Redis
- **Sliding Window Summary**: From the 5th conversation onwards, maintains context through sliding summaries
- **Fast Access**: Redis ensures quick retrieval for real-time conversations

### Long-Term Memory Strategy
- **Cross-Conversation Context**: Maintains user context across different conversation sessions
- **Key Points Extraction**: LLM generates 5 main points from the last 8 conversations
- **Persistent Storage**: Important points stored in both Redis and MongoDB

### Chat History Management
- **Complete History**: All conversations saved in MongoDB for pagination and user access
- **Optimized Cache**: Redis maintains only the most recent 8 messages for processing
- **Automatic Cleanup**: Older messages automatically rotated out to maintain performance

## 🔄 Hydration & Flushing System

### User Session Management
- **Logout Flushing**: All Redis memory data flushed to MongoDB on logout
- **Login Hydration**: User data reloaded from MongoDB to Redis on login
- **Seamless Experience**: Fast and smooth user experience through intelligent caching

### Production Alternatives
- **TTL-Based Approach**: Time-to-Live mechanism for microservice architectures
- **Stateless Design**: Can operate without explicit login/logout tracking

## 🤖 Agentic Workflow (LangGraph)

### Agent State Management
```python
class AgentState(TypedDict):
    messages: List[Message]
    user_id: str
    conversation_id: str
    current_response: Optional[str]
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[List[Dict]]
```

### Intelligent Tool Selection

#### 1. DB Search Tool
- Searches user's conversation history and memory
- Provides contextual responses based on past interactions
- Prioritizes user-specific information

#### 2. Web Search Tool
- Activated when information isn't available in memory
- Handles real-time/current information requests
- Uses Tavily web search (OpenAI compatible)

### Example Workflow
```
User: "I am Kaushik"
AI: "Hi Kaushik!"

User: "What is my name?"
AI: "Your name is Kaushik" (from memory)

User: "What's the current temperature in Bangalore?"
AI: [Triggers web search for real-time data]
```

## 🛠️ Technical Stack

- **LLM**: Gemini-1.5-flash (OpenAI compatible via LangChain)
- **Memory Cache**: Redis
- **Database**: MongoDB
- **Framework**: FastAPI
- **Agent Framework**: LangGraph
- **Search**: Tavily Web Search
- **Deployment**: Docker & Docker Compose

## 🚀 Getting Started

### Prerequisites
- Docker and Docker Compose
- Gemini API Key (or OpenAI API Key)
- Tavily API Key for web search

### Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd conversational-ai-service
```

2. **Environment Configuration**
```bash
cp .env.example .env
# Add your API keys to .env file
```

3. **Start Services**
```bash
docker-compose up --build
```

4. **Health Check**
```bash
curl http://localhost:8000/health
```

## 📊 Demo & Screenshots

### System Architecture
![Architecture Overview](./architecture-diagram.png)

### API Responses Demo
![API Demo](./api-demo-screenshot.png) <!-- Add actual screenshot -->

### Memory Management in Action
![Memory Management](./memory-demo-screenshot.png) <!-- Add actual screenshot -->

### Video Walkthrough
[Watch Full Demo Video](./demo-video.mp4) <!-- Replace with actual video file -->

## 🔮 Future Enhancements

### Performance Improvements
- **Server-Sent Events (SSE)**: Streaming responses like ChatGPT/Gemini
- **Parallel Processing**: Memory management and response generation in parallel
- **Response Caching**: Intelligent caching for frequently asked questions

### Production Features
- **Guardrails**: Polite handling of abusive or sensitive queries
- **Rate Limiting**: API protection and fair usage policies especially token rate limiting
- **Monitoring**: Comprehensive logging and metrics
- **A/B Testing**: Response quality optimization

### Advanced Capabilities
- **Custom Model Hosting**: On-premises deployment using vLLM
- **Multi-modal Support**: Image and voice conversation capabilities
- **Advanced Analytics**: Conversation insights and user behavior analysis

## 🏢 Production Experience

This project leverages production experience gained at **ZenteiQ.ai** (Indian Institute of Sciences, Bangalore), where similar conversational AI services have been deployed at scale. The architecture considers real-world factors including:

- Scalability patterns for high-traffic scenarios
- Data privacy and security compliance
- Cost optimization strategies
- User experience optimization
- System reliability and fault tolerance

## 👨‍💻 About the Developer


I have just graduated from MSRIT Bangalore and have learnt a lot here at ZenteiQ.ai how a product is built end-to-end from scratch. I am relly very interested in joining Shram.ai and want to contribute to the products being developed here at Shram.AI and be a part of the dynamic team.

I am always ready for interview to discuss how we can make this Conversational-AI into production if I join Shram.AI and how can I contribute in making this product.

