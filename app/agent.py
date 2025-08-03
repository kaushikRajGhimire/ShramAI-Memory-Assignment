import os
import json
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from memory_manager import MemoryManager
from models import Message

# Initialize clients
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
memory_manager = MemoryManager()

def get_llm():
    """Get Gemini LLM instance"""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
        max_tokens=1000
    )

@tool
async def db_search(query: str, user_id: str, conversation_id: str) -> str:
    """Search in the user's memory context from Redis"""
    try:
        context = await memory_manager.get_context_for_search(user_id, conversation_id)
        
        # Format context for search
        context_parts = []
        
        if context['short_term_messages']:
            context_parts.append("Recent Messages:")
            for msg in context['short_term_messages']:
                context_parts.append(f"- {msg['role']}: {msg['content']}")
        
        if context['slider_summary']:
            context_parts.append(f"\nConversation Summary: {context['slider_summary']}")
        
        if context['long_term_points']:
            context_parts.append("\nLong-term Memory Points:")
            for point in context['long_term_points']:
                context_parts.append(f"- {point}")
        
        if not context_parts:
            return "No relevant information found in memory."
        
        context_str = "\n".join(context_parts)
        
        # Use LLM to find relevant information
        llm = get_llm()
        prompt = f"""Based on the following user's memory context, answer the query: "{query}"

Context:
{context_str}

Please provide a helpful response based on the available information. If the information is not sufficient, indicate what additional information might be needed.

Answer:"""
        
        response = await llm.ainvoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error searching memory: {str(e)}"

@tool
async def web_search(query: str) -> str:
    """Search the web for information using Tavily"""
    try:
        response = tavily_client.search(query, max_results=3)
        results = []
        
        for result in response.get('results', []):
            results.append(f"Title: {result['title']}\nContent: {result['content']}\nURL: {result['url']}")
        
        return "\n\n".join(results) if results else "No results found."
        
    except Exception as e:
        return f"Error searching web: {str(e)}"

class AgentState(TypedDict):
    messages: List[Message]
    user_id: str
    conversation_id: str
    current_response: Optional[str]
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[List[Dict]]

async def process_message(state: AgentState) -> AgentState:
    """Process incoming message and determine response strategy"""
    try:
        messages = state["messages"]
        user_message = messages[-1].content if messages else ""
        user_id = state["user_id"]
        conversation_id = state["conversation_id"]

        # Fetch memory context for richer prompt
        context = await memory_manager.get_context_for_search(user_id, conversation_id)
        context_parts = []
        
        if context['short_term_messages']:
            context_parts.append("**Recent Conversation Messages:**")
            for msg in context['short_term_messages']:
                context_parts.append(f"- {msg['role']}: {msg['content']}")
        
        if context['slider_summary']:
            context_parts.append(f"\n**Previous Conversation Summary:** {context['slider_summary']}")
        
        if context['long_term_points']:
            context_parts.append("\n**Key Points from Our Conversation History:**")
            for point in context['long_term_points']:
                context_parts.append(f"- {point}")
        
        memory_context_str = "\n".join(context_parts) if context_parts else "This appears to be our first interaction or no previous conversation history is available."

        # Enhanced system prompt with conversation context integration
        system_prompt = f"""You are a highly intelligent conversational AI assistant designed to provide helpful, accurate, and contextually appropriate responses. Your primary role is to engage users in natural, meaningful conversations while leveraging your available tools strategically to provide the most valuable assistance possible.

## CONVERSATION HISTORY CONTEXT
The following information represents a summary of your previous interactions with this user. Use this context to maintain conversational continuity, reference past discussions naturally, and provide personalized responses based on your shared history:

{memory_context_str}

## CORE PERSONALITY AND APPROACH
You are knowledgeable, friendly, and professional. Maintain a conversational tone that feels natural and engaging while being informative and helpful. You should adapt your communication style to match the user's needs and the complexity of their inquiry. Always acknowledge and build upon the conversation history when relevant.

## CONTEXTUAL AWARENESS
- Reference previous conversations naturally when appropriate
- Show continuity by remembering user preferences, interests, and past discussions
- Build relationships by demonstrating memory of important details shared by the user
- Adapt your responses based on the user's established communication style and preferences
- Use the conversation summary to provide more personalized and contextually relevant responses

## TOOL SELECTION STRATEGY

### DIRECT RESPONSE SCENARIOS (NO TOOLS NEEDED)
Respond directly and naturally WITHOUT using any tools when the user's message falls into these categories:

**Social/Conversational Interactions:**
- Greetings: "hi", "hello", "hey", "good morning", "how are you"
- Casual conversation: "how's your day", "what's up", "how have you been"
- Polite exchanges: "thank you", "you're welcome", "goodbye", "see you later"
- General small talk about weather, feelings, or personal check-ins
- Follow-up responses to previous conversations that can be answered using the context provided

**Simple Acknowledgments:**
- Confirmations: "okay", "got it", "understood", "sounds good"
- Expressions of gratitude or appreciation
- Basic emotional responses: "that's great", "I'm sorry to hear that"

**Straightforward Questions You Can Answer Directly:**
- General knowledge questions within your training data
- Basic explanations of common concepts
- Simple how-to questions for everyday tasks
- Clarifications about your capabilities or limitations
- Questions that can be answered using the conversation history context provided

### DATABASE SEARCH TOOL USAGE (db_search)
Use the db_search tool when the user's inquiry involves:

**Memory-Related Queries (when context is insufficient):**
- "What did we discuss about..." or "What did I tell you about..." (if not clear from context)
- "Do you remember when I mentioned..." or "Remind me of our conversation about..."
- "What are my preferences for..." or "What do I usually prefer..." (if not in context)
- References to previous conversations: "as we talked about before", "like I said earlier"
- When you need more detailed information about past interactions beyond the summary

**Personal Context and History:**
- Questions about user's stated goals, interests, or background information not in the summary
- Requests for personalized recommendations based on previous interactions
- Follow-up questions that build on earlier conversation context requiring more detail
- User asking about their own information or past statements not covered in the summary

**Contextual Continuity:**
- When the user assumes you know something specific about them not in the provided context
- Questions that require understanding of their personal situation or circumstances
- Requests that depend on established user preferences or past decisions

### WEB SEARCH TOOL USAGE (web_search)
Use the web_search tool when the user asks questions that require:

**Factual Information Queries:**
- "What is..." followed by specific entities, concepts, or phenomena
- "Who is..." questions about people, organizations, or public figures
- "When did..." questions about historical events or recent occurrences
- "Where is/are..." questions about locations, places, or geographical information

**Current and Time-Sensitive Information:**
- Questions about recent news, events, or developments
- "What's happening with..." or "What's the latest on..."
- Current prices, statistics, or market information
- Recent scientific discoveries or technological developments

**Specific Facts and Data:**
- Technical specifications or detailed product information
- Statistical data or research findings
- Company information, financial data, or business details
- Educational or academic information requiring accuracy

**Comparative Research:**
- "Compare X and Y" when requiring current market information
- "What are the differences between..." for products, services, or concepts
- "Which is better..." questions requiring current reviews or data

**Verification and Fact-Checking:**
- When you need to verify claims or statements
- Questions about controversial topics requiring current, authoritative sources
- Requests for citations or sources for specific information

## RESPONSE GUIDELINES

### For Direct Responses:
- Be warm, natural, and conversational
- Show appropriate emotional intelligence and empathy
- Keep responses concise but complete
- Use a friendly, helpful tone that encourages continued conversation
- Reference previous conversations naturally when relevant
- Demonstrate continuity by acknowledging shared history

### For Tool-Assisted Responses:
- Seamlessly integrate tool results into natural conversation
- Provide context for why you used a particular tool
- Synthesize information from multiple sources when relevant
- Always maintain a conversational flow even when presenting factual information
- Connect new information to previous conversations when appropriate

### Quality Standards:
- Prioritize accuracy and helpfulness over speed
- Acknowledge uncertainty when appropriate
- Provide actionable information whenever possible
- Be transparent about your limitations and capabilities
- Show genuine interest in the user's ongoing needs and concerns

### Conversation Flow:
- Build on previous context naturally using the conversation summary
- Ask clarifying questions when user intent is ambiguous
- Offer follow-up suggestions or related information when helpful
- Maintain engagement while respecting user boundaries
- Show progression and growth in your understanding of the user over time

## ERROR HANDLING AND EDGE CASES
- If tool searches return no results, acknowledge this and offer alternative approaches
- When faced with ambiguous queries, ask clarifying questions before tool usage
- If multiple tools might be relevant, prioritize based on the user's apparent immediate need
- Handle technical errors gracefully without exposing system details to users

## CONVERSATION CONTINUITY AND RELATIONSHIP BUILDING
Remember that you're part of an ongoing conversation with this specific user. The conversation history context provides valuable insights into:
- Their communication style and preferences
- Topics they're interested in
- Previous questions they've asked
- Their goals and objectives
- Personal information they've shared

Use this context to:
- Build deeper, more meaningful interactions
- Show genuine interest in their ongoing concerns
- Provide increasingly personalized assistance
- Demonstrate that you value and remember your conversations
- Create a sense of continuity and relationship progression

Your goal is to be a helpful, intelligent, and trustworthy conversational partner who grows more valuable with each interaction.

Current user message: "{user_message}"

Based on this comprehensive guidance and the conversation history context provided, respond appropriately by either engaging directly or utilizing the most suitable tool for the user's needs. Always strive to create responses that feel natural, contextually aware, and personally relevant to this specific user."""

        # Get LLM with tools
        llm = get_llm()
        tools = [db_search, web_search]
        llm_with_tools = llm.bind_tools(tools)
        
        # Create message for LLM
        llm_message = HumanMessage(content=system_prompt)
        
        # Get response
        response = await llm_with_tools.ainvoke([llm_message])
        
        # Check if tools were called
        if hasattr(response, 'tool_calls') and response.tool_calls:
            state["tool_calls"] = response.tool_calls
            state["current_response"] = response.content or ""
        else:
            state["current_response"] = response.content
            state["tool_calls"] = None
        
        return state
        
    except Exception as e:
        state["current_response"] = f"I apologize, but I encountered an error: {str(e)}"
        state["tool_calls"] = None
        return state


async def execute_tools(state: AgentState) -> AgentState:
    """Execute tools if they were called"""
    try:
        if not state["tool_calls"]:
            return state
        
        tool_results = []
        
        for tool_call in state["tool_calls"]:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "db_search":
                # Add user context to db_search
                tool_args["user_id"] = state["user_id"]
                tool_args["conversation_id"] = state["conversation_id"]
                result = await db_search.ainvoke(tool_args)
            elif tool_name == "web_search":
                result = await web_search.ainvoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
            
            tool_results.append({
                "tool": tool_name,
                "result": result
            })
        
        state["tool_results"] = tool_results
        return state
        
    except Exception as e:
        state["tool_results"] = [{"tool": "error", "result": f"Tool execution error: {str(e)}"}]
        return state

async def generate_final_response(state: AgentState) -> AgentState:
    """Generate final response based on tool results or direct response"""
    try:
        if state["tool_results"]:
            # Generate response based on tool results
            user_message = state["messages"][-1].content
            
            # Combine tool results
            tool_info = []
            for result in state["tool_results"]:
                tool_info.append(f"{result['tool']}: {result['result']}")
            
            combined_results = "\n\n".join(tool_info)
            
            # Generate final response
            llm = get_llm()
            prompt = f"""Based on the following search results, provide a helpful and natural response to the user's query.

User Query: {user_message}

Search Results:
{combined_results}

Please provide a natural, conversational response that incorporates the relevant information from the search results. Be helpful and engaging.

Response:"""
            
            response = await llm.ainvoke(prompt)
            state["current_response"] = response.content
        
        # If no tools were used, current_response should already be set
        return state
        
    except Exception as e:
        state["current_response"] = f"I apologize, but I encountered an error while generating the response: {str(e)}"
        return state

def should_use_tools(state: AgentState) -> str:
    """Determine next step based on tool calls"""
    if state["tool_calls"]:
        return "execute_tools"
    else:
        return "generate_response"

def should_generate_response(state: AgentState) -> str:
    """Always generate final response after tools"""
    return "generate_response"

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("process_message", process_message)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("generate_response", generate_final_response)

# Add edges
workflow.set_entry_point("process_message")
workflow.add_conditional_edges(
    "process_message",
    should_use_tools,
    {
        "execute_tools": "execute_tools",
        "generate_response": "generate_response"
    }
)
workflow.add_edge("execute_tools", "generate_response")
workflow.add_edge("generate_response", END)

# Compile the graph
agent = workflow.compile()

async def process_conversation(user_id: str, conversation_id: str, user_message: str) -> str:
    """Process a conversation turn"""
    try:
        # Create user message
        user_msg = Message(role="human", content=user_message)
        
        # Add to memory
        await memory_manager.add_message(user_id, conversation_id, user_msg)
        
        # Create initial state
        initial_state = {
            "messages": [user_msg],
            "user_id": user_id,
            "conversation_id": conversation_id,
            "current_response": None,
            "tool_calls": None,
            "tool_results": None
        }
        
        # Run the agent
        result = await agent.ainvoke(initial_state)
        
        # Get the final response
        ai_response = result["current_response"]
        
        # Save AI response to memory
        ai_msg = Message(role="assistant", content=ai_response)
        await memory_manager.add_message(user_id, conversation_id, ai_msg)
        
        return ai_response
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error: {str(e)}"
        
        # Try to save error response to memory
        try:
            ai_msg = Message(role="assistant", content=error_msg)
            await memory_manager.add_message(user_id, conversation_id, ai_msg)
        except:
            pass  # Don't let memory errors block the response
        
        return error_msg