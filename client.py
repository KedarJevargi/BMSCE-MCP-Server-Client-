import asyncio
import json
import sys
from typing import Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# For Ollama integration
import ollama

# Import configuration
from config import (
    LLM_MODEL, TOOL_SELECTION_TEMPERATURE, RESPONSE_TEMPERATURE,
    CHAT_TEMPERATURE, TOOL_SELECTION_MAX_TOKENS, RESPONSE_MAX_TOKENS,
    CHAT_MAX_TOKENS, TOP_P, ENABLE_STREAMING
)

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.available_tools = []
        self.client_context = None
        self.session_context = None

    async def connect_to_server(self, server_script_path: str):
        """Connect to the MCP server"""
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        self.client_context = stdio_client(server_params)
        self.stdio, self.write = await self.client_context.__aenter__()

        self.session_context = ClientSession(self.stdio, self.write)
        self.session = await self.session_context.__aenter__()

        await self.session.initialize()

        response = await self.session.list_tools()
        self.available_tools = response.tools
        print(f"✅ Connected to MCP Server\n")

    async def process_tool_call(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool call on the MCP server"""
        if not self.session:
            raise RuntimeError("Not connected to server")

        result = await self.session.call_tool(tool_name, tool_args)
        return result.content[0].text

    def get_tools_for_llm(self) -> str:
        """Convert MCP tools to natural language description"""
        tools_desc = []
        for tool in self.available_tools:
            desc = f"- {tool.name}: {tool.description}"
            tools_desc.append(desc)
        return "\n".join(tools_desc)

    async def generate_response(self, prompt: str, temperature: float, max_tokens: int):
        """Generate response with optional streaming based on config"""
        
        if ENABLE_STREAMING:
            # Stream the response
            try:
                stream = ollama.generate(
                    model=LLM_MODEL,
                    prompt=prompt,
                    stream=True,
                    options={
                        "temperature": temperature,
                        "top_p": TOP_P,
                        "num_predict": max_tokens,
                    }
                )

                for chunk in stream:
                    if 'response' in chunk:
                        print(chunk['response'], end='', flush=True)
                
                print("\n")  # Add newline at the end
                
            except Exception as e:
                print(f"Error during streaming: {e}")
                # Fallback to non-streaming
                response = ollama.generate(
                    model=LLM_MODEL,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "top_p": TOP_P,
                        "num_predict": max_tokens,
                    }
                )
                print(f"{response['response'].strip()}\n")
        else:
            # Non-streaming response
            response = ollama.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": TOP_P,
                    "num_predict": max_tokens,
                }
            )
            print(f"{response['response'].strip()}\n")

    async def make_natural_response(self, user_query: str, raw_data: str):
        """Convert raw JSON data into natural, student-friendly response"""

        # Optimized prompt - more concise
        prompt = f"""You are BMSCE Assistant, a friendly AI for BMS College students.

Student asked: "{user_query}"

Data retrieved:
{raw_data}

Present this naturally and conversationally. Use neutral pronouns (they/them) for people.

Guidelines:
- Be warm and helpful like a senior student
- Use clear numbering for lists
- Keep it concise but informative
- Use emojis sparingly
- End with a friendly note if appropriate

Your response:"""

        await self.generate_response(prompt, RESPONSE_TEMPERATURE, RESPONSE_MAX_TOKENS)

    async def chat_with_mistral(self, user_message: str):
        """Chat with Mistral using MCP tools"""

        # Optimized decision prompt - more concise
        decision_prompt = f"""Analyze and select ONE tool:

Tools:
1. get_latest_news - news, events, festivals, workshops
2. get_college_notifications - official notices, announcements, deadlines
3. query_knowledge_base - search syllabus, academic topics (needs "query_text")
4. get_professor_details - professor info (needs "name")
5. none - greetings, chat

Question: "{user_message}"

Respond ONLY with JSON:
{{"tool": "tool_name", "arguments": {{}}}}

JSON:"""

        # Run with lower temperature for faster, more deterministic responses
        decision_response = ollama.generate(
            model=LLM_MODEL,
            prompt=decision_prompt,
            options={
                "temperature": TOOL_SELECTION_TEMPERATURE,
                "top_p": 0.5,
                "num_predict": TOOL_SELECTION_MAX_TOKENS,
            }
        )

        tool_call = self._extract_tool_call(decision_response['response'])

        if tool_call and tool_call.get('tool') != 'none':
            tool_name = tool_call['tool']
            tool_args = tool_call.get('arguments', {})

            try:
                # Show a loading indicator while fetching data
                print("🔍 Searching...", end='', flush=True)
                raw_data = await self.process_tool_call(tool_name, tool_args)
                print("\r" + " " * 20 + "\r", end='', flush=True)  # Clear the loading message
                
                # Check for error or no results
                try:
                    data_json = json.loads(raw_data)
                    
                    # Handle various error formats
                    if isinstance(data_json, dict):
                        if data_json.get('error'):
                            print(f"Oops! {data_json['error']}\n")
                            return
                        elif data_json.get('no_results'):
                            print(f"I couldn't find relevant information about that. {data_json.get('message', 'Try asking something else!')} 😊\n")
                            return
                    elif isinstance(data_json, list) and len(data_json) == 0:
                        print("I couldn't find any information about that. Could you rephrase your question? 😊\n")
                        return
                        
                except json.JSONDecodeError:
                    pass  # Not JSON, proceed

                # Generate the natural response (with or without streaming based on config)
                await self.make_natural_response(user_message, raw_data)

            except Exception as e:
                print(f"Oops! I had trouble getting that information. Could you try asking in a different way? 😊\n")
        else:
            # Optimized chat prompt
            chat_prompt = f"""You are BMSCE Assistant for BMS College students.

User: {user_message}

Respond warmly and concisely. No markdown formatting.**Do not use any Markdown formatting like bolding (`**...**`) or italics.**

Response:"""

            # Generate chat response (with or without streaming based on config)
            await self.generate_response(chat_prompt, CHAT_TEMPERATURE, CHAT_MAX_TOKENS)

    def _extract_tool_call(self, text: str) -> Optional[dict]:
        """Extract tool call from LLM response"""
        try:
            text = text.replace('```json', '').replace('```', '').strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                tool_call = json.loads(json_str)
                return tool_call
        except:
            pass
        return None

    async def close(self):
        """Close the connection"""
        if hasattr(self, 'session_context') and self.session_context:
            await self.session_context.__aexit__(None, None, None)
        if hasattr(self, 'client_context') and self.client_context:
            await self.client_context.__aexit__(None, None, None)


async def main():
    client = MCPClient()

    streaming_status = "with streaming!" if ENABLE_STREAMING else "without streaming"
    
    print("\n" + "🎓 " * 20)
    print("\n   Welcome to BMSCE Assistant! 🤖")
    print(f"   Your friendly AI helper for all things BMS College")
    print(f"   (Running {streaming_status})\n")
    print("🎓 " * 20 + "\n")

    await client.connect_to_server("main.py")

    print("💬 Hey there! Ask me about college events, notifications, or anything else!")
    print("   Type 'quit' or 'exit' when you're done.\n")
    print("─" * 60 + "\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n👋 See you later! Have an awesome day! 🌟\n")
                break

            if not user_input:
                continue

            print()
            await client.chat_with_mistral(user_input)
            print("─" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\n👋 Catch you later! Take care! 🌟\n")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())