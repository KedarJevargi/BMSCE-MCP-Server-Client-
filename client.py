import asyncio
import json
import sys
from typing import Optional, Dict, Any
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
        # Map tools that require arguments for easy validation
        self.tool_arg_map = {
            'query_knowledge_base': ['query_text'],
            'get_professor_details': ['name']
        }

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

    async def process_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
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
                print("Falling back to non-streaming...")
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

        # Optimized prompt - **CRITICAL: Added aggressive filtering instruction**
        prompt = f"""You are BMSCE Assistant, a friendly AI for BMS College students.

Student asked: "{user_query}"

Data retrieved:
{raw_data}

Present this naturally and conversationally. Use neutral pronouns (they/them) for people.

Guidelines:
- Be warm and helpful like a senior student
- Use clear numbering or bullet points for lists
- **Crucially: Only include coherent, well-formed contact details (Name, Role, Contact). Aggressively filter out and discard any fragmented, garbled, or unformatted text fragments.**
- Keep it concise but informative
- Use emojis sparingly
- End with a friendly note if appropriate

Your response:"""

        await self.generate_response(prompt, RESPONSE_TEMPERATURE, RESPONSE_MAX_TOKENS)

    async def _handle_chat_fallback(self, user_message: str, error_message: str = None):
        """
        Handles the conversational response when no tool is selected or a tool fails.
        """
        # Optimized chat prompt - **CRITICAL: Added hallucination prevention rule**
        chat_prompt = f"""You are BMSCE Assistant for BMS College students.

User: {user_message}
{f"The previous attempt to find information failed with: '{error_message}'. Do NOT mention this error to the user." if error_message else ""}

Respond warmly and concisely. 
No markdown formatting. Do not use any Markdown formatting like bolding (`**...**`) or italics.
Focus ONLY on answering the user's message.
**CRITICAL RULE: If you cannot find the answer in the provided tools/data, state clearly that you do not have that specific information, and DO NOT fabricate names, roles, or details.**

Response:"""
        
        # Generate chat response (with or without streaming based on config)
        await self.generate_response(chat_prompt, CHAT_TEMPERATURE, CHAT_MAX_TOKENS)

    async def chat_with_mistral(self, user_message: str):
        """
        Chat with Mistral using MCP tools with improved tool selection and error handling.
        """

        # Optimized decision prompt - STRICT JSON output requested
        decision_prompt = f"""Analyze the user's question and select the single BEST tool to answer it.

Tools:
1. get_latest_news - news, events, festivals, workshops
2. get_college_notifications - official notices, announcements, deadlines
3. query_knowledge_base - search syllabus, academic topics,clubs,research and development,exams, rules and regulations  (requires "query_text")
4. get_professor_details - professor info (requires "name")
5. none - greetings, casual chat

Question: "{user_message}"

Respond ONLY with a JSON object. Do not include any text, thoughts, or markdown code fences (```json).
The JSON must follow this structure: {{"tool": "tool_name", "arguments": {{"arg1": "value1", ...}}}}

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
        
        # 1. Handle 'none' tool or failed extraction
        if not tool_call or tool_call.get('tool') == 'none':
            await self._handle_chat_fallback(user_message)
            return

        tool_name = tool_call.get('tool')
        tool_args = tool_call.get('arguments', {})

        # 2. Basic Argument Validation 
        required_args = self.tool_arg_map.get(tool_name)
        if required_args and not all(arg in tool_args and tool_args[arg] for arg in required_args):
            print(f"⚠️ Tool selected: '{tool_name}', but missing required arguments or arguments were empty. Falling back to chat.\n")
            await self._handle_chat_fallback(user_message, f"Tool selection failed due to missing arguments for {tool_name}.")
            return

        # 3. Process Tool Call
        try:
            # Show a loading indicator
            print("🔍 Searching...", end='', flush=True)
            raw_data = await self.process_tool_call(tool_name, tool_args)
            print("\r" + " " * 20 + "\r", end='', flush=True) # Clear the loading message
            
            # 4. Error/No Results Detection and Graceful Fallback
            is_error = False
            try:
                data_json = json.loads(raw_data)
                
                # Check for explicit error or 'not found' messages from the server tools
                if isinstance(data_json, dict):
                    error_message = data_json.get('error', '').lower()
                    if 'error' in data_json or 'not found' in error_message:
                        is_error = True
                elif isinstance(data_json, list) and len(data_json) == 0:
                    is_error = True
                        
            except json.JSONDecodeError:
                pass  # Not JSON, proceed with natural response
            
            if is_error:
                # CRITICAL: Do NOT print the raw error message to the user!
                print(f"🤔 The search for that information didn't return a result. Let me try answering conversationally.")
                await self._handle_chat_fallback(user_message, raw_data)
                return

            # 5. Generate the natural response
            await self.make_natural_response(user_message, raw_data)

        except Exception as e:
            print(f"Oops! I had trouble getting that information. Could you try asking in a different way? 😊\n")
            # Fallback on connection/tool execution error
            await self._handle_chat_fallback(user_message, f"Tool execution failed: {e}")

    def _extract_tool_call(self, text: str) -> Optional[dict]:
        """Extract tool call from LLM response"""
        # Improved extraction
        try:
            text = text.replace('```json', '').replace('```', '').strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = text[start:end]
                tool_call = json.loads(json_str)
                if 'tool' in tool_call and 'arguments' in tool_call:
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

    try:
        await client.connect_to_server("main.py")

        print("💬 Hey there! Ask me about college events, notifications, or anything else!")
        print("   Type 'quit' or 'exit' when you're done.\n")
        print("─" * 60 + "\n")

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
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to run client or connect to server. Details: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())