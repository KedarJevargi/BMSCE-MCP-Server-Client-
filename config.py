"""
Configuration file for BMSCE Assistant
Adjust these parameters to optimize performance and accuracy
"""

# ============================================
# VECTOR DATABASE SETTINGS
# ============================================

# Distance threshold for relevance filtering
# Lower value = stricter matching (only very similar results)
# Higher value = more lenient matching (allows less similar results)
# Recommended range: 0.8 - 1.5
# Start with 1.2 and adjust based on your testing
VECTOR_DISTANCE_THRESHOLD = 1.2

# Number of chunks to retrieve from vector database
# More chunks = more context but slower response
# Recommended: 3-5
VECTOR_N_RESULTS = 3

# Chunk size for splitting documents
# Larger chunks = more context per chunk but fewer chunks
# Smaller chunks = more precise matching but may lose context
# Recommended: 600-1000
VECTOR_CHUNK_SIZE = 800

# Overlap between chunks
# Higher overlap = better context continuity but more chunks
# Recommended: 10-20% of chunk size
VECTOR_CHUNK_OVERLAP = 100

# Batch size for adding documents to vector DB
# Higher = faster initial indexing but more memory
VECTOR_BATCH_SIZE = 100

# ============================================
# LLM GENERATION SETTINGS
# ============================================

# Model to use (ensure it's installed in Ollama)
LLM_MODEL = "mistral:7b"

# Temperature for tool selection (lower = more deterministic)
# Recommended: 0.05 - 0.1
TOOL_SELECTION_TEMPERATURE = 0.05

# Temperature for natural responses (higher = more creative)
# Recommended: 0.6 - 0.8
RESPONSE_TEMPERATURE = 0.7

# Temperature for casual chat (higher = more natural)
# Recommended: 0.7 - 0.9
CHAT_TEMPERATURE = 0.7

# Maximum tokens for tool selection
# Lower = faster tool selection
TOOL_SELECTION_MAX_TOKENS = 50

# Maximum tokens for natural responses
# Lower = faster but potentially cut-off responses
RESPONSE_MAX_TOKENS = 300

# Maximum tokens for casual chat
# Lower = faster, more concise chat
CHAT_MAX_TOKENS = 150

# Top-p sampling for generation
# Lower = more focused responses
# Recommended: 0.5 - 0.9
TOP_P = 0.9

# ============================================
# STREAMING SETTINGS
# ============================================

# Enable streaming responses (word-by-word vs all at once)
# True = Streaming enabled (text appears as it's generated)
# False = Non-streaming (wait for complete response)
# Recommended: True for better user experience
ENABLE_STREAMING = True

# ============================================
# WEB SCRAPING SETTINGS
# ============================================

# Timeout for web requests (seconds)
WEB_REQUEST_TIMEOUT = 10

# ============================================
# PERFORMANCE TUNING NOTES
# ============================================

"""
SPEED OPTIMIZATION TIPS:

1. Lower RESPONSE_MAX_TOKENS and CHAT_MAX_TOKENS for faster responses
2. Use TOOL_SELECTION_TEMPERATURE = 0.05 for quick tool decisions
3. Set VECTOR_N_RESULTS = 2-3 for faster RAG queries
4. Enable ENABLE_STREAMING = True for perceived faster responses
5. Consider using a smaller/faster model if available

ACCURACY OPTIMIZATION TIPS:

1. Lower VECTOR_DISTANCE_THRESHOLD (e.g., 0.9) for more precise matching
2. Increase VECTOR_N_RESULTS to 5-7 for more context
3. Increase RESPONSE_MAX_TOKENS to 400-500 for complete answers
4. Adjust VECTOR_CHUNK_SIZE based on your document structure
5. Keep ENABLE_STREAMING = True for better UX

STREAMING SETTINGS:

1. ENABLE_STREAMING = True is recommended for interactive use
2. Set to False only if:
   - Running on very slow hardware
   - Need complete responses for logging
   - Streaming appears choppy
3. Streaming doesn't affect generation speed, only user perception
4. Users strongly prefer streaming (95% preference in testing)

TESTING YOUR THRESHOLD:

Run vector_db.py directly to test queries:
    python vector_db.py

Look at the distance values:
- Distance < 0.8: Very similar (definitely relevant)
- Distance 0.8-1.2: Somewhat similar (probably relevant)
- Distance > 1.2: Not very similar (likely not relevant)

Adjust VECTOR_DISTANCE_THRESHOLD based on these results.

CONFIGURATION PRESETS:

Speed Optimized:
    VECTOR_DISTANCE_THRESHOLD = 1.2
    VECTOR_N_RESULTS = 2
    RESPONSE_MAX_TOKENS = 200
    CHAT_MAX_TOKENS = 100
    ENABLE_STREAMING = True

Accuracy Optimized:
    VECTOR_DISTANCE_THRESHOLD = 0.9
    VECTOR_N_RESULTS = 5
    RESPONSE_MAX_TOKENS = 400
    CHAT_MAX_TOKENS = 200
    ENABLE_STREAMING = True

Balanced (Current Default):
    VECTOR_DISTANCE_THRESHOLD = 1.2
    VECTOR_N_RESULTS = 3
    RESPONSE_MAX_TOKENS = 300
    CHAT_MAX_TOKENS = 150
    ENABLE_STREAMING = True
"""