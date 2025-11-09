import json
from fastmcp import FastMCP
from web_scrap import get_news_events, get_notifications
from vector_db import collection

# --- 1. IMPORT THE DATA ---
from professor_resources import PROFESSOR_DATA

mcp = FastMCP("MCP for BMS College of Engineering")



# --- 3. TOOLS ---

@mcp.tool()
def get_latest_news():
    """
    Extracts the 'News & Events' Website,
    and returns the data as a JSON string.
    """ 
    return get_news_events()


@mcp.tool()
def get_college_notifications():
    """
    Extracts 'College Notifications' from the Website,
    and returns the data as a JSON string.
    """
    return get_notifications()


@mcp.tool()
def query_knowledge_base(query_text: str, n_results: int = 3) -> str:
    """
    Queries the ChromaDB vector store to find the most relevant document chunks for a given text query.
    """
    if not collection:
        return json.dumps({"error": "Cannot query. ChromaDB collection is not available."})
    
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return json.dumps(results['documents'][0], indent=2)
    except Exception as e:
        return json.dumps({"error": f"An error occurred during the query: {e}"})


# --- MODIFIED TOOL ---
@mcp.tool()
def get_professor_details(name: str) -> str:
    """
    Searches for and returns the complete details for a specific professor by their name.
    Use this to find a professor's email, phone, department, or specialization.
    This search is flexible and will find partial matches.
    """
    search_name = name.lower().strip()
    all_professors = PROFESSOR_DATA
    
    found_professors = []
    
    # Use 'in' for a flexible "contains" search instead of '=='
    for prof in all_professors:
        if search_name in prof["name"].lower().strip():
            found_professors.append(prof)
    
    # --- Handle search results ---
    
    if len(found_professors) == 1:
        # Perfect! Found exactly one match.
        return json.dumps(found_professors[0], indent=2)
        
    elif len(found_professors) > 1:
        # Ambiguous match. Return a list of names to the user.
        matches = [p['name'] for p in found_professors]
        return json.dumps({
            "error": "Ambiguous query. Multiple professors found.",
            "matches": matches
        })
        
    else:
        # No professor found
        return json.dumps({"error": f"Professor '{name}' not found."})


# --- Main execution ---
if __name__ == "__main__":
    mcp.run()