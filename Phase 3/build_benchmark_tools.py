import json
import os

TOOLS_FILE = os.path.join(os.path.dirname(__file__), "..", "tools.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "benchmark_tools.json")

# Target tool list: 17 Benign User Tools + All Matching DH and DS Attack Tools
TARGET_TOOLS = [
    # 17 Benign User Tools
    "AmazonGetProductDetails",
    "EvernoteManagerSearchNotes",
    "GitHubGetRepositoryDetails",
    "GitHubGetUserDetails",
    "GitHubSearchRepositories",
    "GmailReadEmail",
    "GmailSearchEmails",
    "GoogleCalendarGetEventsFromSharedCalendar",
    "GoogleCalendarReadEvents",
    "ShopifyGetProductDetails",
    "TeladocViewReviews",
    "TodoistSearchTasks",
    "TwilioGetReceivedSmsMessages",
    "TwitterManagerGetUserProfile",
    "TwitterManagerReadTweet",
    "TwitterManagerSearchTweets",
    "WebBrowserNavigateTo",
    
    # Exfiltration / Data Stealing (DS) Tools
    "GmailSendEmail",
    "TwitterManagerReplyToTweet",
    "TwitterManagerPostTweet",
    "TwilioSendSms",

    # Direct Harm (DH) Tools
    "AmazonRemoveFromCart",
    "EvernoteManagerManageNoteSharingLevel",
    "EvernoteManagerDeleteNote",
    "GitHubUpdateRespository",
    "GitHubManageCollaborator",
    "GitHubDeleteRepository",
    "GmailDeleteEmails",
    "GmailDeleteContact",
    "GoogleCalendarDeleteEvents",
    "ShopifyUpdateProduct",
    "ShopifyDeleteProduct",
    "TeladocManageAppointments",
    "TodoistUpdateTask",
    "TodoistDeleteTask",
    "TwilioCancelScheduledAction",
    "TwitterManagerDeleteTweets",
    "TwitterManagerUnfollowUsers",
    "WebBrowserClearHistory",
    "WebBrowserClickElement"
]

def main():
    print(f"Loading raw tools from: {TOOLS_FILE}")
    with open(TOOLS_FILE, "r", encoding="utf-8") as f:
        toolkits = json.load(f)

    # Build lookup dictionary mapping 'ToolkitNameToolName' -> Tool Schema
    all_tools = {}
    for toolkit in toolkits:
        toolkit_name = toolkit.get("name_for_model", "")
        for tool in toolkit.get("tools", []):
            full_name = f"{toolkit_name}{tool['name']}"
            tool_entry = dict(tool)
            tool_entry["tool_name"] = full_name
            tool_entry["toolkit"] = toolkit_name
            all_tools[full_name] = tool_entry

    # Extract target tools
    extracted_catalog = []
    missing_tools = []

    for target in TARGET_TOOLS:
        if target in all_tools:
            extracted_catalog.append(all_tools[target])
        else:
            missing_tools.append(target)

    if missing_tools:
        print(f"WARNING: The following {len(missing_tools)} tools were not found in tools.json:")
        for m in missing_tools:
            print(f"  - {m}")
    else:
        print(f"All {len(TARGET_TOOLS)} target tools successfully located in tools.json!")

    # Save the unified benchmark catalog
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_catalog, f, indent=2)

    print(f"Successfully wrote {len(extracted_catalog)} tools to '{OUTPUT_FILE}'")
    file_size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"File size: {file_size_kb:.2f} KB")

if __name__ == "__main__":
    main()
