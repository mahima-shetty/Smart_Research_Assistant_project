from langserve import RemoteRunnable

# Connect to your endpoints
research = RemoteRunnable("http://localhost:8000/research/")


# Use them like normal runnables
result = research.invoke({"query": "What are quantum computers?"})
print(result)




# from langserve import RemoteRunnable


# # Connect to your endpoints
# summarize = RemoteRunnable("http://localhost:8000/summarize/")

# # Use the summarize endpoint with the correct parameter (text, not query)
# result = summarize.invoke({
#     "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals."
# })

# print(result)