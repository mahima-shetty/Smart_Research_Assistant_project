conda create -n myenv python==3.11 -y 

conda activate myenv

pip install -r requirements.txt


# run the project 


uvicorn langserve_app:app --reload
python sdk_client.py
streamlit run streamlit_app.py




http://localhost:8000/summarize/playground/ 
http://localhost:8000/research/playground/  "# Smart_Research_Assistant_project" 
