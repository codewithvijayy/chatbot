from fastapi import FastAPI,HTTPException,WebSocket
import asyncio
from retriever.data_retriever import retrieve_data
from model.model import Query

app = FastAPI()


@app.websocket("/ws/v1/chatbot/")
async def chat_bot(websocket:WebSocket):
   origin = websocket.headers.get("origin")
   print("WebSocket origin:", origin)

   await websocket.accept()
   try:
       while True:
            await asyncio.sleep(0.1)
            data = await websocket.receive_json()
            user_query = data.get("query")
            print(user_query)
            result = retrieve_data.data_retrieve(query=user_query)
            await asyncio.sleep(0.1)

            await websocket.send_json({
                "data":result,
                "chatbot":True,
            })
            print(result)
       
   except Exception as e:
       raise HTTPException(500,detail="internal server issue")
   
@app.post("/chat")
def getapi(query : Query):
    result = retrieve_data.data_retrieve(query=query.query)
    return {"data":result}
    
   




    
