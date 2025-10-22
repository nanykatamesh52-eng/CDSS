import sys
import os

import sqlite3

# __import__('pysqlite3')
# import sys,os
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import warnings
import json
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
import uvicorn
import traceback
from langchain.agents.agent import AgentExecutor


from models.model import QAModel, LocalModel
from models.agent import QAAGent

warnings.filterwarnings("ignore")


@asynccontextmanager
async def lifespan(APP: FastAPI):
    local_llm = LocalModel().llm
    # agent
    qa_agent = QAAGent(local_llm)
    medicine_db = "medicine_info"
    indication_db = "indications_info"
    
    if not os.path.exists(f"{medicine_db}.db"):
        qa_agent.create_db("data/medicines_data.csv", db_name=medicine_db)
    med_db = qa_agent.get_db(db_name=medicine_db)
    med_agent_executor = qa_agent.get_executor(med_db)
    
    if not os.path.exists(f"{indication_db}.db"):
        qa_agent.create_db("data/indications_data.csv", db_name=indication_db)
        
    ind_db = qa_agent.get_db(db_name=indication_db)
    ind_agent_executor = qa_agent.get_executor(ind_db)
    
    APP.state.med_agent_executor = med_agent_executor
    APP.state.ind_agent_executor = ind_agent_executor
    APP.state.qa_agent = qa_agent
    APP.state.descriptions = None
    APP.state.med_db = med_db
    APP.state.ind_db = ind_db
    
    yield

APP = FastAPI(lifespan=lifespan)

origins = ["*"]

APP.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@APP.get('/healthcheck')
def healthcheck():
    return {'msg': "okay"}


@APP.post("/update-db")
async def update_db(file: UploadFile = File(...), column_name_descriptions: Optional[Dict[str, str]] = None):
    try:
        if file:
            # Save the uploaded file to a temporary directory
            temp_dir = ".temp"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, file.filename)
            
            # with open(file_path,"wb") as f:
            #      f.write(await file.read())
         
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_path)                       
            
            # Check if the specified file is a CSV or Excel file
            if not file.filename.lower().endswith('.csv'):
            
                raise ValueError("Error parsing input file. The specified file must be a CSV file.")
            
            # Remove the previous database file
            if os.path.exists("nitco.db"):
                os.remove("nitco.db")
            
            # Create a new database
            APP.state.qa_agent.create_db(file_path,"nitco")
            db = APP.state.qa_agent.get_db("nitco")

            # APP.state.qa_agent.create_db(file_path)
            # db = APP.state.qa_agent.get_db()
            agent_executor = APP.state.qa_agent.get_executor(db)
            APP.state.agent_executor = agent_executor
            APP.state.descriptions = column_name_descriptions
            
            return {"msg": "Database updated successfully"}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@APP.post("/medicines_inference")
async def medicines_inference(query: str):
    try:
        
        if query:
            response = APP.state.qa_agent.run(APP.state.med_agent_executor,
                                              query,
                                              descriptions=APP.state.descriptions,
                                            #   db=APP.state.med_db,
                                              medicine_related=True)
            output = {
                "input": query,
                "output": response["output"],
            }
            return output
        else:
            raise HTTPException(status_code=400, detail="Query is required.")
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@APP.post("/indications-inference")
async def indications_inference(query: str):
    try:
        if query:
            response = APP.state.qa_agent.run(APP.state.ind_agent_executor,
                                              query,
                                              descriptions=APP.state.descriptions,
                                            #   db=APP.state.ind_db,
                                              medicine_related=False)
            
            return response
        else:
            raise HTTPException(status_code=400, detail="Query is required.")
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    uvicorn.run('app:APP', host="0.0.0.0", port=port)
