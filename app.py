from fastapi import FastAPI
import uvicorn
import os 
import sys 
from fastapi.templating import Jija2Template
from startlette.responses import RedirectResponse
from fastapi.responses import Response
from transformer_question_answering.pipeline.