## Tomato Annotation Lab (Windows Guide)

This guide helps you set up, run, and manage the Tomato Annotation Lab on a Windows computer.

## What You Need

* Windows Command Prompt (CMD) or PowerShell
* Python installed and added to your Windows PATH
* Node.js / npm installed for Windows

------------------------------
## 1. Open The Project Folder
Open Command Prompt (CMD) and run:

cd C:\Users\sysname\Documents\tomatoplucker

(Note: Replace sysname with your actual Windows username, or paste your exact folder path).
## 2. Activate Python Environment
Run:

.\tfenv\Scripts\activate

Your terminal prompt will now change to show:

(tfenv) C:\Users\sysname\Documents\tomatoplucker>

## 3. Install Python Packages
Run this once to install dependencies:

pip install -r requirements.txt

## 4. Start Backend API
Keep this terminal window open.

.\tfenv\Scripts\uvicorn app:app --host 127.0.0.1 --port 8000

When it is working, you will see:
Uvicorn running on http://127.0.0.1:8000
Backend API interactive docs:
http://127.0.0.1:8000/docs

## 5. Start Frontend
Open a second Command Prompt window.
Navigate to your folder and start the frontend:

cd C:\Users\sysname\Documents\tomatoplucker
npm run dev

When it is working, you will see:
Tomato frontend running at http://127.0.0.1:5173
Open this address in your web browser:
http://127.0.0.1:5173

------------------------------
## Daily Run Commands (Windows Quick Start)## Terminal 1 (Backend):

cd C:\Users\sysname\Documents\tomatoplucker
.\tfenv\Scripts\activate
.\tfenv\Scripts\uvicorn app:app --host 127.0.0.1 --port 8000

Terminal 2 (Frontend):

cd C:\Users\sysname\Documents\tomatoplucker
npm run dev

------------------------------
## Command Line Prediction (Windows)
To test a single image from your command prompt:

.\tfenv\Scripts\python predict.py dataset\images\Test\Ripe\IMG_20220514_090722.jpg

To test your own image:

.\tfenv\Scripts\python predict.py "C:\path\to\your\tomato-image.jpg"

## Train Model Manually (Windows)
To manually trigger full training:

.\tfenv\Scripts\activate
.\tfenv\Scripts\python tensor.py


