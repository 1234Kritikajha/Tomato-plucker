# Tomato Annotation Lab

This project detects tomatoes in an image, lets the user annotate each tomato as `Ripe`, `Unripe`, or a new custom class, and can learn from the annotated data.

## Objective

The system is built as a tomato ripeness classification and continuous learning workflow:

1. A user uploads tomato images.
2. The backend detects tomato regions and classifies each one as `Ripe`, `Unripe`, or another learned class.
3. If a prediction is wrong, the user can select the tomato box, correct its label, or draw a missing tomato box manually.
4. Corrected labels are saved as cropped training samples in `learning_samples/<label>/`.
5. Every correction is also logged in `learning_samples/annotations.jsonl` with the source image name, bounding box, model prediction, corrected label, crop path, and timestamp.
6. The learning manager retrains from the original training data plus saved user samples, then reloads the model so future predictions can improve from user feedback.

Expected outcome: the app does not only label images once; it keeps collecting corrected annotation data and uses that feedback to reduce repeated mistakes over time.

## What You Need

- Mac Terminal
- Python virtual environment already in this folder: `tfenv`
- Node.js / npm

## 1. Open The Project Folder

Open Terminal and run:

```bash
cd /Users/sysname/Documents/tomatoplucker
```

## 2. Activate Python Environment

Run:

```bash
source tfenv/bin/activate
```

Your terminal should show something like:

```text
(tfenv)
```

## 3. Install Python Packages

Run this once:

```bash
pip install -r requirements.txt
```

## 4. Start Backend API

Keep this terminal open.

```bash
./tfenv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

When it is working, you will see:

```text
Uvicorn running on http://127.0.0.1:8000
```

Backend API docs:

```text
http://127.0.0.1:8000/docs
```

## 5. Start Frontend

Open a second Terminal window.

Run:

```bash
cd /Users/sysname/Documents/tomatoplucker
npm run dev
```

When it is working, you will see:

```text
Tomato frontend running at http://127.0.0.1:5173
```

Open this in your browser:

```text
http://127.0.0.1:5173
```

## Daily Run Commands

Terminal 1:

```bash
cd /Users/sysname/Documents/tomatoplucker
source tfenv/bin/activate
./tfenv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd /Users/sysname/Documents/tomatoplucker
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

## How To Use The App

1. Upload a tomato image.
2. Click `Detect all tomatoes`.
3. The app will draw boxes around detected tomatoes.
4. Click any tomato box to select it.
5. Change its label to `Ripe`, `Unripe`, or another class.
6. If a tomato is missed, click `Draw tomato box` and drag around that tomato.
7. Add new class names using the `Add class` field.
8. Click `Learn annotations` to save labeled tomato crops and start retraining.

## If Port Is Already In Use

If you see:

```text
address already in use
```

It means the project is already running in another terminal.

Check backend port:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

Check frontend port:

```bash
lsof -nP -iTCP:5173 -sTCP:LISTEN
```

You will see a `PID` number. Stop it like this:

```bash
kill PID_NUMBER
```

Example:

```bash
kill 12345
```

Then run the backend/frontend commands again.

## Command Line Prediction

To test one image from Terminal:

```bash
./tfenv/bin/python predict.py dataset/images/Test/Ripe/IMG_20220514_090722.jpg
```

To use your own image:

```bash
./tfenv/bin/python predict.py /path/to/your/tomato-image.jpg
```

## API Test Command

Use this only if you want to test backend directly:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@dataset/images/Test/Unripe/IMG_20220506_093233.jpg"
```

## Train Model Manually

Usually the app learns from annotations using `Learn annotations`.

If you want to run full training manually:

```bash
source tfenv/bin/activate
./tfenv/bin/python tensor.py
```

Training folders:

- `dataset/images/Train/Ripe`
- `dataset/images/Train/Unripe`
- `dataset/images/val/Ripe`
- `dataset/images/val/Unripe`
- `dataset/images/Test/Ripe`
- `dataset/images/Test/Unripe`

Model files:

- `models/tomato_model.keras`
- `models/class_names.json`

## Stop The Project

In each running terminal, press:

```text
Control + C
```

If that does not work, use the port commands from `If Port Is Already In Use`.
