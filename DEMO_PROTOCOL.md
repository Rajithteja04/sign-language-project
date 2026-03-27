# Final Demo Protocol (Module 2 + Module 3)

## Goal
Show live sign recognition (Module 2) and sentence formation (Module 3) using the current MS-ASL demo vocabulary:

- `cousin`
- `eat`
- `finish`
- `nice`
- `teacher`

## Pre-Demo Setup
1. Open terminal in repo:
   - `C:\Users\rajit\FinalYearProject\SignLanguageTranslation\sign-language-project`
2. Activate env:
   - `.\.venv311\Scripts\Activate.ps1`
3. Optional preflight:
   - `python -m scripts.preflight_runtime --camera-index 0`
4. Run app:
   - `python -m app.app`
5. Open:
   - `http://127.0.0.1:5000`

## Live Demo Script
1. Click `Start`.
2. Keep hand steady in frame; show one sign clearly for ~1–2 seconds.
3. Wait for word commit into `Recognized Words`.
4. Repeat for the planned sequence.
5. Show generated sentence in `Generated Sentence`.
6. Click `Clear Sentence` before next test case.

## Recommended 5 Demo Cases

### Case 1
- Input word sequence: `teacher nice`
- Expected sentence: `The teacher is nice.`

### Case 2
- Input word sequence: `eat finish`
- Expected sentence: `I finished eating.`

### Case 3
- Input word sequence: `cousin teacher`
- Expected sentence: `My cousin is a teacher.`

### Case 4
- Input word sequence: `eat`
- Expected sentence: `I want to eat.`

### Case 5
- Input word sequence: `cousin eat finish nice teacher`
- Expected sentence: `My cousin and teacher finished eating, and it was nice.`

## What to Say During Demo
- Module 2:
  - “The model predicts one word at a time from live camera frames.”
  - “Confidence, margin, voting window, and cooldown reduce false positives.”
- Module 3:
  - “Committed words are sent to semantic correction.”
  - “We use T5-based correction with a stable fallback for demo reliability.”

## Troubleshooting
- If camera not starting:
  - close other apps using webcam (Camera/Zoom/Teams), rerun app.
- If UI looks stale:
  - press `Ctrl + F5`.
- If random predictions appear:
  - keep neutral pose, then show sign with clear hand motion and visibility.

