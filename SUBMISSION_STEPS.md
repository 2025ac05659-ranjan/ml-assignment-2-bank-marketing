# Assignment 2 Submission Steps

## Step 1 - Verify Files

Confirm the project folder contains:

- `app.py`
- `requirements.txt`
- `README.md`
- `test_data.csv`
- `data/bank-additional-full.csv`
- `model/train_models.py`
- `outputs/model_comparison.csv`

## Step 2 - Run on BITS Virtual Lab

On BITS Virtual Lab, open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal and upload `test_data.csv`.

Take one screenshot showing the assignment/app execution on BITS Virtual Lab.
This screenshot is worth 1 mark.

## Step 3 - Push to GitHub

Create a new GitHub repository and push this project.

Recommended commit sequence:

```bash
git init
git add data/bank-additional-full.csv requirements.txt
git commit -m "Add dataset and requirements"
git add model/train_models.py
git commit -m "Add model training and evaluation pipeline"
git add app.py test_data.csv outputs README.md
git commit -m "Add Streamlit app and assignment report content"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Step 4 - Deploy on Streamlit Community Cloud

1. Go to https://streamlit.io/cloud
2. Sign in with GitHub.
3. Click New App.
4. Select your repository.
5. Select branch `main`.
6. Set main file path as `app.py`.
7. Click Deploy.
8. Open the deployed app link and test upload using `test_data.csv`.

## Step 5 - Prepare Final PDF

The assignment PDF must contain these items in this exact order:

1. GitHub repository clickable link
2. Live Streamlit app clickable link
3. Screenshot of assignment execution on BITS Virtual Lab
4. Full README content

Make sure the GitHub and Streamlit links are clickable in the PDF.

## Step 6 - Final Checklist

- GitHub repo link works
- Streamlit app link opens
- App loads without errors
- Uploading `test_data.csv` works
- All required models are available
- Metrics are visible
- Confusion matrix or classification report is visible
- README content is included in the submitted PDF
- BITS Virtual Lab screenshot is included
- Submission is final, not draft

Submit before 18 August 2026, 23:59 IST.
