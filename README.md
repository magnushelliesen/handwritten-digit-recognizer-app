# handwritten-digit-recognizer-app

The app is hosted on [Google Cloud Run](https://handwritten-digit-recognizer-app-569320861368.asia-east1.run.app/). Any changes to the app are rolled out using a [GitHub Action](https://github.com/magnushelliesen/handwritten-digit-recognizer-app/blob/main/.github/workflows/release.yml). (The necessary service account credentials for the Google Cloud project are stored as _repo secrets_.)

A [GitHub Action](https://github.com/magnushelliesen/handwritten-digit-recognizer-app/blob/main/.github/workflows/pre_commit_checks.yml) runs Mypy and Black upon PR into main.

---
<p align="center">
  <img src="https://github.com/user-attachments/assets/f29b80bf-7967-44c6-95a7-09e00ecb3190" alt="Image Description" width="400"/>
</p>

---
The app uses the neural network-class: [https://github.com/magnushelliesen/neural-network](https://github.com/magnushelliesen/neural-network).
