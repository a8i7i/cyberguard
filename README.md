# CyberGuard: layered AI threat detection (IBM Z Datathon 2026)

Detects scam messages, malicious URLs, suspicious images (OCR) and anomalous device/app behaviour in one Streamlit dashboard.

## Run
```
pip install -r requirements.txt
sudo apt install tesseract-ocr        # for the image module (optional)
streamlit run app.py
```
Optional: set `VT_API_KEY` (or paste the key in the sidebar) for live VirusTotal lookups.

## Modules
| Module | Technique |
|---|---|
| SMS / message | TF-IDF + Logistic Regression on the SMS Spam Collection, with word-level explanation |
| URL check | Heuristics (IP host, risky TLD, shorteners, keywords...) + VirusTotal API v3 |
| Image | Tesseract OCR -> same message + URL pipeline |
| Anomaly | Isolation Forest on simulated telemetry (CPU, memory, network, failed logins, API calls) |
| Storage | SQLite scan history (swap for a cloud DB in production) |

## IBM Z notes
Only lightweight libraries (scikit-learn, pandas, numpy) are used; no PyTorch/TensorFlow, which keeps the stack portable to Linux on s390x.
