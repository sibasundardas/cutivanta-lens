# Cutivanta Lens deployment

This Streamlit app provides the complete project analysis:

1. U-Net + EfficientNetV2-S lesion segmentation
2. Padded, masked lesion crop
3. DenseNet201 seven-class classification
4. Grad-CAM and non-diagnostic visual descriptors
5. Illustrated PDF report with the original and lesion-overlay images

The project logo is stored at `assets/cutivanta-logo.png` and is used in the
app header and browser tab.

## Run locally

From the `Deploy` directory:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

If a browser tab was already open while Streamlit restarted and shows a
`Failed to fetch dynamically imported module` message, close that tab and
open `http://localhost:8501` again (or use a hard refresh with `Ctrl+F5`).
The app avoids optional chart/dataframe frontend bundles to reduce this issue.

The files `unet_efficientnetv2s_final.keras` and `densenet201_best.h5` must remain in the same directory as `app.py`.

See [DEPLOYMENT.md](DEPLOYMENT.md) for Git LFS, GitHub push, and Streamlit
Community Cloud deployment steps.

## Important model limitation

The classifier has no normal-skin class and the segmenter was not trained as a lesion-presence detector. The UI therefore reports image/localization warnings but does not claim that an image is normal. A real gatekeeper needs a separately trained normal-vs-lesion or out-of-distribution model.
