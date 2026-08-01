# Push and deploy CutiVanta Lens

The app uses two large model files. GitHub's normal repository limit is not
suitable for the 635 MB `.keras` model, so install and use Git LFS first.
Streamlit Community Cloud supports Git LFS repositories.

## 1. Test locally

```powershell
cd "C:\Users\sibas\OneDrive\Desktop\Skin\Deploy"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`. If an old tab displays a dynamic-module error,
close it and reopen the URL, or use `Ctrl+F5`.

## 2. Create an empty GitHub repository

In GitHub, create a repository such as `cutivanta-lens`. Do not initialize it
with a README or `.gitignore` because this folder already contains them.

## 3. Initialize Git and Git LFS

Install Git LFS from <https://git-lfs.com/> if `git lfs version` is not
recognized. Then run:

```powershell
cd "C:\Users\sibas\OneDrive\Desktop\Skin\Deploy"
git init
git branch -M main
git lfs install
git lfs track "unet_efficientnetv2s_final.keras"
git lfs track "densenet201_best.h5"
git add .gitattributes .gitignore
git add app.py requirements.txt README.md DEPLOYMENT.md .streamlit/config.toml
git add unet_efficientnetv2s_final.keras densenet201_best.h5
git status
git commit -m "Deploy CutiVanta Lens Streamlit app"
git remote add origin https://github.com/YOUR_USERNAME/cutivanta-lens.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Before committing, confirm
that `git status` does not include `unet_efficientnetv2s_final.h5`; it is an
unused 381 MB duplicate and is intentionally ignored.

## 4. Deploy on Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io/>.
2. Connect the GitHub account containing the repository.
3. Select **Create app** and choose the `cutivanta-lens` repository.
4. Select branch `main` and entrypoint `app.py`.
5. In **Advanced settings**, select **Python 3.12**. TensorFlow 2.18 does not
   provide Python 3.14 wheels, so do not select Python 3.14.
6. Deploy and monitor the build logs while dependencies and LFS models load.

### If the app was accidentally created with Python 3.14

Streamlit does not support changing the Python version of an existing deployed
app. In Streamlit Community Cloud:

1. Open the app and select **Manage app**.
2. Save the current repository, branch, entrypoint, subdomain, and any secrets.
3. Delete the deployed app (this does not delete the GitHub repository).
4. Select **Create app** and choose the same repository and `app.py`.
5. Open **Advanced settings**, choose **Python 3.12**, and deploy again.

The requirements use `tensorflow-cpu==2.18.0`. It loads the same Keras models
and exposes the same `tensorflow` Python import while avoiding the much larger
GPU-enabled Linux wheel.

## Resource note

Streamlit Community Cloud documents a maximum of roughly 2.7 GB memory. These
TensorFlow models are large and may use substantially more memory after loading
than their compressed files use on disk. If the app reports a resource-limit
error, deploy the same repository on a service with more RAM, or export smaller
quantized/TFLite models and update `app.py` to use them.

Official references:

- Streamlit file organization and Git LFS: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization>
- Streamlit deployment steps: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>
- GitHub Git LFS limits: <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage>
