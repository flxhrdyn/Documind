# Deployment

## Backend - Azure Container Apps

1. Build & push image:
   ```bash
   az acr build --registry <registry> --image invenioai-api:latest \
     --file backend/Dockerfile.api .
   ```
2. Create/update the Container App with env + secrets:
   - Secrets: `GROQ_API_KEY`, `QDRANT_URL` (and `QDRANT_API_KEY` if used).
   - Env: `INVENIOAI_ALLOWED_ORIGINS=https://<your-app>.vercel.app`
   - Do NOT set `INVENIOAI_API_KEY` (public, no-auth deployment).
   - Target port: 8000. Enable ingress (external).
3. Note the app FQDN, e.g. `https://invenioai-api.<region>.azurecontainerapps.io`.

## Frontend - Vercel

1. Import the repo in Vercel. Set **Root Directory** to `frontend-react`.
2. Framework preset: Vite. Build command `npm run build`, output `dist`.
3. Environment variable:
   - `VITE_API_BASE_URL=https://invenioai-api.<region>.azurecontainerapps.io`
4. Deploy. After the first deploy, copy the production domain and set it as
   `INVENIOAI_ALLOWED_ORIGINS` on the Azure Container App, then redeploy the backend.

## Old Streamlit HF Space

Left running, untouched, as an archived demo. Not maintained.
