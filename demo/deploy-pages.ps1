# Deploys the LipScribe demo page (only the 3 frontend files) to Cloudflare Pages.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File demo\deploy-pages.ps1
# First time only:  npx wrangler login

$ErrorActionPreference = "Stop"
$project = "lipscribe"
$site = Join-Path $env:TEMP "lipscribe-site"

if (Test-Path $site) { Remove-Item $site -Recurse -Force }
New-Item -ItemType Directory -Path $site | Out-Null
Copy-Item demo\index.html, demo\app.js, demo\mock-backend.js -Destination $site

# Create the Pages project the first time (ignored if it already exists).
$ErrorActionPreference = "Continue"
try { npx wrangler pages project create $project --production-branch main 2>&1 | Out-Null } catch { }
$ErrorActionPreference = "Stop"

npx wrangler pages deploy $site --project-name $project --branch main --commit-dirty=true
Write-Host "Done. Open https://$project.pages.dev"
