# Create repository under RemoteSensingTools

This project is initialized locally. To publish it under **https://github.com/RemoteSensingTools**:

## Option A: GitHub web UI

1. Go to **https://github.com/organizations/RemoteSensingTools/repositories/new**
   - Or: GitHub → RemoteSensingTools → Repositories → **New repository**.

2. Set:
   - **Repository name:** `AtmosTransportModel`
   - **Visibility:** Public (or Private)
   - **Do not** initialize with README, .gitignore, or license (we already have them).

3. Create the repository, then in your local clone run:

```bash
cd /home/cfranken/code/gitHub/AtmosTransportModel
git remote add origin https://github.com/RemoteSensingTools/AtmosTransportModel.git
git push -u origin main
```

## Option B: GitHub CLI (`gh`)

If you have [GitHub CLI](https://cli.github.com/) installed and authenticated:

```bash
cd /home/cfranken/code/gitHub/AtmosTransportModel
gh repo create RemoteSensingTools/AtmosTransportModel --public --source=. --remote=origin --push
```

You need **create-repository** permission in the RemoteSensingTools organization (admin or member with repo creation).
