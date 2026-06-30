import os
import glob
import subprocess

# 1. Clear Notebook outputs
for nb in glob.glob('notebooks/*.ipynb'):
    subprocess.run(['jupyter', 'nbconvert', '--clear-output', '--inplace', nb])

# 2. Git rm cached data files (models, metrics, intermediate data)
files_to_remove = []
for ext in ['*.parquet', '*.pkl', '*.png', 'data/08_reporting/*.json', 'data/08_reporting/*.html']:
    files_to_remove.extend(glob.glob(f'data/**/{ext}', recursive=True))
    files_to_remove.extend(glob.glob(f'data/{ext}', recursive=True))

for f in files_to_remove:
    subprocess.run(['git', 'rm', '--cached', f, '--ignore-unmatch'])

# 3. DevOps Fixes
# Delete bad lock file
if os.path.exists('requirements-lock.txt'): os.remove('requirements-lock.txt')

# Consolidate requirements into pyproject.toml (simple string append for now)
try:
    reqs = open('requirements.txt').read().splitlines()
    serve_reqs = open('requirements-serve.txt').read().splitlines()
    with open('pyproject.toml', 'a') as f:
        f.write('\n[project]\ndependencies = [\n')
        for r in reqs:
            if r.strip() and not r.startswith('#'): f.write(f'  "{r.strip()}",\n')
        f.write(']\n\n[project.optional-dependencies]\nserve = [\n')
        for r in serve_reqs:
            if r.strip() and not r.startswith('#'): f.write(f'  "{r.strip()}",\n')
        f.write(']\n')
    os.remove('requirements.txt')
    os.remove('requirements-serve.txt')
    subprocess.run(['git', 'rm', '--cached', 'requirements*.txt', '--ignore-unmatch'])
except Exception as e:
    print('DevOps reqs error:', e)

# 4. Dockerfile non-root user
try:
    dockerfile = open('Dockerfile').read()
    if 'useradd' not in dockerfile:
        dockerfile = dockerfile.replace('COPY . /app', 'RUN useradd -m appuser\nUSER appuser\nCOPY . /app')
        open('Dockerfile', 'w').write(dockerfile)
except:
    pass

