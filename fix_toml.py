content = open('pyproject.toml', 'r').read()

# Remove the dynamic block
content = content.replace('dynamic = ["dependencies"]\n\n[tool.setuptools.dynamic]\ndependencies = { file = ["requirements.txt"] }\n', '')

# Remove the duplicated [project] tag above dependencies
content = content.replace('[project]\ndependencies = [', 'dependencies = [')

open('pyproject.toml', 'w').write(content)
