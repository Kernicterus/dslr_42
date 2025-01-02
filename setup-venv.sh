#!/bin/bash

VENV_NAME="venv"

if [ ! -f "requirements.txt" ]; then
    echo "Erreur : Le fichier requirements.txt n'existe pas dans le répertoire actuel."
    exit 1
fi

echo "Création de l'environnement virtuel..."
python3 -m venv $VENV_NAME

source "$VENV_NAME/bin/activate"

echo "Installation des dépendances depuis requirements.txt..."
pip install -r requirements.txt

if [ ! -d ".vscode" ]; then
    mkdir .vscode
fi

echo "Configuration de VS Code pour utiliser l'environnement virtuel..."
cat <<EOL > .vscode/settings.json
{
    "python.pythonPath": "\${workspaceFolder}/$VENV_NAME/bin/python"
}
EOL