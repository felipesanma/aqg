# aqg
Automatic Question Generation


### Configuración del entorno virtual

Crear un ambiente virtual para instalar las dependencias a nivel de proyecto con el siguiente comando:

Windows:
```
py -m venv nombre_env
```

Linux y macOS:
```
python3 -m venv nombre_env
```

Después de ejecutar estos comandos, se creará una carpeta con el nombre del entorno virtual en el directorio actual. Para activar el entorno virtual, sigue estos pasos adicionales:

Windows (PowerShell):
```
.\nombre_env\Scripts\Activate.ps1
```

Windows (Command Prompt):
```
nombre_env\Scripts\activate.bat
```

Linux y macOS:
```
source nombre_env/bin/activate
```

Una vez inicializado tu entorno virtual puedes instalar las dependencias con el siguiente comando:

Windows:
```
pip install -r requirements.txt
```

Linux y macOS:
```
pip3 install -r requirements.txt
```