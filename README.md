# MEDA_Challenge

## Directorio Compartido

En la raíz de este proyecto ejecutar ```ln -s /lustre/proyectos/p032 compartido```. Este será el enlace
simbólico hacia el directorio que compartimos todos. En este se encuentran tanto los datasets como el entorno de python.

## Pasos para ejecutar una libreta con GPU

1. Conectarse a la Yuca en una terminal ```ssh yuca```.
1. Colocarse en la raíz del proyecto ```MEDA_CHALLENGE```.
2. Ejecutar ```./scripts/gpu_session_start.sh```. 
3. Ejecutar ```./scripts/jupyter_gpu_start.sh```.
4. Abrir otra terminal aparte y hacer un tunel ssh con ```ssh -L 9999:amdgpu-hpc-01:10000 yuca```
5. Copiar el enlace de la primera terminal y agregarlo como kernel en la libreta de jupyter de NUESTRA computadora. ```http://127.0.0.1:10000/lab?token=xxxxx``` es como se verá.
6. Ya puedes correr tu libreta.

> [!IMPORTANT]
> Ten cuidado con el puerto en el que esta conectado, a veces es el 10000.

> [!IMPORTANT]
> Solo una persona a la vez puede usar los recursos de la GPU.
