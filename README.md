# Modelos de Redes Neuronales Inspirados en el Cerebro

Este repositorio contiene el código fuente asociado al Trabajo Fin de Máster titulado **"Modelos de redes neuronales inspirados en el cerebro: integración de aprendizaje por refuerzo y RNN para tareas cognitivas"**, realizado en el Máster en Ciencia de Datos de la Universidad Autónoma de Madrid.

## 🧠 Descripción

El proyecto explora el uso de redes neuronales recurrentes (RNN) entrenadas mediante aprendizaje por refuerzo (AR) para simular procesos cognitivos inspirados en el cerebro. Utiliza una arquitectura **Actor-Crítico** con unidades GRU modificadas y conexiones dispersas, aplicadas a tareas cognitivas simples y entornos de decisión biológicamente motivados.

Se desarrollan dos entornos principales:
- **CartPole**: entorno clásico de control.
- **Elección Económica**: simulación de un experimento (Padoa-Schioppa C, Assad JA. Neurons in the orbitofrontal cortex encode economic value. Nature. 2006 May 11;441(7090):223-6).

## 🛠️ Estructura del repositorio
src:
- modules: módulos implementados para la arquitectura y el entrenamiento de la red.
- notebooks: código completo con los resultados, proceso de entrenamiento y procedimiento.
- requirements: requisitos para correr el código correctamente.

results:
- chekpoints: modelos finales guardados para cada etapa del entrenamiento.
- outputs: resultados del proceso de entrenamiento de cada etapa.
- copy: copia de seguridad de los checkpoints y outputs.

## 📦 Requisitos
- Python 3.12.3
- Gymnasium
- NumPy
- PyTorch
- Matplotlib
- Seaborn
- Jupyter (opcional)

## 📊 Resultados
Incluyen:
- Recompensas por episodio.
- Actividad simulada de neuronas.
- Curvas psicométricas.
- Comparación con datos experimentales.

## 🧪 Inspiración biológica
El proyecto aplica:
- Aprendizaje por refuerzo (REINFORCE + línea base).
- Arquitectura Actor-Crítico.
- Unidades GRU bioinspiradas.
- Evaluación en tareas cognitivas reales.

## 📄 Licencia
Trabajo académico. Para uso y redistribución, consulta con el autor.

## ✍️ Autor
Iñigo Martínez Ciriza

Tutor: Carlos María Alaíz Gudín

Universidad Autónoma de Madrid

Junio 2025
