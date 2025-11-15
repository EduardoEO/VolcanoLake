# VolcanoLake

Proyecto de Inteligencia Artificial que implementa un agente de Reinforcement Learning basado en Q-Learning con estrategia epsilon-greedy para navegar entornos tipo grid-world.

## 📋 Descripción

VolcanoLake es un entorno inspirado en FrozenLake de Gymnasium donde el agente debe navegar desde un punto de inicio hasta una meta, evitando peligros (lava, agua) y recolectando recompensas (tesoros) por el camino. El proyecto incluye tres versiones evolutivas que demuestran diferentes enfoques de diseño y complejidad creciente.

## 🎮 Mecánicas del Juego

### Tipos de Casillas
- **S (Start)**: Punto de inicio del agente
- **G (Goal)**: Meta (+10 puntos, termina episodio con éxito)
- **L (Lava)**: Lava (-10 puntos, termina episodio con fracaso)
- **W (Water)**: Agua (-1 punto, causa deslizamiento probabilístico)
- **T (Treasure)**: Tesoro (+5 puntos, se consume al recogerlo)
- **. (Ground)**: Tierra (-0.001 puntos, neutral)

### Sistema de Deslizamiento (Agua)
Cuando el agente está sobre una casilla de agua ('W'), el movimiento se vuelve probabilístico:
- **80%**: Movimiento exitoso en la dirección deseada
- **10%**: Desliza 90° a la izquierda (cardinal) o solo componente vertical (diagonal)
- **10%**: Desliza 90° a la derecha (cardinal) o solo componente horizontal (diagonal)

### Acciones Disponibles
El agente puede moverse en 8 direcciones:
- **0-3**: Movimientos cardinales (Arriba, Derecha, Abajo, Izquierda)
- **4-7**: Movimientos diagonales (Arriba-Derecha, Abajo-Derecha, Abajo-Izquierda, Arriba-Izquierda)

## 📁 Estructura del Proyecto

```
VolcanoLake/
│
├── VolcanoLake_v1/              # Versión base con FrozenLake estándar
│   ├── main.py                  # Script principal de entrenamiento
│   ├── agent.py                 # Implementación del agente Q-Learning
│   └── utils.py                 # Funciones de visualización
│
├── VolcanoLake_v2/              # Versión mejorada con wrappers
│   ├── main.py                  # Script de entrenamiento extendido
│   ├── agent.py                 # Agente Q-Learning mejorado
│   ├── wrappers.py              # Wrappers personalizados de Gymnasium
│   └── utils.py                 # Utilidades de gráficos
│
├── VolcanoLake_v3/              # Entorno personalizado completo ⭐
│   ├── agent/
│   │   └── qlearning_agent.py  # Agente Q-Learning optimizado
│   │
│   ├── envs/
│   │   └── volcano_lake_env.py # Entorno personalizado (Gymnasium API)
│   │
│   ├── maps/                    # Mapas de diferentes tamaños
│   │   ├── map_5x5.csv         # Mapa pequeño (aprendizaje rápido)
│   │   ├── map_25x25.csv       # Mapa mediano (por defecto)
│   │   ├── map_50x50.csv       # Mapa grande (desafío)
│   │   └── map_100x100.csv     # Mapa muy grande (exploracion avanzada)
│   │
│   ├── plots/                   # Gráficas generadas (creadas automáticamente)
│   │   ├── volcanolake_policy_map.png        # Mapa de políticas óptimas
│   │   ├── volcanolake_training_metrics.png  # Métricas de entrenamiento
│   │   └── volcanolake_value_heatmap.png     # Mapa de calor de valores Q
│   │
│   ├── scripts/
│   │   └── generate_maps.py    # Generador automático de mapas CSV
│   │
│   ├── training/
│   │   └── train_agent.py      # Lógica de entrenamiento modular
│   │
│   ├── utils/
│   │   └── plotting.py         # Funciones de visualización avanzada
│   │
│   ├── videos/                  # Videos de episodios (creados automáticamente)
│   │   └── rl-video-episode-99999.mp4  # Último episodio grabado
│   │
│   ├── wrappers/
│   │   └── wrappers.py         # Wrappers adicionales para el entorno
│   │
│   └── main.py                  # Punto de entrada principal
│
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Este archivo
```

## 🔄 Versiones del Proyecto

### VolcanoLake_v1: Base con FrozenLake
**Objetivo**: Familiarización con Gymnasium y Q-Learning básico

- **Entorno**: FrozenLake estándar de Gymnasium (4x4)
- **Características**:
  - Implementación básica de Q-Learning desde cero
  - Política epsilon-greedy simple
  - Con deslizamiento natural (`is_slippery=True`)
  - Entorno predeterminado sin modificaciones
  - Métricas básicas de rendimiento

**Ideal para**: Entender los fundamentos del aprendizaje por refuerzo

### VolcanoLake_v2: Extensión con Wrappers
**Objetivo**: Exploración de wrappers y modificación dinámica del entorno

- **Entorno**: FrozenLake + Wrappers personalizados de Gymnasium
- **Características**:
  - Wrappers de modificación del entorno:
    - `IncreasingHoles`: Añade agujeros progresivamente durante el entrenamiento
    - `LimitedVision`: Restringe la observación del agente (POMDP)
    - `LimitedVisionRewardShaping`: Moldeado de recompensas con información parcial
  - Grabación automática de videos del último episodio
  - Límite de tiempo por episodio (`TimeLimit` wrapper)
  - Estadísticas de entrenamiento extendidas
  - Visualización mejorada de métricas

**Ideal para**: Entender cómo modificar entornos existentes y aplicar transfer learning

### VolcanoLake_v3: Entorno Personalizado ⭐
**Objetivo**: Implementación completa de un entorno custom siguiendo Gymnasium API

- **Entorno**: VolcanoLake completamente personalizado
- **Características principales**:
  - **Entorno propio** con 6 tipos de casillas diferentes
  - **Mapas escalables** vía CSV (5x5 hasta 100x100)
  - **8 direcciones de movimiento** (cardinal + diagonal)
  - **Sistema probabilístico** de deslizamiento en agua
  - **Tesoros dinámicos** que se consumen durante el episodio
  - **Renderizado avanzado** con Pygame (modos `human` y `rgb_array`)
  - **Tracking completo**:
    - Tesoros recolectados por episodio
    - Tasa de éxito/fracaso
    - Longitud de episodios
    - Error TD durante entrenamiento
  - **Clipping de bordes**: El agente no puede salir del mapa
  - **Arquitectura modular** con separación clara de responsabilidades
  - **Generador de mapas** para crear escenarios personalizados

**Ideal para**: Proyectos avanzados y research en RL custom environments

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/VolcanoLake.git
   cd VolcanoLake
   ```

2. **Crear entorno virtual** (recomendado):
   ```bash
   python -m venv venv
   
   # En Windows:
   venv\Scripts\activate
   
   # En Linux/Mac:
   source venv/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar instalación**:
   ```bash
   python -c "import gymnasium; print('Gymnasium OK')"
   ```

## 🎯 Uso

### Ejecutar VolcanoLake_v1
```bash
cd VolcanoLake_v1
python main.py
```

### Ejecutar VolcanoLake_v2
```bash
cd VolcanoLake_v2
python main.py
```

### Ejecutar VolcanoLake_v3
```bash
cd VolcanoLake_v3
python main.py
```

### Parámetros Configurables

Todas las versiones comparten la misma estructura de configuración del agente Q-Learning. Los parámetros principales se encuentran en el archivo `main.py` (v1, v2) o `training/train_agent.py` (v3):

#### Hiperparámetros del Agente Q-Learning

```python
# Número de episodios de entrenamiento
n_episodes = 100_000

# Hiperparámetros del agente
learning_rate = 0.01          # Tasa de aprendizaje (α): controla cuánto aprende en cada paso
discount_factor = 0.99        # Factor de descuento (γ): importancia del futuro vs presente
start_epsilon = 1.0           # Epsilon inicial: 100% exploración al inicio
final_epsilon = 0.1           # Epsilon final: 10% exploración al final
epsilon_decay = start_epsilon / (n_episodes / 2)  # Decaimiento lineal

# Visualización
plot_save = True              # True: guardar en plots/, False: mostrar en pantalla
render_mode = "rgb_array"     # 'human' (ventana), 'rgb_array' (video), None (sin render)
```

#### Configuración de Wrappers

Los wrappers permiten modificar el comportamiento del entorno sin cambiar su código base. Puedes activar/desactivar los que necesites comentando/descomentando las líneas correspondientes.

##### ⚠️ Reglas Importantes:
1. **El orden importa**: Los wrappers se aplican de arriba hacia abajo
2. **RecordEpisodeStatistics** debe ir SIEMPRE al final
3. Wrappers de modificación del entorno van primero, estadísticas al final

---

##### Versión 2 (v2) - Archivo: `main.py`

```python
# Crear entorno base
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# --- WRAPPERS OPCIONALES (comentar/descomentar según necesites) ---

# 1️⃣ IncreasingHoles: Añade agujeros progresivamente durante entrenamiento
env = IncreasingHoles(env, max_holes=2, hole_increase_rate=0.0001)

# 2️⃣ LimitedVisionRewardShaping: Visión limitada + señales de recompensa
#    El agente solo ve la casilla de la derecha
# env = LimitedVisionRewardShaping(env)

# 3️⃣ LimitedVision: Visión limitada según dirección del movimiento
#    El agente ve solo hacia donde se movió (POMDP)
# env = LimitedVision(env)

# 4️⃣ TimeLimit: Límite de pasos por episodio (evita loops infinitos)
env = gym.wrappers.TimeLimit(env, max_episode_steps=20)

# 5️⃣ RecordVideo: Graba el último episodio
video_folder = os.path.join(os.path.dirname(__file__), "videos")
os.makedirs(video_folder, exist_ok=True)
env = gym.wrappers.RecordVideo(
    env, 
    video_folder=video_folder,
    episode_trigger=lambda ep: ep == n_episodes - 1
)

# 6️⃣ RecordEpisodeStatistics: SIEMPRE AL FINAL
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
```

---

##### Versión 3 (v3) - Archivo: `training/train_agent.py`

```python
# Configuración del mapa
map_file_path = 'maps/map_25x25.csv'  # Opciones: map_5x5, map_25x25, map_50x50, map_100x100

# Crear entorno personalizado
env = VolcanoLakeEnv(map_file_path=map_file_path, render_mode=render_mode)

# --- WRAPPERS OPCIONALES (comentar/descomentar según necesites) ---

# 1️⃣ IncreasingHoles: Añade lava dinámicamente (requiere adaptación)
# from wrappers.wrappers import IncreasingHoles
# env = IncreasingHoles(env, max_holes=5, hole_increase_rate=0.0001)

# 2️⃣ LimitedVision: Observación parcial del entorno (POMDP)
# from wrappers.wrappers import LimitedVision
# env = LimitedVision(env)

# 3️⃣ LimitedVisionRewardShaping: Visión limitada + guía con recompensas
# from wrappers.wrappers import LimitedVisionRewardShaping
# env = LimitedVisionRewardShaping(env)

# 4️⃣ TimeLimit: Límite de pasos (ajustar según tamaño del mapa)
env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

# 5️⃣ RecordVideo: Graba episodios clave
video_folder = os.path.join(os.path.dirname(__file__), "..", "videos")
os.makedirs(video_folder, exist_ok=True)
env = gym.wrappers.RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda ep: ep % 10000 == 0 or ep == n_episodes - 1
)

# 6️⃣ RecordEpisodeStatistics: SIEMPRE AL FINAL
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
```

---

#### Guía Rápida de Wrappers

| Wrapper | Qué hace | Cuándo usar |
|---------|----------|-------------|
| **IncreasingHoles** | Añade obstáculos progresivamente | Curriculum learning, aumentar dificultad |
| **LimitedVision** | Restringe observación del agente | Investigación POMDP, visión parcial |
| **LimitedVisionRewardShaping** | Visión limitada + recompensas guía | POMDP + ayuda al aprendizaje |
| **TimeLimit** | Límite máximo de pasos | Siempre (evita loops infinitos) |
| **RecordVideo** | Graba episodios | Análisis visual, demos |
| **RecordEpisodeStatistics** | Métricas de rendimiento | Siempre (análisis de entrenamiento) |

## 📊 Algoritmo Q-Learning

### Ecuación de Actualización (Bellman)
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                      ︸____________︸
                         TD Error
```

**Componentes**:
- **s**: Estado actual del agente
- **a**: Acción ejecutada
- **r**: Recompensa inmediata obtenida
- **s'**: Estado siguiente (después de ejecutar la acción)
- **α** (alpha): Learning rate - controla cuánto actualizamos (0-1)
- **γ** (gamma): Discount factor - importancia del futuro vs presente (0-1)

### Política Epsilon-Greedy

Balance dinámico entre **exploración** (descubrir) y **explotación** (aprovechar):

```python
if random() < epsilon:
    acción = acción_aleatoria()      # EXPLORACIÓN: Probar cosas nuevas
else:
    acción = argmax(Q[estado])       # EXPLOTACIÓN: Usar lo aprendido
    
# epsilon decrece con el tiempo: 1.0 → 0.1
# Más exploración al inicio, más explotación al final
```

## 📈 Métricas de Entrenamiento

El sistema genera automáticamente gráficas con tres métricas clave:

### 1. Recompensas Acumuladas por Episodio
- **Qué muestra**: Suma total de recompensas obtenidas en cada episodio
- **Objetivo**: Curva ascendente indica aprendizaje exitoso
- **Suavizado**: Media móvil de 500 episodios para reducir ruido

### 2. Duración de Episodios
- **Qué muestra**: Número de pasos hasta terminar cada episodio
- **Objetivo**: Duración estable indica convergencia de la política
- **Interpretación**: 
  - Muy corto → Muerte prematura (mala política)
  - Muy largo → Deambular sin propósito
  - Óptimo → Ruta eficiente hacia la meta

### 3. Error TD (Temporal Difference)
- **Qué muestra**: Magnitud de las actualizaciones en la Q-table
- **Objetivo**: Error decreciente indica convergencia
- **Fórmula**: `|r + γ max Q(s',a') - Q(s,a)|`

**Ubicación de las gráficas**: 
- v1/v2: Se muestran en pantalla o se guardan en raíz
- v3: Carpeta `plots/` dentro de VolcanoLake_v3

## 🛠️ Tecnologías Utilizadas

| Librería | Versión | Propósito |
|----------|---------|-----------|
| **Gymnasium** | 1.2.1 | Framework de entornos de RL (sucesor de OpenAI Gym) |
| **NumPy** | 2.3.4 | Operaciones numéricas y manejo de arrays |
| **Matplotlib** | 3.10.7 | Visualización de métricas y gráficas |
| **Pygame** | 2.6.1 | Renderizado gráfico 2D del entorno |
| **tqdm** | 4.67.1 | Barras de progreso para el entrenamiento |

## 📝 Crear Mapas Personalizados (v3)

### Usando el Generador Automático

```bash
cd VolcanoLake_v3/scripts
python generate_maps.py
```

Esto creará 4 mapas predefinidos en la carpeta `maps/`:
- `map_5x5.csv` (25 casillas)
- `map_25x25.csv` (625 casillas)
- `map_50x50.csv` (2,500 casillas)
- `map_100x100.csv` (10,000 casillas)

### Creación Manual

Crea un archivo CSV con la estructura:

```csv
S,.,.,W,L
.,L,.,W,.
T,.,W,.,L
.,L,.,.,T
.,.,L,W,G
```

**Reglas**:
- Debe haber exactamente **1 casilla 'S'** (inicio)
- Debe haber exactamente **1 casilla 'G'** (meta)
- Puedes añadir múltiples 'L' (lava), 'W' (agua), 'T' (tesoros)
- Las casillas '.' representan terreno neutral

**Cargar en el código**:

```python
from envs.volcano_lake_env import VolcanoLakeEnv

env = VolcanoLakeEnv(
    map_file_path='maps/tu_mapa_custom.csv', 
    render_mode='human'
)
```

## 🎓 Contexto Académico

**Asignatura**: Inteligencia Artificial  
**Nivel**: Tercer año de Ingeniería Informática  
**Universidad**: CUNEF Universidad  
**Año académico**: 2025-2026

### Objetivos de Aprendizaje

Este proyecto demuestra:
1. ✅ Implementación práctica de algoritmos de **Reinforcement Learning**
2. ✅ Diseño de **entornos personalizados** siguiendo Gymnasium API
3. ✅ Aplicación de **wrappers** para modificar entornos existentes
4. ✅ Visualización y análisis de **métricas de aprendizaje**
5. ✅ Desarrollo de código modular y bien documentado
6. ✅ Integración de múltiples librerías de Python científico

### Competencias Desarrolladas
- Diseño de sistemas de IA
- Programación orientada a objetos
- Análisis de algoritmos de aprendizaje
- Documentación técnica
- Control de versiones con Git

## 👥 Autores

- **Eduardo Estefanía Ovejero** - Ingeniería Informática, CUNEF Universidad
- **Álvaro Martín García** - Ingeniería Informática, CUNEF Universidad

---

<div align="center">

**VolcanoLake**

*Proyecto académico - CUNEF Universidad - 2025-2026*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2.1-green.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>