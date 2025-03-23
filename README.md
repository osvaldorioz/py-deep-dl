
El algoritmo **Deep Reinforcement Learning (DRL)** es una técnica de aprendizaje automático que combina **Redes Neuronales Profundas (DNNs)** con **Aprendizaje por Refuerzo (RL)**. Su objetivo es entrenar a un agente para que tome decisiones óptimas en un entorno desconocido, maximizando una recompensa acumulada a largo plazo.

En este programa se utiliza el algoritmo **Deep Q-Learning (DQL)**, que es una versión mejorada del Q-Learning tradicional. DQL emplea una red neuronal para aproximar la función de valor de acción Q(s,a), que estima la recompensa esperada para cada acción \( a \) en un estado \( s \).

---

### **Implementación del Algoritmo en el Programa**

#### **1. Estructura del Código**
El código tiene dos partes principales:
- **C++ con Pybind11**: Implementa el cálculo de la política de aprendizaje, la actualización de la función Q(s,a) y la toma de decisiones.
- **Python**: Se encarga de ejecutar el entrenamiento y visualizar los resultados.

#### **2. Funcionamiento del Algoritmo**
1. **Inicialización**  
   - Se define un agente **DeepQLearningAgent** con un espacio de estados y acciones.
   - Se inicializan los parámetros clave: tasa de aprendizaje (α), factor de descuento (γ), y epsilon (ϵ) para la exploración-explotación.

2. **Ciclo de Entrenamiento**  
   - Para cada episodio:
     1. Se obtiene el estado inicial.
     2. Se elige una acción usando una política epsilon-greedy.
     3. Se ejecuta la acción y se obtiene la recompensa y el nuevo estado.
     4. Se actualiza la función \( Q(s, a) \) utilizando la ecuación de Bellman:
        ![imagen](https://github.com/user-attachments/assets/44a0d415-d7ff-4df1-9120-8d28992d92e5)

     5. Se repite hasta que se complete el episodio.

3. **Actualización de Epsilon**  
   - Se reduce gradualmente γ (Epsilon) para disminuir la exploración a medida que el agente aprende.

4. **Visualización de Resultados**  
   - Se genera una gráfica en Python que muestra la evolución de la recompensa total a lo largo de los episodios.

---

### **Conclusión**
El código implementa **Deep Q-Learning** en C++ con Pybind11, utilizando Eigen para cálculos matriciales. Python se usa para gestionar el entrenamiento y visualizar los resultados. Esta implementación permite un entrenamiento eficiente del agente sin depender de bibliotecas como PyTorch o TensorFlow, haciendo que el cálculo se realice completamente en C++. 
