# Réplica de `navigation_2d` (Nav2D)

Esta réplica reproduce de forma nativa el experimento de exploración autónoma
del repositorio [`skasperski/navigation_2d`](https://github.com/skasperski/navigation_2d),
sin requerir ROS, Stage, Karto ni RViz durante la ejecución.

La referencia quedó fijada al commit
`3c27da9b0f5699559d9048c13ef4885815193981` para que el experimento no cambie
si el repositorio remoto se modifica.

## Qué se replicó

| Componente Nav2D | Implementación en este simulador |
| --- | --- |
| `tutorial3.launch` | `examples/nav2d_tutorial3_single.sim` |
| `tutorial4.world` (dos poses) | `examples/nav2d_tutorial4_multi.sim` |
| `NearestFrontierPlanner` | `Nav2D nearest-frontier wavefront` |
| `MultiWavefrontPlanner` | `Nav2D multi-wavefront coordinator` |
| Plan global por gradiente/Dijkstra | `Dijkstra` + `Raw grid path` |
| Mapa `autolab_fill.png`, 34 m × 30 m | 54 rectángulos continuos, escala uniforme 0.5 |
| Resolución de mapeo 0.10 m | 0.05 m (escala 0.5) |
| Radio de navegación 0.50 m | radio de seguridad 0.25 m |
| Umbral de rango de Karto 10 m | LiDAR de 5 m |
| Pioneer 3-AT a 0.75 m/s | 0.375 m/s |
| Replan rápido de 1 s | `replan_cooldown = 1.0 s` |

Las posiciones, longitudes, radios y velocidades lineales se multiplicaron
por 0.5 para que el mapa original de 34 m × 30 m quepa sin deformación dentro
del mundo de 20 m × 16 m del simulador. Los ángulos y tiempos se conservaron.
La geometría se obtuvo directamente de los intervalos negros del bitmap de
Stage; no es un redibujo aproximado.

## Algoritmo individual

1. El mapa mantiene tres estados: desconocido, libre observado y ocupado.
2. Una celda libre es frontera si alguna de sus ocho vecinas es desconocida.
3. Desde la celda del robot se expande una onda de costo uniforme por las
   cuatro vecinas cardinales libres.
4. La primera frontera alcanzada es el objetivo. Por eso la elección minimiza
   distancia navegable en el grid y no distancia euclidiana ni distancia al
   centroide de un cluster.
5. Dijkstra calcula la ruta al objetivo. Se vuelve a seleccionar al acercarse,
   al invalidarse la ruta o al cumplirse la cadencia de replanning.

El port añade dos guardas propias del host: descarta objetivos recientemente
alcanzados/fallidos y confirma que el costmap real pueda alcanzarlos. El orden
de la onda entre candidatos válidos no cambia.

## Coordinación multirobot

1. Se insertan simultáneamente las posiciones de todos los robots en una cola
   de prioridad.
2. Cada onda lleva el identificador de su robot. Cuando dos ondas se encuentran,
   la celda queda en la región del robot que llegó primero; esto genera una
   partición de Voronoi por distancia navegable.
3. Cada robot toma la primera frontera dentro de su región.
4. Si su región no tiene una frontera utilizable, una segunda onda puede cruzar
   regiones ajenas (`mWaitForOthers = false` en Nav2D).
5. Los objetivos ya asignados se reservan para impedir duplicados exactos en la
   ejecución centralizada.

El `tutorial4.launch` histórico únicamente da `Operator`/`Navigator` al segundo
robot; el primero aporta scans al mapa compartido. El preset multirobot conserva
las dos poses y el mapa de ese tutorial, pero activa la exploración en ambos
robots para ejecutar y medir el `MultiWavefrontPlanner` que también viene en el
repositorio. El preset individual es la reproducción directa del tutorial 3.

## Cómo ejecutar

1. Inicia el simulador con `python main.py` desde la raíz del proyecto.
2. Abre el menú de la barra superior y elige **Load .sim…**.
3. Para el experimento original individual, carga
   `examples/nav2d_tutorial3_single.sim`.
4. Para la extensión coordinada, carga
   `examples/nav2d_tutorial4_multi.sim`.
5. Presiona **Start Simulation**. No cambies mapa, resolución, sensor, radios,
   planificador o posiciones si quieres conservar la réplica.

## Protocolo de comparación

Ejecuta cada escenario al menos cinco veces con velocidad de simulación 1.00×.
Detén la corrida cuando `Free-space coverage` deje de aumentar y todos los
robots permanezcan en `HOLD`, o al llegar a 900 s simulados. Desde **Metrics**
registra:

- tiempo de simulación;
- `Free-space coverage`;
- distancia total recorrida;
- solicitudes de planeación y replans de exploración/seguridad;
- celdas revisitadas y razón de revisita;
- en múltiple, celdas y razón de solapamiento entre robots.

Para una comparación A/B justa, duplica el `.sim` y cambia solamente el
selector de exploración o el coordinador. La geometría, sensor, resolución,
cinemática y poses iniciales deben permanecer idénticos.

## Diferencias de plataforma

Esta es una réplica de comportamiento y protocolo, no una ejecución binaria
del stack ROS. El simulador usa su propio mapa de creencia y controlador; no
modela incertidumbre de SLAM, transformaciones TF, tópicos ROS ni el operador
reactivo `nav2d_operator`. El escenario sí conserva los factores que gobiernan
la selección de fronteras y la asignación: geometría, escala, estados del grid,
inflación letal, rango, poses, Dijkstra y cadencia de replanning.

