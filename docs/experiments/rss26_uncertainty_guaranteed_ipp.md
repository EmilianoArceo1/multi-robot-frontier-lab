# Réplica RSS 2026: planificación informativa con garantía de incertidumbre

Esta integración reproduce el experimento principal de
[*Informative Path Planning with Guaranteed Estimation Uncertainty*](https://arxiv.org/abs/2602.05198)
(arXiv `2602.05198v3`, RSS 2026) usando el
[repositorio oficial](https://github.com/itskalvik/uncertainty-guaranteed-ipp)
fijado al commit `f387c57bcaa61bf218d26e63212bf789ef42a534`.

## Alcance correcto

El artículo no propone exploración de frontiers ni coordinación multirobot.
Considera un robot que conoce el dominio, toma muestras de un campo escalar y
elige una trayectoria que reduce la varianza posterior de un proceso gaussiano
por debajo de un umbral. El trabajo multirobot aparece como trabajo futuro.

Por eso esta integración tiene tres capas separadas:

1. **Benchmark fiel:** ejecuta `benchmark.py` oficial con Attentive GP y
   SGP-Tools, sin reimplementar sus optimizadores.
2. **Exportación observacional:** una copia temporal del script, creada después
   de verificar commit y dataset, guarda los arrays que el script oficial sólo
   dibuja. No modifica el algoritmo ni el repositorio de referencia.
3. **Adaptador del simulador:** representa varianza, ruta piloto, sitios, FoVs y
   tour; luego entrega el tour completo al controlador de waypoints sin
   colapsar sus sitios de medición.

## Contrato matemático implementado localmente

El núcleo NumPy de `algorithms/uncertainty_guaranteed_ipp/` implementa un test
ligero del criterio del Teorema 1. Para un candidato `c` y punto de evaluación
`v`, la matriz binaria marca cobertura cuando

```text
|k(c,v)| >= sqrt((k(v,v) - sigma_target²) * (k(c,c) + sigma_noise²)).
```

Incluye `GreedyCover`, `GCBCover` con inserción más cercana/presupuesto y un
certificado que verifica tanto la unión conservadora del teorema como la
varianza posterior conjunta exacta sobre el conjunto finito evaluado.

Ese núcleo usa un kernel RBF estacionario y sirve para probar la integración;
**no reemplaza** al kernel Attentive aprendido del benchmark publicado.

## Prueba rápida dentro del simulador

1. Inicia la aplicación.
2. Carga `examples/rss26_ipp_rbf_smoke.sim`.
3. Presiona **Start Simulation**.

Se verá la varianza posterior, el piloto discontinuo verde, el tour rojo y los
sitios amarillos. El preset fija semilla `1234`, objetivo `0.5`, 29 sitios y un
máximo posterior de aproximadamente `0.3043`. Su metadata dice
`integration_smoke_not_paper_benchmark`; no debe citarse como resultado del
artículo.

Para regenerarlo:

```powershell
python -m experiments.rss26_ipp.generate_rbf_smoke_bundle `
  --output examples/rss26_ipp_rbf_smoke
```

## Benchmark fiel

Las versiones declaradas por los autores están en
`experiments/rss26_ipp/requirements-paper.txt`:

```text
sgptools==2.0.7
tensorflow==2.19.1
tensorflow-probability==0.25.0
```

Instálalas en un entorno Python aislado. Luego verifica la referencia local sin
ejecutar TensorFlow:

```powershell
python -m experiments.rss26_ipp.runner `
  --reference "D:\Texas A&M\uncertainty-guaranteed-ipp_reference" `
  --dataset N47W124.npy `
  --verify-only
```

La verificación exige el commit fijado, la semilla oficial y el SHA-256 exacto
del dataset. Para ejecutar el barrido principal publicado:

```powershell
python -m experiments.rss26_ipp.runner `
  --reference "D:\Texas A&M\uncertainty-guaranteed-ipp_reference" `
  --dataset N47W124.npy `
  --kernel Attentive `
  --variance-ratios 0.9 0.8 0.7 0.6 0.5 `
  --methods HexCover GreedyCover GCBCover GCBCover-Dist ContinuousSGP `
  --output runs/rss26_ipp
```

El comando conserva los parámetros del código oficial:

- 350 muestras sobre la ruta piloto;
- 5000 muestras de entrenamiento;
- 15 puntos inductores;
- grid de evaluación `100 × 100`;
- semilla NumPy/TensorFlow `1234`;
- presupuesto de `GCBCover-Dist` igual a la distancia de `GCBCover` sin
  restricción menos 20 m;
- límite de 170 sitios para `ContinuousSGP`, según el script oficial.

La ejecución completa entrena varios GP y puede tardar considerablemente. Los
resultados quedan en:

```text
runs/rss26_ipp/N47W124/Attentive/
  results.json
  normalized_results.json
  bundles/<método>_ratio<r>/
    data.npz
    manifest.json
    scenario.sim
```

Cada `scenario.sim` se puede cargar directamente en la aplicación. Sus assets
son relativos y están confinados al mismo directorio. La transformación al
mundo `[-10,10] × [-8,8]` es uniforme y centrada: preserva relación de aspecto
y usa letterboxing si el dataset no coincide con el aspecto del canvas.

## Protocolo de comparación

Para cada ratio y método registra:

- máxima varianza posterior;
- MSE y SMSE;
- tiempo del planificador;
- número de sitios de medición;
- longitud de trayectoria.

El umbral es `ratio × máxima varianza previa`. Una réplica válida debe conservar
dataset, commit, kernel, semilla, grid y conjunto de métodos. El simulador añade
sólo ejecución cinemática y visualización; las métricas de optimización se toman
del benchmark oficial, antes de escalar coordenadas al canvas.

## Límites

- Los ensayos físicos ASV/AUV del artículo no se reproducen aquí: el repositorio
  oficial no publica toda la plataforma, telemetría y condiciones de campo.
- La garantía es condicional al modelo GP y al conjunto finito de evaluación;
  no garantiza MSE físico bajo error de modelo/localización.
- El controlador puede desviarse del segmento ideal. Los sitios se preservan,
  pero una desviación física debe analizarse aparte de la garantía del plan.
- Cualquier extensión multirobot sería trabajo nuevo, no una réplica del paper.
