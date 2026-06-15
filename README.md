# Hopfield vs Hebbian Pattern Dataset

Proyecto base en Python para comparar una Red de Hopfield y una Memoria Hebbiana en reconocimiento de patrones.

Por ahora el repositorio incluye solo la estructura inicial y el módulo de generación del dataset de letras A-Z.

## Estructura

- `dataset/`: archivos generados en formato `.npz`
- `src/data/`: generación, carga y visualización del dataset
- `src/models/`: futuras implementaciones de Hopfield y Hebb
- `src/experiments/`: experimentos reproducibles
- `src/utils/`: utilidades compartidas
- `notebooks/`: ejemplos y exploración
- `results/figures/`: figuras generadas
- `results/tables/`: tablas generadas
- `reports/`: reportes y documentación

## Dependencias

- numpy
- pillow
- matplotlib

## Uso rápido

El notebook de ejemplo en `notebooks/` genera el dataset completo, lo visualiza y lo guarda como `.npz`.