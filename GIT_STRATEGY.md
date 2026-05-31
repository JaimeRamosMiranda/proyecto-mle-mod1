# Estrategia de Git — proyecto-mle-mod1

## Modelo de ramas: GitHub Flow

Se adoptó **GitHub Flow** por su simplicidad y adecuación al tamaño del equipo y del proyecto.

```
main ──────●──────────────────────────────●── v1.0.0
            \                            /
development  ●────●────●────●────●──────●
                  │         │
               commits   pull request
```

### Ramas

| Rama | Propósito |
|---|---|
| `main` | Código estable y revisado. Solo recibe cambios via pull request desde `development`. |
| `development` | Rama de trabajo activo. Aquí se desarrollan y prueban los cambios antes de integrarlos. |

### Flujo de trabajo

1. Todo el trabajo nuevo se hace en `development`.
2. Al completar un bloque funcional se abre un **Pull Request** de `development` → `main`.
3. El PR se revisa, se documenta con los cambios realizados y se fusiona con **merge commit** (`git merge --no-ff`).
4. Al finalizar el proyecto se crea el **Release v1.0.0** desde `main`.

### Convención de commits (estilo Conventional Commits)

```
<tipo>: <descripción corta en imperativo>
```

| Tipo | Uso |
|---|---|
| `feat` | Nueva funcionalidad o notebook |
| `fix` | Corrección de errores |
| `docs` | Cambios en README u otra documentación |
| `refactor` | Mejora de código sin cambio funcional |
| `data` | Adición o modificación de datos |
| `chore` | Tareas de mantenimiento (.gitignore, requirements) |

**Ejemplos:**
```bash
git commit -m "feat: add preprocessing notebook with Mahalanobis outlier detection"
git commit -m "feat: add k-means clustering with k=4 and PCA visualization"
git commit -m "docs: complete README with model card and results"
git commit -m "fix: recalculate perfil_bin inside export cell"
git commit -m "chore: add requirements.txt and .gitignore"
```

### Pull Requests

Cada PR debe incluir en su descripción:
- **Qué se hizo:** resumen de los cambios
- **Por qué:** motivación o decisión técnica
- **Cómo probarlo:** pasos para verificar que funciona

### Release v1.0.0

Al finalizar el proyecto se crea el release desde GitHub:

```
Repositorio → Releases → Draft a new release
Tag: v1.0.0
Target: main
Title: v1.0.0 — Análisis de Fenotipos de Pacientes con Insuficiencia Cardíaca
```

Las notas del release deben incluir qué contiene la versión, métricas obtenidas y estructura del repositorio.
