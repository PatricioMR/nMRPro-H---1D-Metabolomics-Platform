# nMRPipe — NMRAnalysis_Gui

Aplicação desktop (Python/Tkinter) para análise de espectros de RMN 1H em
metabolómica e lipidómica: importação, deteção de picos, anotação contra bases
HMDB/BMRB, deconvolução de sobreposição (**JG-Deconv**) e quantificação com
estatística entre grupos.

---

## Funcionalidades

- **Importação** de espectros (CSV/TXT, múltiplas amostras) com identificação de brancos (BLK).
- **Deteção de picos** com correção de baseline e estimativa de SNR.
- **Anotação** contra bases SQLite locais (HMDB por biofluido, BMRB), com
  *scoring* composto, validação por padrão-J (ΔJ) e multiplicidade.
- **JG-Deconv** (*J-Guided Overlap Deconvolution*): resolve a contribuição de
  cada metabolito em picos sobrepostos via cascata corrigida por razão-J + NNLS
  global, com âncoras de concentração e frações por amostra.
- **Quantificação**: integração (trapézio / multipleto / Lorentz), correção de
  protões, normalização (IS, PQN), correção de sobreposição via JG-Deconv,
  com SNR e incerteza sempre reportados.
- **Estatística** entre grupos e **relatório PDF**.
- **Projetos** guardáveis (load/save) com **auto-save** a cada 5 min.

---

## Arquitetura

```
main.py            → ponto de entrada
gui/               → interface (Tkinter) e stages do workflow
  stages/          → visualization, peak_detection, annotation,
                     j_resolve (JG-Deconv), quantification, statistics, MCR
  widgets/         → janelas auxiliares (resultados, PDF, pathways, …)
data/              → loaders, acesso a DB (HMDB/BMRB), API externa
processing/        → DSP: baseline, deteção, alinhamento, integração, qualidade
plotting/          → visualização de espectros
controllers/       → orquestração / lógica de negócio
utils/             → helpers, workers assíncronos
databases/         → bases SQLite (read-only em runtime)
tests/             → unit / integration / smoke (pytest)
```

Convenções científicas: eixo δ (ppm) **sempre invertido**; unidades visíveis
nos plots; erros de parsing/deteção propagados ao utilizador, nunca silenciados.

---

## Workflow

1. **Importar** espectros e definir grupos/amostras.
2. **Detetar picos** (baseline + SNR).
3. **Anotar** contra a base do biofluido (HMDB/BMRB).
4. **JG-Deconv** — resolver picos sobrepostos e aplicar frações à quantificação.
5. **Quantificar** (com correção JG-Deconv opcional).
6. **Estatística** + exportar relatório.

---

## Requisitos

- Python 3.12+ (desenvolvido/testado em 3.14)
- Dependências:

```bash
pip install numpy scipy matplotlib pandas reportlab requests pillow
```

`sqlite3` e `tkinter` vêm com a instalação padrão do CPython.

### Bases de dados

As bases SQLite ativas estão em `databases/`. São **read-only** em runtime.
Backups/versões antigas (`databases/backup_*`, `*.bak*`) ficam fora do controlo
de versões.

---

## Executar

```bash
python main.py
```

---

## Testes

```bash
pytest            # ou: run_tests.bat (Windows)
```

A lógica em `processing/` tem testes unitários em `tests/unit/`; pipelines
completos em `tests/integration/` (usam DB de teste real, não mockada).

---

## Notas de desenvolvimento

- Esquemas SQLite são contratos — alterações exigem migração explícita; queries
  sempre parametrizadas.
- Funcionalidades que alteram estado persistente devem ser cobertas pelo
  save/load do projeto.
- Ver [CLAUDE.md](CLAUDE.md) para as convenções de arquitetura e integridade
  científica seguidas no projeto.
