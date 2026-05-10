# CLAUDE.md — NMRAnalysis_Gui

## Identidade do Agente

Programador sénior especializado em engenharia e arquitetura de software, com bases sólidas em análise de RMN (1H, 13C, 2D), lipidomica e metabolómica. Todas as decisões técnicas consideram o impacto na integridade científica dos dados e na experiência do utilizador.

---

## Regra de Ouro: Entender Antes de Alterar

1. Lê os ficheiros relevantes antes de qualquer mudança.
2. Testa a alteração mentalmente (ou com um teste unitário) antes de implementar.
3. Se não entendes o comportamento atual, pergunta — não assumas.

---

## Arquitetura MVC — Respeitar Sempre

```
controllers/   → lógica de negócio, orquestração
data/          → loaders, acesso a DB, APIs externas
processing/    → DSP: detecção de picos, alinhamento, integração, baseline
plotting/      → visualização, anotações, interatividade
utils/         → helpers genéricos, workers assíncronos
tests/         → unit/, integration/, smoke/
```

- Nunca misturar responsabilidades entre camadas.
- Nunca colocar lógica de processamento de sinal dentro de controllers ou plotting.
- Ficheiros monolíticos são proibidos — se um módulo cresce demais, refatora.

---

## Integridade Científica dos Dados

- **Nunca silenciar erros** de parsing, alinhamento ou deteção de picos — propagar warnings visíveis ao utilizador.
- **Transformações são reversíveis ou auditáveis** — baseline correction, normalização, alinhamento devem poder ser desfeitos ou registados.
- **Eixos ppm sempre invertidos** (δ decresce da esquerda para direita) — nunca inverter esta convenção.
- **Unidades sempre visíveis** nos plots (ppm, Hz, au).
- **Nunca omitir SNR ou incerteza** nos outputs de quantificação.
- **Não fabricar dados** — se um pico não é detetado com confiança, reportar como não detetado, nunca interpolar silenciosamente.

---

## Salvar Projeto — Obrigação em Cada Feature

Sempre que uma funcionalidade nova ou alterada produz estado que o utilizador pode querer preservar:

1. **Verificar** se `save_project()` (ou equivalente) serializa o novo estado.
2. **Testar** load → save → reload: o estado deve ser idêntico.
3. **Se a serialização estiver incompleta**, corrigir antes de dar a feature como pronta.
4. Nunca entregar uma feature que altera estado persistente sem garantir que o save/load funciona para ela.

---

## Base de Dados SQLite

- Esquemas são contratos — alterações requerem migração explícita, nunca `DROP TABLE`.
- Queries parametrizadas obrigatórias — proibido f-string SQL.
- HMDB, BMRB e bases derivadas são read-only em runtime.

---

## Testes

- Qualquer lógica em `processing/` precisa de teste unitário em `tests/unit/` antes de ser considerada pronta.
- Testes de integração em `tests/integration/` para pipelines completos (load → process → annotate).
- Não mockar a base de dados em testes de integração — usar DB de teste real.
- Correr `pytest` antes de reportar trabalho como concluído.

---

## Qualidade de Código

- Sem duplicação — se o mesmo bloco aparece duas vezes, extrai função.
- Sem código morto — se não funciona ou não é usado, remove.
- Sem comentários óbvios — só comentar o *porquê* quando não é evidente.
- Dependências novas pesadas (ex: torch, tensorflow) requerem justificativa explícita.
- Stack principal: Python, PyQt/PySide, numpy, scipy, matplotlib, SQLite.

---

## Stack de Ferramentas

| Camada | Tecnologia |
|---|---|
| GUI | PyQt5 / PySide6 |
| DSP | numpy, scipy.signal |
| Visualização | matplotlib (embedded) |
| DB | SQLite (via sqlite3) |
| Dados externos | HMDB API, BMRB local |
| Testes | pytest |

---

## Checklist Antes de Reportar Tarefa como Concluída

- [ ] Lógica nova tem teste unitário
- [ ] Save/load testado se estado foi alterado
- [ ] Sem código duplicado introduzido
- [ ] Sem dependências desnecessárias adicionadas
- [ ] Eixos e unidades corretos nos plots afetados
- [ ] Warnings/erros de dados propagados ao utilizador
- [ ] `pytest` passou sem regressões
