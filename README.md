# GenAI Research Lab

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B1020,45:4F46E5,100:00D4FF&height=230&section=header&text=GenAI%20Research%20Lab&fontSize=44&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=Experimentar%20%7C%20Testar%20%7C%20Avaliar%20%7C%20Aplicar&descAlignY=58&descSize=18" width="100%" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=19&duration=2800&pause=700&color=00D4FF&center=true&vCenter=true&width=920&lines=Testar+IA+generativa+antes+de+usar+a+resposta;Um+caso+agr%C3%ADcola%2C+doze+experimentos%2C+resultado+gravado;O+modelo+fica+no+texto+ou+completa+a+hist%C3%B3ria%3F;Python+%7C+Gemini+%7C+RAG+%7C+AgroGenAI" alt="Tema do laboratório" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Gemini-GenAI-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Experimentos-12-6D5DFB?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Testes-unittest-00C853?style=for-the-badge" />
</p>

---

# 🎯 Para que serve

Este laboratório serve para testar IA generativa num problema agrícola e guardar o que aconteceu.

A situação é sempre a mesma: uma área com queda de produtividade. Em cima dela eu fui verificando, uma peça por vez, se o modelo ajuda uma análise ou se só escreve uma resposta bonita. A peça só entra na etapa seguinte depois que o teste mostra o que ela faz, inclusive o erro.

O uso que eu tenho em mente é este. Chega um relatório de campo, uma nota ou uma planilha. O sistema lê o que está no arquivo, escolhe o trecho certo, devolve um registro que outra pessoa consegue usar e avisa quando o dado não fecha. Ele não substitui quem conhece a lavoura.

Os números de soja, milho e algodão são inventados. Estão aqui para a ferramenta ter o que responder. Não são dados de fazenda, e nada neste repositório é diagnóstico.

Dá para ler o teste inteiro nesta página. Não precisa rodar o código.

---

# 🧭 Como o laboratório foi montado

```text
um problema agrícola
        ↓
uma peça só por experimento
        ↓
o resultado gravado em results/
        ↓
a peça seguinte usa o que a anterior mostrou
        ↓
AgroGenAI junta busca, análise e revisão humana
```

| O que eu queria saber | Onde isso foi testado |
|---|---|
| A forma do pedido muda a resposta? | 01 |
| A resposta vira dado, ou continua texto solto? | 02 e 11 |
| Modelo mais novo é sempre melhor? | 03 e 04 |
| A busca entrega o parágrafo da lavoura? | 05 a 08 |
| O modelo pede o número ou inventa? | 09 e 10 |
| O fluxo inteiro sabe parar e chamar uma pessoa? | 12 |

---

# 🧪 Os 12 experimentos

| # | Etapa | Resultado que ficou |
|:--:|---|---|
| 01 | Prompt com exemplo e sem exemplo | O pedido com exemplo exige causas, evidências e ações |
| 02 | JSON validado | Causas, evidências e ações em lista, sem texto solto |
| 03 | Dois modelos, o mesmo prompt | 3.7 Flash foi mais rápido e escreveu mais |
| 04 | Rubrica de 1 a 5 | 30/30 nos dois. O juiz, outro modelo, apontou o 3.7 |
| 05 | Busca por palavra | O 1º lugar foi o parágrafo do escritório, nota 4 |
| 06 | Busca por sentido | O 1º lugar foi déficit hídrico, similaridade 0,784 |
| 07 | A mesma busca numa matriz | A ordenação do embedding, pronta para consultar |
| 08 | Resposta com o contexto do embedding | O modelo lê a lavoura, não o escritório |
| 09 | Ferramenta antes do texto | 3.200 kg/ha e estresse hídrico moderado no Sul |
| 10 | Agente com teto de etapas | 2 etapas, as duas consultas, e a ressalva de dado simulado |
| 11 | Relatório vira planilha | Prioridade média e revisão humana marcada |
| 12 | AgroGenAI | Análise de soja com revisão humana verdadeira |

---

# 🌾 Etapa por etapa

## 01 · Comparação de prompt

**O teste.** A mesma queda de produtividade vai duas vezes ao modelo. Na primeira não há exemplo. Na segunda há dois casos agrícolas já analisados, e o pedido exige três blocos: causas, evidências e ações.

**Para que serve.** A instrução é a primeira alavanca. Sem exemplo, o modelo escolhe o formato. Com exemplo, a resposta já nasce no formato de uma análise.

**O resultado.** O que muda de um para o outro é o contrato da resposta, não um número mágico de qualidade. O script mede o tempo das duas chamadas e grava as duas saídas. Few-shot só vale se o exemplo for do tipo de análise que a gente quer repetir.

## 02 · Saída estruturada

**O teste.** O modelo tem de preencher um schema: problema, causas, evidências e ações. Se não couber no schema, não passa.

**Para que serve.** Texto livre não entra numa fila, numa planilha ou num próximo sistema. JSON entra.

**O resultado.** O problema registrado foi a queda de produtividade. Entre as causas apareceu estresse hídrico, praga, solo e falha de manejo. Entre as ações, análise de solo, vistoria e revisão do clima e da irrigação. A resposta deixou de ser um parágrafo e virou lista.

## 03 · Comparação de modelos

**O teste.** Gemini 3.8 Flash e Gemini 3.7 Flash receberam o mesmo pedido: causas, dados a investigar, ações e como a IA ajudaria, sem inventar número da propriedade.

**Para que serve.** Trocar de modelo é decisão de engenharia. Nome mais novo não basta.

**O resultado.**

| Modelo | Tempo | Tamanho |
|---|--:|--:|
| Gemini 3.8 Flash | 77,74 s | 4.130 caracteres |
| Gemini 3.7 Flash | 59,43 s | 4.922 caracteres |

Os dois separaram hipótese de fato. Nesta execução o 3.7 foi mais rápido e mais longo. Tempo, sozinho, não escolhe o modelo. Por isso existe a etapa 04.

## 04 · Avaliação

**O teste.** Uma rubrica de 1 a 5 lê as duas respostas. Os critérios são clareza, completude, aderência ao pedido, qualidade, aplicabilidade e cuidado ao marcar hipótese. Nota fora de 1–5 quebra a execução.

**Para que serve.** Resposta gerada não é resposta confiável. Alguém, ou alguma regra, precisa olhar.

**O resultado.** Os dois modelos ficaram com 30/30. O avaliador indicou o 3.7 Flash porque a resposta desceu mais ao campo: amostragem, análise de tecido, trincheira de raiz. Eu deixo essa nota no registro com o limite junto: quem deu a nota também foi um modelo. Nota máxima dos dois lados não encerra a escolha.

## 05 · RAG por palavra

**O teste.** A pergunta é única, e também é a dos experimentos 06, 07 e 08:

> Como a falta de chuva reduz o que a planta produz?

A busca conta palavras iguais e manda ao modelo só os parágrafos que mais coincidem. A resposta tem de ficar nesse contexto.

**Para que serve.** É o jeito simples de “usar um documento”. Funciona quando o texto repete a pergunta. Falha quando o texto certo usa outras palavras.

**O resultado.**

| Lugar | Nota | Trecho |
|:--:|:--:|---|
| 1 | 4 | Escritório: a falta de chuva no calendário reduz o que a equipe administrativa produz. Não descreve a lavoura. |
| 2 | 1 | Lavoura: períodos secos, menos água no solo, menos grãos, colheita menor. |

Se a resposta fosse gerada aqui, o modelo leria o escritório primeiro.

## 06 · Embeddings

**O teste.** A mesma pergunta e a mesma base. Cada parágrafo vira um vetor com `gemini-embedding-2`. A ordem é a do cosseno, a proximidade de sentido.

**Para que serve.** A busca deixa de depender da palavra copiada. “Falta de chuva” pode encontrar “déficit hídrico”.

**O resultado.** O escritório, que tinha ganhado por palavra, não entrou nos três primeiros.

| Lugar | Similaridade | Trecho |
|:--:|:--:|---|
| 1 | 0,784 | Disponibilidade de água e déficit hídrico reduzindo a produtividade |
| 2 | 0,756 | Solo seco, menos grãos, colheita menor |
| 3 | 0,713 | Água, solo, pragas, clima, manejo e nutrientes |

```text
a mesma pergunta
     ├── palavra   →  escritório
     └── embedding →  água e colheita
```

Esse é o resultado que eu guardo do laboratório. O método de busca muda o texto que o modelo tem permissão de usar.

## 07 · Busca vetorial

**O teste.** A conta do experimento 06, com os vetores numa matriz NumPy.

**Para que serve.** Separar “transformar o texto em vetor” de “consultar esses vetores”. A matriz é o índice. A pergunta nova só precisa de um vetor para ser comparada com ele.

**O resultado.** A ordem é a mesma do embedding. O que este experimento acrescenta é o índice: a base consultável, não uma lista refeita à mão a cada pergunta.

## 08 · RAG com embeddings

**O teste.** O mesmo desenho do experimento 05, trocando a origem do contexto. Entram os parágrafos do embedding, não os da palavra.

**Para que serve.** Mostrar a consequência prática da tabela acima. Quem lê o contexto do 05 lê o escritório. Quem lê o contexto do 08 lê seca, água e colheita.

**O resultado.** A geração desta etapa usa `gemini-3.6-flash`. O 3.8 respondeu 503 na hora do teste, e o cliente tenta de novo nesses casos antes de desistir. O modelo da resposta importa menos, aqui, do que o parágrafo que chegou até ele.

## 09 · Tool calling

**O teste.** O modelo ganha duas ferramentas com dados inventados: produtividade por cultura e região, e clima da região. A pergunta pede a soja no Sul.

**Para que serve.** Número de lavoura não deve sair da memória do modelo. Se o dado existe numa função, a função é quem responde.

**O resultado.** Ele chamou as duas ferramentas antes de escrever.

| Consulta | Retorno |
|---|---|
| Soja, Sul | 3.200 kg/ha |
| Clima, Sul | 820 mm · 22,4 °C · estresse hídrico moderado |

A resposta aponta o estresse moderado como o ponto a investigar e avisa que o número não é de uma propriedade.

## 10 · Agente

**O teste.** O modelo decide se chama ferramenta, lê o retorno e pode chamar de novo. O teto é 4 etapas. Se passar disso sem resposta, o ciclo para.

**Para que serve.** Uma chamada só não é um fluxo. O fluxo precisa de parada. Sem teto, o agente pode ficar pedindo ferramenta.

**O resultado.** Usou 2 etapas, dentro do teto. Na primeira, chamou produtividade e clima. Os retornos foram os mesmos: 3.200 kg/ha, 820 mm, 22,4 °C, estresse moderado. A análise final trata o estresse como alerta de distribuição de chuva e fecha dizendo que o dado é simulado.

## 11 · Automação

**O teste.** Um relatório de campo, em texto corrido, tem de sair como registro: categoria, prioridade, resumo, fatores, ação e se uma pessoa precisa revisar. O script grava JSON e CSV.

**Para que serve.** É o ponto em que a IA entra numa rotina. O próximo passo não é um leitor. É uma planilha, uma fila, um responsável.

**O resultado.**

| Campo | Valor |
|---|---|
| Categoria | Produtividade agrícola |
| Prioridade | Média |
| Revisão humana | Sim |
| Fatores | chuva irregular, déficit hídrico, temperatura alta em alguns dias, sinais no solo |
| Ação | Cruzar histórico, solo, clima, manejo e pragas antes de decidir |

O resumo não inventa a causa. Diz que o relatório não tem dado para fechá-la. A revisão humana ficou marcada por isso.

## 12 · AgroGenAI

**O teste.** As peças anteriores no mesmo relatório de soja: produtividade abaixo do histórico, chuva irregular, possível déficit hídrico, temperatura acima do padrão e sinais no solo. O relatório já diz que isso não basta para nomear a causa.

**Para que serve.** É o protótipo. Não é um produto de diagnóstico. Mostra o fluxo que eu levaria para um relatório real: buscar o trecho, analisar sem sair dele e devolver o que ainda depende de uma pessoa.

```text
┌──────────────────────────────┐
│      Relatório de soja       │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│     Busca por embedding      │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│      Trechos da base         │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│            Gemini            │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│   Análise + revisão humana   │
└──────────────────────────────┘
```

**O resultado.**

| Campo | O que saiu |
|---|---|
| Trecho mais próximo | Histórico, meteorologia e manejo · similaridade 0,795 |
| Categoria | Produtividade agrícola / fatores climáticos e de solo |
| Prioridade | Média |
| Revisão humana | Sim |

O próprio resultado lista o risco: sem vistoria, clima vira causa falsa. A recomendação é coletar chuva, temperatura, solo, manejo e ocorrência de praga. Não é uma ordem de campo.

---

# ⚙️ O que segura os testes

| Módulo | Por que está ali |
|---|---|
| `src/gemini_client.py` | Uma chave, geração, embedding e nova tentativa em 503 ou 429 |
| `src/retrieval.py` | Corte da base, nota por palavra e cosseno |
| `src/agri_data.py` | Tabelas inventadas. `Algodão` e `Centro Oeste` acham a linha |
| `src/config.py` | Caminhos, modelos e a pergunta dos experimentos 05 a 08 |
| `src/results.py` | O JSON cai em `results/` de qualquer pasta |

```text
genai-research-lab/
├── data/            o texto agrícola dos testes
├── experiments/     os 12 testes e o testar.py
├── results/         o registro de cada execução
├── src/             cliente, busca e tabelas
└── tests/           a checagem que não gasta API
```

Dependências: `google-genai`, `python-dotenv`, `pydantic`, `numpy`.

---

# ▶️ Para repetir

O registro já está nesta página. Os comandos servem para rodar de novo.

```bash
git clone https://github.com/carinasouzasants-dev/genai-research-lab.git
cd genai-research-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```env
GEMINI_API_KEY=your_api_key_here
```

O `.env` fica na raiz e não entra no Git.

```bash
python -m unittest tests.test_lab -v
python experiments/testar.py
python experiments/agrogenai_12.py
```

No `testar.py`, Enter usa a pergunta do laboratório e `sair` encerra. A busca por palavra aparece na hora. A resposta `s` chama o Gemini. O experimento 03 precisa ter rodado antes do 04.

---

# 🔐 O que este teste não faz

A base é curta, e um parágrafo foi escrito para repetir as palavras da pergunta. Foi assim que a diferença entre as buscas ficou visível. O juiz do experimento 04 também é um modelo. Cada JSON é uma execução, e muda se o script rodar outra vez.

O AgroGenAI não fecha causa de lavoura. Ele organiza o relatório, mostra de qual trecho a análise saiu e para quando falta gente no meio.

---

# 👩🏻‍💻 Autora

**Carina Souza**  
Estudante de engenharia de software. Montei este laboratório para aprender IA generativa do jeito que eu quero usar: testar a peça, gravar o resultado e só então colocar a peça num fluxo agrícola, com revisão humana no fim.

<p align="center">
  <a href="https://github.com/carinasouzasants-dev">
    <img src="https://img.shields.io/badge/GitHub-Carina%20Souza-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
  <a href="https://www.linkedin.com/in/carina-souza-dev">
    <img src="https://img.shields.io/badge/LinkedIn-Carina%20Souza-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:00D4FF,50:4F46E5,100:0B1020&height=130&section=footer" width="100%" />
</p>

<p align="center">
  <strong>Experimentar. Testar. Avaliar. Aplicar.</strong>
</p>
